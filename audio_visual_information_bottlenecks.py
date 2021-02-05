import gtn
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from models import TDS
from transducer import ConvTransduce1D

class AudioVisualInformationBottleneck(torch.nn.Module):
  '''
  Information bottleneck model for multimoda phone discovery.

  Args:
      input_size: int, dimension of the input features
      output_size: int, number of word types 
      hidden_size: int, number of phone types
  '''
  def __init__(self, input_size, output_size, 
               in_token_size, out_token_size,
               tds_groups, kernel_size, 
               stride, dropout, wfst=False, 
               beta=1., **kwargs): 
    super(AudioVisualInformationBottleneck, self).__init__()
    self.bottleneck = TDS(input_size,\
                          in_token_size,\
                          tds_groups,\
                          kernel_size,\
                          dropout) 

    if wfst: # TODO Add normalization; allow phoneme inputs
      lexicon = [tuple(5 * token_idx + i for i in range(5))
                 for token_idx, token in enumerate(out_token_size)]
      self.transducer = ConvTransduce1D(
          lexicon, kernel_size, stride, blank_idx, **kwargs) 
    else:
      self.transducer = torch.nn.Conv1d(
          in_channels=in_token_size,
          out_channels=out_token_size,
          kernel_size=kernel_size,
          padding=kernel_size // 2,
          stride=stride)
    self.wfst = wfst 
    
    self.predictor = nn.Linear(output_size, out_token_size)
    
    # Prior
    self.in_logit_prior = nn.Parameter(torch.empty(in_token_size))
    self.out_logit_prior = nn.Parameter(torch.empty(out_token_size))
    nn.init.uniform_(self.in_logit_prior, a=-1, b=1)
    nn.init.uniform_(self.out_logit_prior, a=-1, b=1)

    self.beta = beta

  def forward(self, audio_inputs, image_inputs):
    # Inputs shape: [B, H, W]
    in_scores = self.bottleneck(audio_inputs)
    
    # Input token scores shape: [B, W, in_token_size]
    in_probs = F.softmax(in_scores, dim=-1)

    # Output token scores shape: [B, W, out_token_size]
    if self.wfst:
      out_scores = self.transducer(in_scores) 
      out_probs = out_scores
    else:
      out_scores = self.transducer(in_scores.permute(0, 2, 1)).permute(0, 2, 1)
      out_probs = F.softmax(out_scores, dim=-1) 

    # Predictor scores shape: [B, L, out_token_size]
    pred_scores = self.predictor(image_inputs)

    # Attention scores shape: [B, W, L]
    att_scores = torch.matmul(out_probs, pred_scores.permute(0, 2, 1))
    att_first = F.softmax(att_scores, dim=-1) # TODO Apply mask  
    att_second = F.softmax(att_scores, dim=-2)

    # Outputs shape: [B,]
    outputs = ((att_first + att_second) * att_scores).sum(dim=-1).sum(dim=-1)  

    return in_scores,\
           in_probs,\
           out_scores,\
           out_probs,\
           pred_scores,\
           outputs

  def calculate_loss(self, audio_inputs, image_inputs):
    n = audio_inputs.size(0)
    m = nn.LogSoftmax(dim=-1) 
    device = audio_inputs.device
    S = torch.zeros((n, n), dtype=torch.float, device=device)
    beta = torch.tensor(self.beta, device=audio_inputs.device)
    
    in_log_posteriors = []
    in_posteriors = []
    out_log_posteriors = []
    out_posteriors = []
    for i in range(n):
      for j in range(n):
        in_scores,\
        in_probs,\
        out_scores,\
        out_probs,\
        pred_scores,\
        outputs = self(audio_inputs[i].unsqueeze(0),\
                       image_inputs[j].unsqueeze(0))
        S[i][j] = outputs.squeeze(0)
    
        in_log_posteriors.append(m(in_scores))
        in_posteriors.append(in_probs)
        out_log_posteriors.append(m(out_scores))
        out_posteriors.append(out_probs)

    in_log_posteriors = torch.cat(in_log_posteriors)
    in_posteriors = torch.cat(in_posteriors)
    out_log_posteriors = torch.cat(out_log_posteriors)
    out_posteriors = torch.cat(out_posteriors)
    in_log_prior = m(self.in_logit_prior)
    out_log_prior = m(self.out_logit_prior)

    I_ZX = (in_posteriors * (in_log_posteriors - in_log_prior)).sum(dim=-1).sum()
    I_WX = (out_posteriors * (out_log_posteriors - out_log_prior)).sum(dim=-1).sum() 
    I_WY = torch.sum(m(S).diag()) + torch.sum(m(S.transpose(0, 1)).diag())
    I_ZX = I_ZX / n
    I_WX = I_WX / n
    I_WY = I_WY / n
    return I_ZX + I_WX - beta * I_WY, I_ZX, I_WX, I_WY

  def retrieve(self, audio_scores, image_scores):
    n = audio_scores.size(0)
    device = audio_scores.device
    S = torch.zeros((n, n), dtype=torch.float, device=device)
    for i in range(n):
      for j in range(n):
        # Attention scores shape: [B, W, L]
        att_scores = torch.matmul(audio_scores[i].unsqueeze(0), image_scores[j].t().unsqueeze(0))
        att_first = F.softmax(att_scores, dim=-1) # TODO Apply mask  
        att_second = F.softmax(att_scores, dim=-2)

        # Outputs shape: [B,]
        outputs = ((att_first + att_second) * att_scores).sum(dim=-1).sum(dim=-1)
        S[i][j] = outputs.squeeze(0) 
    
    _, A2I_idxs = S.topk(10, 1)
    _, I2A_idxs = S.topk(10, 0)
    return A2I_idxs, I2A_idxs.t() 

if __name__ == '__main__':
  config_path = 'configs/flickr8k/bottleneck.json' 
  config = json.load(open(config_path, 'r'))
  model = AudioVisualInformationBottleneck(config['data']['num_features'], 
                                   config['data']['num_visual_features'],
                                   **config['model'])
  
  audio_inputs = torch.randn(4, 80, 100)
  image_inputs = torch.randn(4, 8, 4096)
  # in_scores, in_probs, out_scores, out_probs, outputs = model(audio_inputs, image_inputs)
  # print('in_scores.size(), out_scores.size(), outputs.size(): ', in_scores.size(), out_scores.size(), outputs.size())
  print(model.calculate_loss(audio_inputs, image_inputs))
