import gtn
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from models import TDS
from transducer import ConvTransduce1D

class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(80,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x, l=5):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        if l == 3:
            return x
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        if l == 4:
            return x
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        return x

class PositionDependentUnigramBottleneck(torch.nn.Module):
  '''
  Information bottleneck model with position-dependent unigram assumption
  for multimoda phone discovery.

  Args:
      input_size: int, dimension of the input features
      output_size: int, number of word types 
      hidden_size: int, number of phone types
  '''
  def __init__(self, input_size, output_size,
               bottleneck_size,
               tds_groups, kernel_size, 
               stride, dropout, 
               beta=10., **kwargs): 
    super(PositionDependentUnigramBottleneck, self).__init__()
    self.bottleneck = TDS(input_size, bottleneck_size, tds_groups, kernel_size, dropout)
    
    self.predictor = BiLSTM(output_size, bottleneck_size) # TODO
    
    # Prior
    self.logit_prior = nn.Parameter(torch.empty(bottleneck_size))
    nn.init.ones_(self.logit_prior)
    self.beta = beta

  def forward(self, inputs, outputs, input_masks):
    device = audio_inputs.device
    # Inputs shape: [B, F, T]
    in_scores = self.bottleneck(inputs)
    pooling_ratio = inputs.size(-1) // in_scores.size(1)
    score_masks = input_masks[:, ::pooling_ratio].unsqueeze(-1)
    
    # Input token scores shape: [B, S, bottleneck_size]
    in_scores = in_scores * score_masks
    
    # Output token scores shape: [B, S, bottleneck_size]
    out_scores = self.predictor(outputs, in_scores.size(-1))
    return in_scores,
           out_scores
  
  def calculate_loss(self,
                     in_scores,
                     out_scores):
    m = nn.LogSoftmax(dim=2)
    beta = torch.tensor(self.beta, device=audio_inputs.device)
    
    log_prior = m(self.logit_prior)
    in_log_posteriors = m(in_scores)
    out_log_posteriors = m(out_scores)

    in_scores_ = torch.where(in_scores != 0,
                in_scores,
                torch.tensor(-9e9, device=device))
    out_scores_ = torch.where(out_scores != 0,
                             out_scores,
                             torch.tensor(-9e9, device=device))
    in_posteriors = F.softmax(in_scores_)
    out_posteriors = F.softmax(out_scores_)

    
    I_ZX = (in_posteriors * (in_log_posteriors - in_log_prior)).sum(dim=-1).mean()
    I_ZY = (out_posteriors * out_log_posteriors).sum(dim=-1).mean()
    
    return I_ZX - beta * I_ZY, I_ZX, I_ZY
