import gtn
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from models import TDS
import math

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

class BLSTM(nn.Module):
  def __init__(self, input_size, n_class, embedding_dim=100, n_layers=1):
    super(BLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    # self.i2h = nn.Linear(40 + embedding_dim, embedding_dim)
    # self.i2o = nn.Linear(40 + embedding_dim, n_class) 
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2 * embedding_dim, n_class)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x.unsqueeze(0)

    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    embed, _ = self.rnn(x, (h0, c0))
    outputs = []
    for b in range(B):
      # out = self.softmax(self.fc(embed[b]))
      out = self.fc(embed[b])
      outputs.append(out)

    if save_features:
      return embed, torch.stack(outputs, dim=1)
    else:
      return torch.stack(outputs, dim=1)
    
class PositionalEncoding(nn.Module):
    '''
    Borrowed from this tutorial:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, input_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-math.log(10000.0) / input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
        
class PDMLP(torch.nn.Module):
  '''
  Position-dependent multi-layer peceptron

  Args:
      input_size: int, dimension of the input features
      output_size: int, number of output classes
      hidden_size: int, dimension of the hidden embeddings 
  '''
  def __init__(self, input_size, output_size, hidden_size=512):
      super(PDMLP, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, output_size) 
      self.pe = PositionalEncoding(hidden_size) 
      # nn.init.orthogonal_(self.fc1.weight)
      # nn.init.zeros_(self.fc1.bias)

  def forward(self, x):
      outputs = F.relu(self.fc1(x))
      outputs = self.pe(outputs)
      outputs = self.fc2(outputs)
      return outputs.permute(1, 0, 2)
      
class PositionDependentUnigramBottleneck(torch.nn.Module):
  '''
  Information bottleneck model with position-dependent unigram assumption
  for multimoda phone discovery.

  Args:
      input_size: int, dimension of the input features
      output_size: int, dimension of the output features
      bottleneck_size: int, number of phoneme types
  '''
  def __init__(self, input_size, output_size,
               model_type,
               bottleneck_size, hidden_size,
               tds_groups, kernel_size, 
               dropout, 
               beta=100., **kwargs): 
    super(PositionDependentUnigramBottleneck, self).__init__()
    self.bottleneck_size = bottleneck_size
    self.output_size = output_size
    if model_type == 'blstm':
        self.tds = BLSTM(input_size, output_size)
    elif model_type == 'tds':
        self.tds = TDS(input_size, output_size, tds_groups, kernel_size, dropout)
    elif model_type == 'pdmlp':
        self.tds = PDMLP(input_size, output_size)
        
    self.beta = beta

  def forward(self, inputs, input_masks):
    m = nn.LogSoftmax(dim=-1)
    B = inputs.size(0)
    
    device = inputs.device
    # Inputs shape: [B, F, T]
    in_scores = self.tds(inputs.permute(0, 2, 1)).permute(1, 0, 2) * input_masks.unsqueeze(-1)
    out_scores = in_scores.sum(dim=1)
    
    return in_scores, out_scores

  def calculate_loss(self,
                     in_scores,
                     prediction_loss):
    m = nn.LogSoftmax(dim=-1)

    in_scores = in_scores.sum(dim=1)
    in_log_posteriors = m(in_scores)
    in_posteriors = F.softmax(in_scores, dim=-1)

    I_ZX = (in_posteriors * in_log_posteriors).sum(dim=-1).mean()
    
    return prediction_loss, I_ZX, prediction_loss
