import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

import time
from numbers import Number

class Davenet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Davenet, self).__init__()
        self.K = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(80,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim*2, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))
        self.decode = nn.Sequential(nn.Linear(self.K, 65))

        
    def forward(self, x, num_sample=1, masks=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        if not masks is None:
            x = x * masks[:, ::16].unsqueeze(2)
        statistics = x.sum(dim=1)
        
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1)
        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        # for m in self._modules:
        #     xavier_init(self._modules[m])
        pass


class BLSTM(nn.Module):
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80):
    super(BLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.decode = nn.Linear(2*embedding_dim, self.n_class)

  def forward(self, x, num_sample=1, masks=None):
    device = x.device
    if x.dim() < 3:
        x = x.unsqueeze(0)
    elif x.dim() > 3:
        x = x.squeeze(1)
    x = x.permute(0, 2, 1)
        
    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.K))
    c0 = torch.zeros((2 * self.n_layers, B, self.K))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    x, _ = self.rnn(x, (h0, c0))
    outputs = []
    
    if not masks is None:
      x = x * masks.unsqueeze(2)
    statistics = x.sum(dim=1)
    mu = statistics
    std = F.softplus(0 * torch.ones((mu.size(0), 2*self.K), device=device)) # statistics[:, self.K:]-5,beta=1)
    encoding = self.reparametrize_n(mu,std,num_sample)
    logit = self.decode(encoding)

    if num_sample == 1 : pass
    elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

    return (mu, std), logit
    
  def reparametrize_n(self, mu, std, n=1):
      # reference :
      # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
      def expand(v):
          if isinstance(v, Number):
              return torch.Tensor([v]).expand(n, 1)
          else:
              return v.expand(n, *v.size())

      if n != 1 :
          mu = expand(mu)
          std = expand(std)

      eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

      return mu + eps * std

  def weight_init(self):
      pass

class GumbelBLSTM(nn.Module):
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80, ds_ratio=1):
    super(GumbelBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = ds_ratio
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.bottleneck = nn.Linear(2*embedding_dim, 49)
    self.decode = nn.Linear(49, self.n_class)

  def forward(self, x, num_sample=1, masks=None, temp=1., return_encoding=False):
    ds_ratio = self.ds_ratio
    device = x.device
    if x.dim() < 3:
        x = x.unsqueeze(0)
    elif x.dim() > 3:
        x = x.squeeze(1)
    x = x.permute(0, 2, 1)
        
    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.K))
    c0 = torch.zeros((2 * self.n_layers, B, self.K))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    x, _ = self.rnn(x, (h0, c0))
    x = self.bottleneck(x)
    
    if not masks is None:
      x = x * masks.unsqueeze(2)

    in_logit = x.sum(dim=1)
    encoding = self.reparametrize_n(x,num_sample,temp)
    L = ds_ratio * (T // ds_ratio)
    if encoding.dim() > 3:
        encoding = encoding[:, :, :L].view(num_sample, B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
    else:
        encoding = encoding[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
    logit = self.decode(encoding)
    logit = logit.sum(dim=-2)

    if num_sample == 1 : pass
    elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

    if return_encoding:
        return in_logit, logit, encoding
    else:
        return in_logit, logit
    
  def reparametrize_n(self, x, n=1, temp=1.):
      # reference :
      # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
      # param x: FloatTensor of size (batch size, num. frames, num. classes) 
      # param n: number of samples
      # return encoding: FloatTensor of size (n, batch size, num. frames, num. classes)
      def expand(v):
          if isinstance(v, Number):
              return torch.Tensor([v]).expand(n, 1)
          else:
              return v.expand(n, *v.size())

      if n != 1 :
          x = expand(x)
      encoding = F.gumbel_softmax(x, tau=temp)

      return encoding

  def weight_init(self):
      pass

class ExactDiscreteBLSTM(nn.Module):
    def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80):
        super(ExactDiscreteBLSTM, self).__init__()
        self.K = embedding_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.bottleneck = nn.Linear(2*embedding_dim, 49)
        self.decode = nn.Linear(49, self.n_class)

    def forward(self, x, num_sample=1, masks=None, temp=1.):
        device = x.device
        if x.dim() < 3:
            x = x.unsqueeze(0)
        elif x.dim() > 3:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        B = x.size(0)
        T = x.size(1)

        h0 = torch.zeros((2 * self.n_layers, B, self.K))
        c0 = torch.zeros((2 * self.n_layers, B, self.K))
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        x, _ = self.rnn(x, (h0, c0))
        x = self.bottleneck(x)

        if not masks is None:
            x = x * masks.unsqueeze(2)

        in_logit = x.sum(dim=1)
        logit = torch.matmul(F.softmax(x, dim=-1), F.log_softmax(self.decode.weight.t(), dim=-1))
        logit = logit.sum(dim=-2)
        return in_logit, logit

    def weight_init(self):
        pass

class GumbelPyramidalBLSTM(nn.Module):
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80, ds_ratio=1.)
    super(GumbelPyramidalBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first)
    self.rnn2 = nn.LSTM(input_size=embedding_dim*4, hidden_size=embedding_dim, num_layers=n_layers, batch_first)
    self.rnn3 = nn.LSTM(input_size=embedding_dim*4, hidden_size=embedding_dim, num_layers=n_layers, batch_first)

  def forward(self, x, num_sample=1, masks=None, temp=1., return_encoding=False):
    device = x.device
    if x.dim() < 3:
      x = x.unsqueeze(0)
    elif x.dim() > 3:
      x = x.squeeze(1)
    x = x.permute(0, 2, 1)

    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.K))
    c0 = torch.zeros((2 * self.n_layers, B, self.K))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()

    x, _ = self.rnn1(x, (h0, c0))
    print('rnn1: ', x.size())
    L = 2 * (T // 2)
    x, _ = self.rnn2(x[:, :L].view(B, L // 2, -1), (h0, c0))
    print('rnn2: ', x.size())
    L = 2 * (L // 2)
    x, _ = self.rnn3(x[:, :L].view(B, L // 2, -1), (h0, c0))
    print('rnn3: ', x.size())

    if not masks is None:
      x = x * masks[:, ::4].unsqueeze(2)
    
    encoding = self.reparametrize_n(x,num_sample,temp)
    logit = self.decode(encoding)
    logit = logit.sum(dim=-2)

    if num_sample == 1 : pass
    elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

    if return_encoding:
      return in_logit, logit, encoding
    else:
      return in_logit, logit

'''
class GumbelMarkovBLSTM(nn.Module): # TODO
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80)
    super(GumbelMarkovBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first)
'''
    
class BigToyNet(nn.Module):
    def __init__(self, K=256):
        super(BigToyNet, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Linear(10240, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.K))

        self.decode = nn.Sequential(
                nn.Linear(self.K, 65))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        std = F.softplus(-1*torch.ones((x.size(0), self.K), device=x.device),beta=1) # XXX F.softplus(statistics[:,self.K:]-5,beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

  
    
class ToyNet(nn.Module):
    def __init__(self, K=256):
        super(ToyNet, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.K))

        self.decode = nn.Sequential(
                nn.Linear(self.K, 10))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
