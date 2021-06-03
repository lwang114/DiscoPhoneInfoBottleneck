import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

class VQCPCEncoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim):
        super(VQCPCEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
        self.encoder = nn.Sequential(
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def encode(self, mel):
        z = self.conv(mel)
        z = self.encoder(z.transpose(1, 2))
        z, indices = self.codebook.encode(z)
        c, _ = self.rnn(z)
        return z, c, indices

    def forward(self, mels):
        z = self.conv(mels)
        z = self.encoder(z.transpose(1, 2))
        z, loss = self.codebook(z)
        c, _ = self.rnn(z)
        return z, c, loss

class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
            
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss
    
class GumbelEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, decay=0.999, epsilon=1e-5):
        super(GumbelEmbeddingEMA, self).__init__()
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 10
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_embeddings))
        self.register_buffer('ema_weight', self.embedding.clone())

    def forward(self, logit, x):
        M, D = self.embedding.size()
        prob = F.softmax(logit, dim=-1)
        quantized = torch.mm(prob, self.embedding)

        if self.training:
            # indices = torch.argmin(prob, dim=-1)
            # encodings = F.one_hot(indices, M).float()
            encodings = prob.detach()
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            dw = torch.matmul(encodings.t(), x.detach())
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        return quantized
            
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
  def __init__(self, embedding_dim=100, n_layers=1, input_size=80):
    super(BLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)

  def forward(self, x, masks=None):
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
    if not masks is None:
        x = x * masks.unsqueeze(2)
    return x
 
class GaussianBLSTM(nn.Module):
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80):
    super(GaussianBLSTM, self).__init__()
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
    elif num_sample > 1 : logit = torch.log(F.softmax(logit, dim=2).mean(0))

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

class GumbelMLP(nn.Module):
  def __init__(self,
               embedding_dim,
               n_layers=1,
               n_class=65,
               input_size=80):
    super(GumbelMLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.mlp = nn.Sequential(
                 nn.Linear(input_size, embedding_dim),
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.ReLU(),
                 nn.Dropout(0.3)
               )
    self.bottleneck = nn.Linear(embedding_dim, 49)
    self.decode = nn.Linear(49, self.n_class)

  def forward(self, x, 
              num_sample=1,
              masks=None,
              temp=1.,
              return_feat=False):
    x = x.permute(0, 2, 1)
    B = x.size(0)
    D = x.size(2)
    embed = self.mlp(x)
    logits = self.bottleneck(embed) 

    if masks is not None:
      logits = logits * masks.unsqueeze(2)
    logit = logits.sum(1)
    encoding = self.reparametrize_n(logits, 
                                     n=num_sample, 
                                     temp=temp)
    out = self.decode(encoding)
    if num_sample > 1:
      out = out.mean(0)
        
    if return_feat:
      return logit, out, encoding, embed
    else:
      return logit, out

  def reparametrize_n(self, x, n=1, temp=1.):
    # reference :
    # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
    # param x: FloatTensor of size (batch size, num. frames, num. classes) 
    # param n: number of samples
    # return encoding: FloatTensor of size (n, batch size, num. frames, num. classes)
    def expand(v):
        if v.ndim < 1:
            return torch.Tensor([v]).expand(n, 1)
        else:
            return v.expand(n, *v.size())

    if n != 1 :
        x = expand(x)
    encoding = F.gumbel_softmax(x, tau=temp)
    return encoding         


class GumbelBLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65, 
               input_size=80, 
               ds_ratio=1,
               bidirectional=True):
    super(GumbelBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = ds_ratio
    self.bidirectional = bidirectional
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
    self.bottleneck = nn.Linear(2*embedding_dim if bidirectional else embedding_dim, 49)
    self.decode = nn.Linear(49, self.n_class)

  def forward(self, x, 
              num_sample=1, 
              masks=None, 
              temp=1., 
              return_feat=False):
    ds_ratio = self.ds_ratio
    device = x.device
    if x.dim() < 3:
        x = x.unsqueeze(0)
    elif x.dim() > 3:
        x = x.squeeze(1)
    x = x.permute(0, 2, 1)
        
    B = x.size(0)
    T = x.size(1)
    if self.bidirectional:
      h0 = torch.zeros((2 * self.n_layers, B, self.K))
      c0 = torch.zeros((2 * self.n_layers, B, self.K))
    else:
      h0 = torch.zeros((self.n_layers, B, self.K))
      c0 = torch.zeros((self.n_layers, B, self.K))

    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    embed, _ = self.rnn(x, (h0, c0))
    x = self.bottleneck(embed)
    
    if masks is not None:
      x = x * masks.unsqueeze(2)

    in_logit = x.sum(dim=1)
    encoding = self.reparametrize_n(x, num_sample, temp)
    L = ds_ratio * (T // ds_ratio)
    if encoding.dim() > 3:
        encoding = encoding[:, :, :L].view(num_sample, B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
    else:
        encoding = encoding[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
    logit = self.decode(encoding)

    if num_sample > 1:
      logit = torch.log(F.softmax(logit, dim=2).mean(0))

    if return_feat:
        embedding = embed[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
        return in_logit, logit, encoding, embedding
    return in_logit, logit
    
  def reparametrize_n(self, x, n=1, temp=1.):
      # reference :
      # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
      # param x: FloatTensor of size (batch size, num. frames, num. classes) 
      # param n: number of samples
      # return encoding: FloatTensor of size (n, batch size, num. frames, num. classes)
      def expand(v):
          if v.ndim < 1:
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
        return in_logit, logit
    
    def weight_init(self):
        pass

class GumbelPyramidalBLSTM(nn.Module):
  def __init__(self, 
               embedding_dim=100, 
               n_layers=1, 
               n_class=65, 
               input_size=80, 
               ds_ratio=1.,
               bidirectional=True):
    super(GumbelPyramidalBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.bidirectional = bidirectional
    self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
    self.rnn2 = nn.LSTM(input_size=embedding_dim*4, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
    self.rnn3 = nn.LSTM(input_size=embedding_dim*4, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
    self.bottleneck = nn.Linear(2*embedding_dim if bidirectional else embedding_dim, 
                                49)
    self.decode = nn.Linear(49, self.n_class)
    
  def forward(self, x, num_sample=1, masks=None, temp=1., return_feat=None):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    elif x.dim() > 3:
      x = x.squeeze(1)
    x = x.permute(0, 2, 1)

    B = x.size(0)
    T = x.size(1)
    if self.bidirectional:
      h0 = torch.zeros((2 * self.n_layers, B, self.K))
      c0 = torch.zeros((2 * self.n_layers, B, self.K))
    else:
      h0 = torch.zeros((self.n_layers, B, self.K))
      c0 = torch.zeros((self.n_layers, B, self.K))


    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()

    x, _ = self.rnn1(x, (h0, c0))
    L = T // 2
    x = x[:, :2*L].contiguous().view(B, L, -1)
    x, _ = self.rnn2(x, (h0, c0))
    L = L // 2
    x = x[:, :2*L].contiguous().view(B, L, -1)
    embed, _ = self.rnn3(x, (h0, c0))
    x = self.bottleneck(embed)
    # if not masks is None:
    #   x = x * masks[:, ::4].unsqueeze(2)

    in_logit = x.sum(dim=1)
    encoding = self.reparametrize_n(x,num_sample,temp)
    logit = self.decode(encoding)

    if num_sample > 1: 
      logit = F.softmax(logit, dim=2).mean(0)

    if return_feat:
      if return_feat == 'bottleneck':
          return in_logit, logit, encoding
      elif return_feat == 'rnn':
          return in_logit, logit, embed
    else:
      return in_logit, logit

  def weight_init(self):
      pass

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
  
class GumbelMarkovModelCell(nn.Module):
  def __init__(self, input_size, num_states):
    """
    Discrete deep Markov model with Gumbel softmax samples. 
    """
    super(GumbelMarkovModelCell, self).__init__()
    self.num_states = num_states
    self.weight_ih = nn.Parameter(
        torch.FloatTensor(input_size, num_states))
    self.weight_hh = nn.Parameter(
        torch.FloatTensor(num_states, num_states))
    self.bias = nn.Parameter(torch.FloatTensor(num_states))

    self.fc = nn.Linear(input_size, num_states) 
    self.trans = nn.Parameter(torch.FloatTensor(num_states, num_states))
    
    self.reset_parameters()

  def reset_parameters(self):
    init.orthogonal_(self.weight_ih.data)
    init.eye_(self.weight_hh)
    init.constant_(self.bias.data, val=0)
    init.eye_(self.trans)
    
  def forward(self, input_, z_0, temp=1.): # TODO Generalize to k-steps
    """
    :param input_: FloatTensor of size (batch, input size), input features
    :param z_0: FloatTensor of size (batch, num. states), sample at the current time step
    :return z_1: FloatTensor of size (batch, num. states), sample for the next time step 
    :return logit_z1_given_z0: FloatTensor of size (batch, num. states)
    """
    batch_size = input_.size(0)
    bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
    logit_z_1 = self.fc(input_)
    wh_b = torch.addmm(bias_batch, z_0, self.weight_hh)
    wi = torch.mm(input_, self.weight_ih)
    g = wh_b + wi

    logit_prior_z1_given_z0 = torch.mm(z_0, self.trans)
    logit_z1_given_z0 = torch.sigmoid(g)*logit_prior_z1_given_z0 +\
                        (1 - torch.sigmoid(g))*logit_z_1
    z_1 = F.gumbel_softmax(logit_z1_given_z0, tau=temp)
    
    return z_1, logit_z1_given_z0


class GumbelMarkovBLSTM(nn.Module):
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80):
    super(GumbelMarkovBLSTM, self).__init__()
    self.K = embedding_dim
    self.bottleneck_dim = 49
    self.n_layers = n_layers
    self.n_class = n_class
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.bottleneck = GumbelMarkovModelCell(embedding_dim*2, self.bottleneck_dim)
    self.decode = nn.Linear(self.bottleneck_dim, self.n_class) 

  @staticmethod
  def _forward_bottleneck(cell, input_, length, z_0, n=1, temp=1.):
    device = input_.device
    def expand(v):
      if isinstance(v, Number):
        return torch.Tensor([v]).expand(n, 1)
      else:
        return v.expand(n, *v.size())

    B = z_0.size(0)
    num_states = z_0.size(-1)
    in_size = input_.size()[1:]
    if n != 1:
        z_0 = expand(z_0).contiguous().view(B*n, num_states)
        input_ = expand(input_).contiguous().view(B*n, *in_size) 
        length = expand(length).flatten()
    
    input_ = input_.permute(1, 0, 2)
    max_time = input_.size(0)
    output = []
    logits = []
    for time in range(max_time):
      z_1, logit = cell(input_=input_[time], z_0=z_0, temp=temp)
      mask = (time < length).float().unsqueeze(1).expand_as(z_1).to(device)
      z_1 = z_1*mask + z_0*(1 - mask)
      output.append(z_1)
      logits.append(logit)
      z_0 = z_1
    output = torch.stack(output, 1)
    logits = torch.stack(logits, 1)
    
    if n != 1:
      output = output.view(n, B, max_time, num_states)
      logits = logits.view(n, B, max_time, num_states)
      
    return output, logits

  def forward(self, x, num_sample=1, masks=None, temp=1., return_feat=None):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    elif x.dim() > 3:
      x = x.squeeze(1)
    x = x.permute(0, 2, 1)

    B = x.size(0)   
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.K)).to(x.device)
    c0 = torch.zeros((2 * self.n_layers, B, self.K)).to(x.device)
    z0 = torch.zeros((B, 49)).to(x.device)
    embed, _ = self.rnn(x, (h0, c0))
    
    if not masks is None:
      length = masks.sum(-1)
    else:
      length = T * torch.ones(B, dtype=torch.int)
    encoding, in_logit = GumbelMarkovBLSTM._forward_bottleneck(
        cell=self.bottleneck, input_=x, length=length, z_0=z0, n=num_sample, temp=temp)
    logit = self.decode(encoding)

    if num_sample != 1:
        logit = torch.log(F.softmax(logit, dim=2).mean(0))
    
    if return_feat:
        if return_feat == 'bottleneck':
            return in_logit, logit, encoding
        elif return_feat == 'rnn':
            return in_logit, logit, embed
    else:
        return in_logit, logit
    
  def weight_init(self):
      pass
   
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
