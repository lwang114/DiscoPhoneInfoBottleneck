import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from utils.utils import cuda

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

class VQMLP(torch.nn.Module):
  def __init__(self, 
               embedding_dim,
               n_layers=1,
               n_class=65,
               n_embeddings=40,
               input_size=80):
    super(VQMLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.conv = nn.Conv2d(1, embedding_dim,
                          kernel_size=(input_size, 5),
                          stride=(1, 1),
                          padding=(0, 2))
    self.mlp = nn.Sequential(
                 nn.LayerNorm(embedding_dim),
                 nn.Dropout(0.2),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.LayerNorm(embedding_dim),

                 nn.Dropout(0.2),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.LayerNorm(embedding_dim),
                 nn.Dropout(0.2),
                 nn.ReLU()
               )
    self.bottleneck = VQEmbeddingEMA(n_embeddings, embedding_dim)
    self.decode = nn.Linear(n_embeddings, self.n_class)
    self.ds_ratio = 1

  def forward(self, x,
              num_sample=1,
              masks=None,
              temp=1.,
              return_feat=False):
    B = x.size(0)
    D = x.size(1)
    T = x.size(2)

    x = self.conv(x.unsqueeze(1)).squeeze(2)  
    x = x.permute(0, 2, 1)
    embed = self.mlp(x)
    x_flat = embed.view(-1, D)
    quantized, loss = self.bottleneck(embed)
    logits = torch.addmm(torch.sum(self.bottleneck.embedding ** 2, dim=1) +
                         torch.sum(x_flat ** 2, dim=1, keepdim=True),
                         x_flat, self.bottleneck.embedding.t(),
                         alpha=2.0, beta=-1.0).view(B, T, -1)

    logits = logits / ((D ** 0.5) * temp)
    encoding = F.softmax(logits, dim=-1)
    if masks is not None:
      quantized = quantized * masks.unsqueeze(2)
      logits = logits * masks.unsqueeze(2)
    out = self.decode(encoding)
    out = torch.cat((out, quantized), dim=2)
    if return_feat:
      return logits, out, encoding, embed 
    else:
      return logits, out
    
  def quantize_loss(self, embed, quantized, masks=None):
    if masks is not None:
      embed = masks.unsqueeze(2) * embed
      masks = masks.unsqueeze(2) * quantized.detach() 
    return self.bottleneck.commitment_cost * F.mse_loss(embed, masks)

class TDSBlock(torch.nn.Module):
    def __init__(self, in_channels, num_features, kernel_size, dropout):
        super(TDSBlock, self).__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        fc_size = in_channels * num_features
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.Dropout(dropout),
        )
        self.instance_norms = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm1d(fc_size, affine=True),
                torch.nn.InstanceNorm1d(fc_size, affine=True),
            ]
        )

    def forward(self, inputs, return_feat=False):
        # inputs shape: [B, C * H, W]
        B, CH, W = inputs.shape
        C, H = self.in_channels, self.num_features
        outputs = self.conv(inputs.view(B, C, H, W)).view(B, CH, W) + inputs
        outputs = self.instance_norms[0](outputs)

        outputs = self.fc(outputs.transpose(1, 2)).transpose(1, 2) + outputs
        outputs = self.instance_norms[1](outputs)

        # outputs shape: [B, C * H, W]
        return outputs


class GumbelTDS(torch.nn.Module):
    def __init__(self, 
                 tds_groups=[
                   { "channels" : 4, "num_blocks" : 5 },
                   { "channels" : 8, "num_blocks" : 5 },
                   { "channels" : 16, "num_blocks" : 5 }],
                 kernel_size=5, 
                 dropout=0.2,
                 n_class=258,
                 n_gumbel_units=49,
                 input_size=80):
        super(GumbelTDS, self).__init__()
        modules = []
        in_channels = input_size
        for tds_group in tds_groups:
            # add downsample layer:
            out_channels = input_size * tds_group["channels"]
            modules.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=tds_group.get("stride", 2),
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.InstanceNorm1d(out_channels, affine=True),
                ]
            )
            for _ in range(tds_group["num_blocks"]):
                modules.append(
                    TDSBlock(tds_group["channels"], input_size, kernel_size, dropout)
                )
            in_channels = out_channels
        self.tds = torch.nn.Sequential(*modules)
        self.bottleneck = torch.nn.Linear(in_channels, n_gumbel_units)
        self.linear = torch.nn.Linear(n_gumbel_units, n_class)
        self.ds_ratio = 2 ** len(tds_groups)

    def forward(self, inputs,
                masks=None,
                num_sample=1,
                temp=1.,
                return_feat=False):
        # inputs shape: [B, H, W]
        embeddings = self.tds(inputs)
        embeddings = embeddings.permute(0, 2, 1)
        in_logits = self.bottleneck(embeddings)

        if masks is not None:
          in_logits = in_logits * masks.unsqueeze(2)
        encodings = self.reparametrize_n(in_logits,
                                         n=num_sample,
                                         temp=temp)

        # outputs shape: [B, W, output_size]
        out_logits = self.linear(encodings)
        if num_sample > 1:
          out_logits = out_logits.mean(0) 

        if return_feat:
          if masks is not None:
            embeddings = (embeddings * masks.unsqueeze(2)).sum(1)
            embeddings = embeddings / masks.sum(-1, keepdim=True)
          else:
            embeddings = embeddings.sum(-2)
          return in_logits, out_logits, encodings, embeddings 
        return in_logits, out_logits 
    
    def reparametrize_n(self, x, n=1, temp=1.):
      def expand(v):
        if v.ndim < 1:
          return torch.Tensor([v]).expand(n, 1)
        else:
          return v.expand(n, *v.size())

      if n != 1:
          x = expand(x)
      encoding = F.gumbel_softmax(x, tau=temp)

      return encoding

class MLP(nn.Module):
  def __init__(self,
               embedding_dim,
               n_layers=1,
               n_class=65,
               input_size=80,
               max_seq_len=100,
               context_width=5):
    super(MLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = 1
    self.mlp = nn.Sequential(
                 # nn.Linear(embedding_dim, embedding_dim),
                 nn.Linear(input_size, embedding_dim),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.ReLU(),
               )
    self.decode = nn.Linear(embedding_dim * round(max_seq_len // self.ds_ratio),
                            self.n_class,
                            bias=False)
    
  def forward(self, x,
              masks=None,
              return_feat=False):
    B = x.size(0)
    x = x.permute(0, 2, 1)
    embed = self.mlp(x)
    out = self.decode(embed.view(B, -1))
    if return_feat:
      return out, embed
    else:
      return out
 
class GumbelMLP(nn.Module):
  def __init__(self,
               embedding_dim,
               n_layers=1,
               n_class=65,
               n_gumbel_units=40,
               input_size=80,
               max_seq_len=100,
               context_width=5):
    super(GumbelMLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = 1
    self.batchnorm1 = nn.BatchNorm2d(1)
    self.conv1 = nn.Conv2d(1, embedding_dim,
                          kernel_size=(input_size, context_width),
                          stride=(1, 1),
                          padding=(0, int(context_width // 2)), bias=False) # nn.Linear(input_size, embedding_dim),
    # self.conv2 = nn.Conv2d(128, embedding_dim,
    #                        kernel_size=(1, 5), 
    #                        stride=(1, 1), 
    #                        padding=(0, 2))
    # self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    self.mlp = nn.Sequential(
                 # nn.Linear(embedding_dim, embedding_dim),
                 nn.Linear(input_size, embedding_dim),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.ReLU(),
               )
    self.bottleneck = nn.Linear(embedding_dim, n_gumbel_units)
    # self.decode = nn.Linear(n_gumbel_units, self.n_class) # XXX 
    self.decode = nn.Linear(n_gumbel_units * round(max_seq_len // self.ds_ratio), 
                            self.n_class,
                            bias=False)

  def forward(self, x, 
              num_sample=1,
              masks=None,
              temp=1.,
              return_feat=False):
    B = x.size(0)
    # x = x.unsqueeze(1)
    # x = self.batchnorm1(x)
    # x = F.relu(self.conv1(x))
    # x = self.pool(x)
    # x = F.relu(self.conv2(x))
    # x = self.pool(x).squeeze(2)
    # x = x.squeeze(2).permute(0, 2, 1)
    x = x.permute(0, 2, 1)
    embed = self.mlp(x)
    logits = self.bottleneck(embed) 
    encoding = self.reparametrize_n(logits, 
                                    n=num_sample, 
                                    temp=temp)
    if masks is not None:
      encoding = encoding * masks.unsqueeze(-1)
   
    if num_sample > 1:
      out = self.decode(encoding.view(num_sample, B, -1))
      out = out.mean(0)
    else:
      # out = self.decode(encoding).sum(-2) # XXX
      out = self.decode(encoding.view(B, -1))

    if return_feat:
      return logits, out, encoding, embed
    else:
      return logits, out

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

class BLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65,
               input_size=80, 
               ds_ratio=1,
               bidirectional=True,
               decoder=None):
    super(BLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = ds_ratio
    self.bidirectional = bidirectional
    self.rnn = nn.LSTM(input_size=input_size,
                       hidden_size=embedding_dim,
                       num_layers=n_layers,
                       batch_first=True,
                       bidirectional=bidirectional)
    if decoder is None:
      self.decode = nn.Linear(2 * embedding_dim if bidirectional
                              else embedding_dim, self.n_class)
    else:
      self.decode = decoder

  def forward(self, x, 
              return_feat=False):
    device = x.device
    ds_ratio = self.ds_ratio
    if x.dim() < 3:
        x = x.unsqueeze(0)
    elif x.dim() > 3:
        x = x.squeeze(1)
    x = x.permute(0, 2, 1)
    
    B = x.size(0)
    T = x.size(1)
    if self.bidirectional:
      h0 = torch.zeros((2 * self.n_layers, B, self.K), device=device)
      c0 = torch.zeros((2 * self.n_layers, B, self.K), device=device)
    else:
      h0 = torch.zeros((self.n_layers, B, self.K), device=device)
      c0 = torch.zeros((self.n_layers, B, self.K), device=device)
       
    embed, _ = self.rnn(x, (h0, c0))
    logit = self.decode(embed)

    if return_feat:
        L = ds_ratio * (T // ds_ratio)
        embedding = embed[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1)
        embedding = embedding.sum(-2)
        return logit, embedding
    return logit


class GaussianBLSTM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 n_layers=1,
                 n_class=65,
                 input_size=80,
                 ds_ratio=1,
                 bidirectional=True):
        super(GaussianBLSTM, self).__init__()
        self.K = 2 * embedding_dim if bidirectional\
                 else embedding_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.ds_ratio = ds_ratio
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=embedding_dim,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=bidirectional)
        self.encode = nn.Linear(2 * embedding_dim if bidirectional
                                else embedding_dim,
                                4 * embedding_dim if bidirectional
                                else 2 * embedding_dim)
        self.decode = nn.Linear(2 * embedding_dim if bidirectional
                                else embedding_dim, self.n_class)
        
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
          h0 = torch.zeros((2 * self.n_layers, B, int(self.K // 2)), device=x.device)
          c0 = torch.zeros((2 * self.n_layers, B, int(self.K // 2)), device=x.device)
        else:
          h0 = torch.zeros((self.n_layers, B, self.K), device=x.device)
          c0 = torch.zeros((self.n_layers, B, self.K), device=x.device)          
        embedding, _ = self.rnn(x, (h0, c0))
        statistics = self.encode(embedding) 
        mu = statistics[:, :, :self.K]
        std = F.softplus(statistics[:, :, self.K:]-5, beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)

        if num_sample == 1: pass
        elif num_sample > 1: logit = torch.log(F.softmax(logit, dim=2).mean(0))

        if return_feat:
          return (mu, std), logit, embedding
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
      

class GumbelBLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65, 
               n_gumbel_units=49,
               input_size=80, 
               ds_ratio=1,
               bidirectional=True,
               decoder=None):
    super(GumbelBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = ds_ratio
    self.bidirectional = bidirectional
    self.rnn = nn.LSTM(input_size=input_size,
                       hidden_size=embedding_dim,
                       num_layers=n_layers,
                       batch_first=True,
                       bidirectional=bidirectional)
    self.bottleneck = nn.Linear(2*embedding_dim if bidirectional 
                                                else embedding_dim, n_gumbel_units)
    if decoder is None:
      self.decode = nn.Linear(n_gumbel_units, self.n_class)
    else:
      self.decode = decoder

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
    in_logit = self.bottleneck(embed)
    
    if masks is not None:
      in_logit = in_logit * masks.unsqueeze(2)

    encoding = self.reparametrize_n(in_logit, num_sample, temp)
    L = ds_ratio * (T // ds_ratio)
    if encoding.dim() > 3:
        encoding = encoding[:, :, :L].view(num_sample, B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
    else:
        encoding = encoding[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1).mean(dim=-2)
    logit = self.decode(encoding)

    if num_sample > 1:
      logit = torch.log(F.softmax(logit, dim=2).mean(0))

    if return_feat:
        embedding = embed[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1)
        if masks is not None:
          embedding = embedding.sum(-2) # TODO * masks.unsqueeze(-1)).sum(dim=-2) / masks.sum(-1).unsqueeze(-1) 
        else:
          embedding = embedding.sum(-2)
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

      if n != 1:
          x = expand(x)
      encoding = F.gumbel_softmax(x, tau=temp)

      return encoding

  def weight_init(self):
      pass

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
        
class Reshape(nn.Module):
  def __init__(self, out_size):
    super(Reshape, self).__init__()
    self.out_size = out_size

  def forward(self, x):
    return x.contiguous().view(-1, *self.out_size)

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
