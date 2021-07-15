import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils.utils import cuda

class BLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65,
               input_size=80, 
               ds_ratio=1,
               bidirectional=True):
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

  def forward(self, x,
              masks=None):
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
    return embed

class GumbelBLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65, 
               n_gumbel_units=49,
               input_size=80, 
               ds_ratio=1,
               bidirectional=True):
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
    self.decode = nn.Linear(n_gumbel_units, self.n_class, bias=False)

  def init_states(self, device, batch_size=1):
    if self.bidirectional:
      h0 = torch.zeros((2 * self.n_layers, batch_size, self.K))
      c0 = torch.zeros((2 * self.n_layers, batch_size, self.K))
    else:
      h0 = torch.zeros((self.n_layers, batch_size, self.K))
      c0 = torch.zeros((self.n_layers, batch_size, self.K))
    h0 = h0.to(device)
    c0 = c0.to(device)
    return h0, c0

  def forward(self, x, 
              num_sample=1, 
              masks=None, 
              temp=1., 
              return_feat=False):
    ds_ratio = self.ds_ratio
    device = x.device
    assert x.dim() == 3
    x = x.permute(0, 2, 1)
    
    B = x.size(0)
    T = x.size(1)
    h0, c0 = self.init_states(device=device, batch_size=B)
    embed, _ = self.rnn(x, (h0, c0))
    in_logit = self.bottleneck(embed)
    
    if masks is not None:
      in_logit = in_logit * masks.unsqueeze(2)
    encoding = self.reparametrize_n(in_logit, num_sample, temp)    
    logit = self.decode(encoding)
    if num_sample > 1:
      logit = torch.log(F.softmax(logit, dim=2).mean(0))

    if return_feat:
      return in_logit, logit, encoding, embed
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

  def reverse_forward(self, y, ignore_index=-100):
    y = torch.where(y != ignore_index,
                    y, torch.tensor(0, dtype=torch.long, device=y.device))
    return F.softmax(self.decode.weight[y], dim=-1)

  def weight_init(self):
      pass

class Davenet(nn.Module):
    def __init__(self, input_dim, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(input_dim,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x, l=5, masks=None):
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
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        return x


class DotProductClassAttender(nn.Module):
  def __init__(self, 
               input_dim,
               hidden_dim,
               n_class):
    super(DotProductClassAttender, self).__init__()
    self.attention = nn.Linear(input_dim, n_class, bias=False)
    self.classifier = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                      )

  def forward(self, x, mask):
    """
    Args :
        x : FloatTensor of size (batch size, seq length, input size)
        mask : FloatTensor of size (batch size, seq length)
    """
    attn_weights = self.attention(x).permute(0, 2, 1)
    attn_weights = attn_weights * mask.unsqueeze(-2)
    attn_weights = torch.where(attn_weights != 0,
                               attn_weights,
                               torch.tensor(-1e10, device=x.device))
    
    # (batch size, n class, seq length)
    attn_weights = F.softmax(attn_weights, dim=-1)
    # (batch size, n class, input size)
    attn_applied = torch.bmm(attn_weights, x)
    # (batch size, n class)
    out = self.classifier(attn_applied).squeeze(-1)
    return out, attn_weights


class HMMPronunciator(nn.Module):
  def __init__(self, 
               vocab,
               phone_set,
               config,
               ignore_index=-100):
    super(HMMPronunciator, self).__init__()
    self.n_word_class = len(vocab)
    self.n_phone_class = len(phone_set)
    self.vocab = vocab
    self.phone_set = phone_set
    self.word_stoi = {w:i for i, w in enumerate(vocab)}
    self.phone_stoi = {phn:i for i, phn in enumerate(phone_set)}
    self.ignore_index = ignore_index
    pretrained_model = config.get('pronounce_model_path', None) 
    if pretrained_model:
      pron_counts = json.load(open(pretrained_model))
      pron_counts = torch.FloatTensor(pron_counts)
    else:
      pron_counts = torch.zeros((self.n_word_class, self.n_phone_class))
    self.register_buffer('pron_counts', pron_counts)
    self.model = HMMWordDiscoverer(config) 
    self.config = config

  def update(self, words, phones):
    pass
  
  def pronounce_prob(self):
    norm = self.pron_counts.sum(1, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.tensor(1., device=norm.device))
    return self.pron_counts / norm

  def train_model(self, num_iter=10):
    self.model.trainUsingEM(num_iter, writeModel=True) 
    obs = self.model.obs
    for w in obs:
      for phn in obs[w]:
        word_idx = self.word_stoi[w]
        phn_idx = self.phone_stoi[phn]
        self.pron_counts[word_idx, phn_idx] = obs[w][phn]
    self.save_readable(filename=os.path.join(self.config['ckpt_dir'], 
                                             'pronounce_prob.json'))

  def forward(self, x):
    x = torch.where(x != self.ignore_index,
                    x, torch.tensor(0, dtype=torch.long, device=x.device)) 
    return self.pronounce_prob()[x] 

  def save_readable(self, filename='pronounce_prob.json'):
    pron_counts = self.pron_counts.cpu().detach().numpy().tolist()
    pron_prob = self.pronounce_prob().cpu().detach().numpy().tolist()
    np.save(str(filename).split('.')[0]+'.npy', pron_prob)

    pron_dict = dict()
    for i, v in enumerate(self.vocab):
      pron_dict[v] = dict() 
      for j, phn in enumerate(self.phone_set):
        pron_dict[v][phn] = {'count': pron_counts[i][j],
                             'prob': pron_prob[i][j]}
    json.dump(pron_dict, open(filename, 'w'), indent=2)

class UnigramPronunciator(nn.Module):
  """
  A simple unigram model for unsupervised pronunciation modeling
  """ 
  def __init__(self, 
               vocab,
               phone_set,
               config,
               ignore_index=-100):
    super(UnigramPronunciator, self).__init__()
    self.n_word_class = len(vocab)
    self.n_phone_class = len(phone_set)
    self.vocab = vocab
    self.phone_set = phone_set
    self.ignore_index = ignore_index

    pron_counts = torch.zeros((self.n_word_class, self.n_phone_class))
    self.register_buffer('pron_counts', pron_counts)

  def update(self, words, phones):
    """
    Args :
        words : list of list of int word labels
        phones : list of list of int phone sequences
    """
    for word, phone in zip(words, phones):
      for wrd in word: 
        if wrd == self.ignore_index:
          continue
        for phn in phone:
          if phn == self.ignore_index:
            continue
          self.pron_counts[wrd, phn] = self.pron_counts[wrd, phn] + 1.
  
  def train_model(self):
    pass
  
  def pronounce_prob(self):
    norm = self.pron_counts.sum(1, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.tensor(1., device=norm.device))
    return self.pron_counts / norm
                                         

  def forward(self, x):
    x = torch.where(x != self.ignore_index,
                    x, torch.tensor(0, dtype=torch.long, device=x.device)) 
    return self.pronounce_prob()[x]
  
  def save_readable(self, filename='pronounce_prob.json'):
    pron_counts = self.pron_counts.cpu().detach().numpy().tolist()
    pron_prob = self.pronounce_prob().cpu().detach().numpy().tolist()
    pron_dict = dict()
    for i, v in enumerate(self.vocab):
      pron_dict[v] = dict() 
      for j, phn in enumerate(self.phone_set):
        pron_dict[v][phn] = {'count': pron_counts[i][j],
                             'prob': pron_prob[i][j]}
    json.dump(pron_dict, open(filename, 'w'), indent=2)



def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
