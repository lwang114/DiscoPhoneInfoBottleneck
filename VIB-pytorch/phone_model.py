import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from hmm_word_discoverer import HMMWordDiscoverer

class HMMPronunciator(nn.Module):
  def __init__(self, 
               vocab,
               phone_set,
               ignore_index=-100,
               config):
    super(HMMPronunciator, self).__init__()
    self.n_word_class = len(vocab)
    self.n_phone_class = len(phone_set)
    self.vocab = vocab
    self.phone_set = phone_set
    self.word_stoi = {w:i for i, w in enumerate(vocab)}
    self.phone_stoi = {phn:i for i, phn in enumerate(phone_set)}
    self.ignore_index = ignore_index
    pretrained_model = config.get('pronounce_prob_path', None) 
    if pretrained_model:
      pron_counts = json.load(open(pretrained_model))
      pron_counts = torch.FloatTensor(pron_counts)
    else:
      pron_counts = torch.zeros((self.n_word_class, self.n_phone_class))
    self.register_buffer('pron_counts', pron_counts)
    self.model = HMMWordDiscoverer(config) # TODO 
    self.config = config

  def update(self, words, phones):
    return
  
  def pronounce_prob(self):
    norm = self.pron_counts.sum(1, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.tensor(1., device=norm.device))
    return self.pron_counts / norm

  def train(num_iter=10):
    self.model.trainUsingEM(num_iter, writeModel=True) 
    trans = self.model.trans
    for w in trans:
      for phn in trans[w]:
        word_idx = self.word_stoi[w]
        phn_idx = self.phone_stoi[phn]
        self.pron_counts[word_idx, phn_idx] = trans[w][phn] 
    self.save_readable(filename=os.path.join(self.config['ckpt_dir'], 
                                             'pronounce_prob.json'))

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

class UnigramPronunciator(nn.Module):
  """
  A simple unigram model for unsupervised pronunciation modeling
  """ 
  def __init__(self, 
               vocab,
               phone_set,
               ignore_index=-100,
               config=None):
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

class PositionDependentUnigramPronunciator(nn.Module):
  """
  Position-dependent unigram model for unsupervised pronunciation modeling
  """ 
  def __init__(self, 
               vocab,
               phone_set,
               ignore_index=-100,
               max_phone_num=20):
    super(UnigramPronunciator, self).__init__()
    self.n_word_class = len(vocab)
    self.n_phone_class = len(phone_set)
    self.max_phone_num = max_phone_num
    self.vocab = vocab
    self.phone_set = phone_set
    self.ignore_index = ignore_index

    pron_counts = torch.zeros((self.n_word_class, self.max_phone_num, self.n_phone_class))
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
        for pos, phn in enumerate(phone):
          if phn == self.ignore_index:
            continue
          self.pron_counts[wrd, pos, phn] = self.pron_counts[wrd, pos, phn] + 1.

  def pronounce_prob(self):
    EPS = 1e-10
    norm = self.pron_counts.sum(dim=-1, keepdim=True)
    return self.pron_counts / (norm + EPS)
                                         
  def forward(self, x):
    x = torch.where(x != self.ignore_index,
                    x, torch.tensor(0, dtype=torch.long, device=x.device)) 
    return self.pronounce_prob()[x]
  
  def save_readable(self, filename='pronounce_prob.json'):
    pron_counts = self.pron_counts.cpu().detach().numpy().tolist()
    pron_prob = self.pronounce_prob().cpu().detach().numpy().tolist()
    pron_dict = dict()
    for i, v in enumerate(self.vocab):
      pron_dict[v] = {pos:dict() for pos in range(self.max_phone_num) if pron_counts[i][pos].sum() > 0}
      for pos in range(len(pron_dict[v])):
        for j, phn in enumerate(self.phone_set):
          pron_dict[v][pos][phn] = {'count': pron_counts[i][pos][j],
                                    'prob': pron_prob[i][pos][j]}
    json.dump(pron_dict, open(filename, 'w'), indent=2)

class LinearPositionAligner(nn.Module):
  """
  Map a posterior sequence to a given length by a softmax kernel linear mapping
  """
  def __init__(self, scale=1.):
    super(LinearPositionAligner, self).__init__()
    self.scale = scale

  def forward(self, x, input_mask, output_mask):
    """
    Args :
        x : FloatTensor of size (batch size, max input len, num. of classes),
        input_mask : FloatTensor of size (batch size, max input len),
        output_mask : FloatTensor of size (batch size, max output len)
    
    Returns :
        output : FloatTensor of size (batch size, max output len, num. of classes) 
    """
    pos_prob = self.position_probability(input_mask, output_mask)
    output = torch.matmul(pos_prob, x.float())
    return output 
  
  def position_probability(self, input_mask, output_mask):
    EPS = 1e-10
    device = input_mask.device
    L = input_mask.size(1)
    T = output_mask.size(1)
    ds_ratios = input_mask.sum(-1) / (output_mask.sum(-1) + EPS)

    # (batch size, T, L)
    mask = input_mask.unsqueeze(-2) * output_mask.unsqueeze(-1)
    in_pos = torch.arange(L, device=device).view(1, 1, -1)
    out_pos = ds_ratios.view(-1, 1, 1) * torch.arange(T, device=device).view(1, -1, 1)

    # (batch size, T, L)
    distances = self.scale * torch.abs(in_pos - out_pos) * mask + 1e9 * (1 - mask)

    prob = F.softmax(-distances, dim=-1)
    return prob
