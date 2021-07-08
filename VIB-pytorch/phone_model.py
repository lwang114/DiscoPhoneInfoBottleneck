import torch
import torch.nn as nn

class UnigramPronunciator(nn.Module):
  """
  A simple unigram model for unsupervised pronunciation modeling
  """ 
  def __init__(self, 
               vocab,
               phone_set,
               ignore_index=-100):
    super(UnigramPronunciator, self).__init__()
    self.n_word_class = len(vocab)
    self.n_phone_class = len(phone_set)
    self.vocab = vocab
    self.phone_set = phone_set
    self.ignore_index = ignore_index

    pron_counts = torch.zeros((self.n_word_class, self.n_phone_class))
    self.register_buffer('pron_counts', pron_prob)

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
          self.pron_counts[x, y] = self.pron_counts[x, y] + 1.

  @property
  def pronounce_prob(self):
    return self.pron_counts / torch.maximum(self.pron_counts.sum(1, keepdim=True), 1) 

  def forward(self, x):
    x = np.where(x != self.ignore_index,
                 x, torch.tensor(0, dtype=torch.long, device=x.device)) 
    return self.pronounce_prob[x]
  
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
