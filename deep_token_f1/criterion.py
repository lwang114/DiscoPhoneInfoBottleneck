# Some code modified from https://github.com/zeakey/iccv2019-fmeasure/blob/master/pytorch/floss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class MicroTokenFLoss(nn.Module):
  def __init__(self, beta=0.3, log_like=False):
    super(MicroTokenFLoss, self).__init__()
    self.beta = beta
    self.log_like = log_like

  def forward(self, prediction, target, mask):
    """
    Args :
        prediction : FloatTensor of size (batch size, sequence length, num. of clusters)
        target : FloatTensor of size (batch size, sequence length, num. of classes) 
        mask : LongTensor of size (batch size, sequence length)

    Returns :
        floss : torch.Float, the negative micro token F1  
    """
    EPS = 1e-10
    prediction = prediction * mask.unsqueeze(-1)
    target = target * mask.unsqueeze(-1)

    TP = (prediction * target).sum() 
    H = self.beta * target.sum() + prediction.sum()
    fmeasure = (1 + self.beta) * TP / (H + EPS)
    if self.log_like:
      floss = -torch.log(fmeasure)
    else:
      floss = 1 - fmeasure 
    return floss

class MacroTokenFLoss(nn.Module):
  def __init__(self, beta=0.3, log_like=False):
    super(MacroTokenFLoss, self).__init__()
    self.beta = beta
    self.log_like = log_like

  def forward(self, prediction, target, mask):
    EPS = 1e-10
    K = prediction.size(-1)
    prediction = prediction * mask.unsqueeze(-1)
    target = target * mask.unsqueeze(-1)

    TP = (prediction * target).view(-1, K).sum(dim=0)
    H = self.beta * target.view(-1, K).sum(dim=0) + prediction.view(-1, K).sum(dim=0)
    fmeasure = (1 + self.beta) * TP / (H + EPS)
    fmeasure = fmeasure.mean()
    if self.log_like:
      floss = -torch.log(fmeasure)
    else:
      floss = 1 - fmeasure
    return floss

class WordLabelABXLoss(nn.Module):
  """
  Loss for weakly supervised ABX accuracy maximization with word-level supervision

  Args :
      beta : float, as in F-beta measure
      position_scale : float, scale factor when scoring the mean squared distance between two frames 
      match_prior : float, prior probability for two frames to match 
  """
  def __init__(self, 
               beta=1.,
               position_scale=0.01, 
               match_prior=0.1):
    super(WordLabelABXLoss, self).__init__()
    self.beta = beta
    self.scale = scale
    self.match_prior = match_prior

  def forward(self, triples, masks):
    """
    Args :
        triples : FloatTensor of size (batch size, 3, max sequence length, num. of clusters), where the second tensor is the positive example and the third is the negative example
        masks : FloatTensor of size (batch size, 3, max sequence length)

    Returns :
        loss : torch.Float, the negative abx f-beta score
    """
    device = triples.device
    max_seq_len = triples.size(2)
    distances = (torch.range(max_seq_len).unsqueeze(-1) - torch.range(max_seq_len).unsqueeze(-2)).pow(2).to(device)
    pos_pair_masks = masks[:, 0].unsqueeze(-1) * masks[:, 1].unsqueeze(-2)
    neg_pair_masks = masks[:, 0].unsqueeze(-1) * masks[:, 2].unsqueeze(-2)

    pos_prior_logits = - position_scale\
                       * (distances < 400).long()\
                       * distances
    pos_prior_probs = F.softmax(position_logits, dim=-1)  
    neg_prior_probs = self.match_prior * np.ones((max_seq_len, max_seq_len), device=device)

    pos_match_counts = torch.matmul(triples[:, 0], triples[:, 1].permute(0, 2, 1)) 
    neg_match_counts = torch.matmul(triples[:, 0], triples[:, 1].permute(0, 2, 1))

    tp = (pos_match_counts * pos_prior_probs * pos_pair_masks) + (neg_match_counts * neg_prior_probs * neg_pair_masks) 
    tp_plus_fp = (pos_match_counts * pos_pair_masks + neg_match_counts * neg_pair_masks).sum()
    n_true = (pos_prior_probs * pos_pair_masks + neg_prior_probs * neg_pair_masks).sum() 

    f_beta = (1 + self.beta.pow(2)) * tp\
             / (self.beta.pow(2) * n_true + tp_plus_fp)
    return -f_beta

