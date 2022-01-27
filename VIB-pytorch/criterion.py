import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentMutualInformationLoss(nn.Module):
  def __init__(self, min_length=-1, max_length=40):
    self.min_length = min_length
    self.max_length = max_length

  def forward(
      self, word_logits, 
      word_labels, segment_masks,
      phoneme_nums, segment_nums
    ):
    """ Maximize the mutual information I(S;Y|X)
    Args :
        word_logits : (batch_size, num. of segments, num. of words) FloatTensor,
        word_labels : int,
        segment_masks : (batch_size, num. of segments,) FloatTensor,
        phoneme_nums : list of ints,
        segment_nums : list of ints

    Returns :
        loss : FloatTensor
    """
    device = word_logits.device
    bsz = word_logits.size(0)
    losses = [
      self.compute_loss(
        word_logits=word_logits[i],
        label=word_labels[i],
        mask=segment_masks[i],
        seg_num=segment_nums[i],
        phn_num=phoneme_nums[i],
        device=device
      ) for i in range(bsz)
    ]
    return torch.mean(torch.stack(losses))
    
  def compute_loss(
      self, word_logits,  
      label, mask, 
      seg_num, phn_num, 
      device
    ):
    log_probs = F.log_softmax(word_logits, dim=-1)[:, label]
    I_SY_X = torch.zeros(phn_num+1, seg_num+1, device=device).log()
    I_SY_X[0, 0] = 0.0
    for i in range(1, phn_num+1):
      for end in range(i, seg_num+1):
        span_ids = [self.get_span_id(begin, end-1) for begin in range(end)]
        span_mask = [mask[self.get_span_id(begin, end-1)] for begin in range(end)]
        span_mask = torch.stack(span_mask)
        I_SY_X[i, end] = ((I_SY_X[i-1][:end] + log_probs[span_ids]) * span_mask).sum()
    return -I_SY_X[phn_num, seg_num]
  
  def get_span_id(self, begin, end):
    return int(end * (end + 1) / 2 + end - begin)
