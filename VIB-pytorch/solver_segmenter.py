# Part of the code modified from https://github.com/felixkreuk/UnsupSeg.git
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from scipy.signal import find_peaks
from tqdm import tqdm
import json
import os
from pyhocon import ConfigFactory
import argparse
from datasets.datasets import return_data

NULL = "###NULL###"
def detect_peaks(x, lengths, prominence=0.1, width=None, distance=None):
    """detect peaks of next_frame_classifier
    
    Arguments:
        x {Tensor} -- batch of confidence per time
    """ 
    out = []

    for xi, li in zip(x, lengths):
        if type(xi) == torch.Tensor:
            xi = xi.cpu().detach().numpy()
        xi = xi[:li]  # shorten to actual length
        xmin, xmax = xi.min(), xi.max()
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)

        if len(peaks) == 0:
            peaks = np.array([len(xi)-1])

        out.append(peaks)

    return out


class PrecisionRecallMetric:
    def __init__(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.eps = 1e-5
        self.data = []
        self.tolerance = 2
        self.prominence_range = np.arange(0, 0.15, 0.01)
        self.width_range = [None, 1]
        self.distance_range = [None, 1]

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        EPS = 1e-7
        
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval

    def zero(self):
        self.data = []

    def update(self, seg, pos_pred, length):
        for seg_i, pos_pred_i, length_i in zip(seg, pos_pred, length):
            self.data.append((seg_i, pos_pred_i.cpu().detach().numpy(), length_i.item()))

    def get_stats(self, width=None, prominence=None, distance=None):
        print(f"calculating metrics using {len(self.data)} entries")
        max_rval = -float("inf")
        best_params = None
        best_peaks = None
        segs = list(map(lambda x: x[0], self.data))
        length = list(map(lambda x: x[2], self.data))
        yhats = list(map(lambda x: x[1], self.data))

        width_range = self.width_range
        distance_range = self.distance_range
        prominence_range = self.prominence_range

        # when testing, we would override the search with specific values from validation
        if prominence is not None:
            width_range = [width]
            distance_range = [distance]
            prominence_range = [prominence]

        for width in width_range:
            for prominence in prominence_range:
                for distance in distance_range:
                    precision_counter = 0
                    recall_counter = 0
                    pred_counter = 0
                    gt_counter = 0
                    peaks = detect_peaks(yhats,
                                         length,
                                         prominence=prominence,
                                         width=width,
                                         distance=distance)

                    for (y, yhat) in zip(segs, peaks):
                        for yhat_i in yhat:
                            min_dist = np.abs(y - yhat_i).min()
                            precision_counter += (min_dist <= self.tolerance)
                        for y_i in y:
                            min_dist = np.abs(yhat - y_i).min()
                            recall_counter += (min_dist <= self.tolerance)
                        pred_counter += len(yhat)
                        gt_counter += len(y)

                    p, r, f1, rval = self.get_metrics(precision_counter,
                                                      recall_counter,
                                                      pred_counter,
                                                      gt_counter)
                    if rval > max_rval:
                        max_rval = rval
                        best_params = width, prominence, distance
                        out = (p, r, f1, rval)
                        best_peaks = peaks
        self.zero()
        print(f"best peak detection params: {best_params} (width, prominence, distance)")
        return out, best_params, best_peaks


class StatsMeter:
    def __init__(self):
        self.data = []

    def update(self, item):
        if type(item) == list:
            self.data.extend(item)
        else:
            self.data.append(item)

    def get_stats(self):
        data = np.array(self.data)
        mean = data.mean()
        self.zero()
        return mean

    def zero(self):
        self.data.clear()
        assert len(self.data) == 0, "StatsMeter didn't clear"


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.msg} -- started")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"{self.msg} -- done in {(time.time() - self.start_time)} secs")


class CPCSegmenter:
  """
  Perform unsupervised phoneme segmentation using CPC feature as in
      
      Self-Supervised Contrastive Learning for Unsupervised Phoneme       
      Segmentation. 
      Felix Kreuk, Joseph Keshet, Yossi Adi 
      INTERSPEECH 2020
  and
      Bhati et. al., "Segmental Contrastive Predictive Coding
      for Unsupervised Word Segmentation". 2021.
  """ 
  def __init__(self, config):    
    self.data_loader = return_data(config) 
    self.metric = PrecisionRecallMetric()  
    self.prominence = config.get("prominence", 0.1) 
    self.width = config.get("width", None)
    self.distance = config.get("distance", None)
    self.do_val = config.get("validate", False)
    self.debug = config.get("debug", False)
    self.val_ratio = config.get("val_ratio", 0.1)
    self.frames_per_sec = 100
    self.result_dir = config.ckpt_dir
    if not os.path.exists(self.result_dir):
      os.makedirs(self.result_dir)

  def score(self, u, v):
    return F.cosine_similarity(u, v, dim=-1)

  def validate(self):
    timer = Timer("Boundary detection on validation set") 
    val_loader = self.data_loader["train"]
    valset = val_loader.dataset
    batch_size = val_loader.batch_size
    n_batches = round(len(val_loader) * self.val_ratio)

    audio_ids = []
    for b_idx, batch in enumerate(val_loader):
      if b_idx > 2 and self.debug:
        break
      if b_idx >= n_batches:
        break
      x = batch[0]
      input_mask = batch[3] 
      lengths = input_mask.sum(-1).long() 
      
      # Compute cosine similarities between adjacent features
      pos_preds = self.score(x[:-1], x[1:]) 
      
      # Update evaluator
      segs = []
      B = x.size(0)
      for idx in range(B):
        global_idx = b_idx * batch_size + idx
        audio_path = valset.dataset[global_idx][0]
        audio_id = os.path.basename(audio_path).split(".")[0]
        audio_ids.append(audio_id) 

        phonemes = testset.dataset[global_idx][2]
        seg_sec = [t for phn in phonemes for t in [phn["begin"], phn["end"]]]
        seg_sec = sorted(set(seg_sec))
        seg = [int(sec*self.frames_per_sec) for sec in seg_sec]
        segs.append(seg)
        
      self.metric.update(segs, pos_preds, lengths) 
    
    # Evaluation using boundary F1 and Rval
    (precision, recall, f1, rval), best_params, _ = self.metric.get_stats()
    info = f"Validation Boundary Precision: {precision:.4f}\tBoundary Recall: {recall:.4f}\tBoundary F1: {f1:.4f}\tR value: {rval:.4f}"
    with open(os.path.join(self.result_dir, "result_file.txt"), "a") as f:
      f.write(info+"\n")
    print(info)

    self.metric.zero()
    return best_params 

  def predict(self):
    """
    Returns:
        boundary_dict: a dict with fields
            [utt_id]: a list of dicts of
                "begin": float, segment begin time in sec
                "end": float, segment end time in sec
                "text": str, NULL
    """
    if self.do_val:
      self.width, self.prominence, self.distance = self.validate()
    timer = Timer("Boundary detection on test set")      
    boundary_dict = dict()
    test_loader = self.data_loader["test"]
    testset = test_loader.dataset
    batch_size = test_loader.batch_size

    audio_ids = []
    for b_idx, batch in enumerate(test_loader): 
      if b_idx > 2 and self.debug:
        break
      x = batch[0]
      input_mask = batch[3] 
      lengths = input_mask.sum(-1).long() 
      
      # Compute cosine similarities between adjacent features
      pos_preds = self.score(x[:-1], x[1:]) 
      
      # Update evaluator
      segs = []
      B = x.size(0)
      for idx in range(B):
        global_idx = b_idx * batch_size + idx
        audio_path = testset.dataset[global_idx][0]
        audio_id = os.path.basename(audio_path).split(".")[0]
        audio_ids.append(audio_id)

        phonemes = testset.dataset[global_idx][2]
        seg_sec = [t for phn in phonemes for t in [phn["begin"], phn["end"]]]
        seg_sec = sorted(set(seg_sec))
        seg = [int(sec*self.frames_per_sec) for sec in seg_sec]
        segs.append(seg)
        
      self.metric.update(segs, pos_preds, lengths) 

    # Evaluation using boundary F1 and Rval
    (precision, recall, f1, rval), _, peaks = self.metric.get_stats(prominence=self.prominence,
                                                      width=self.width, 
                                                      distance=self.distance)
    info = f"Test Boundary Precision: {precision:.4f}\tBoundary Recall: {recall:.4f}\tBoundary F1: {f1:.4f}\tR value: {rval:.4f}"
    with open(os.path.join(self.result_dir, "result_file.txt"), "a") as f:
      f.write(info+"\n")
    print(info)

    for audio_id, peak in zip(audio_ids, peaks):
      if peak[0]:
        peak = np.append(0, peak)
      
      boundary_dict[audio_id] = []
      for begin, end in zip(peak[:-1], peak[1:]):
        begin_sec = float(begin) / self.frames_per_sec
        end_sec = float(end) / self.frames_per_sec
        boundary_dict[audio_id].append({"begin": begin_sec,
                                        "end": end_sec,
                                        "text": NULL})
    json.dump(boundary_dict, open(os.path.join(self.result_dir, f"boundary_detection.json"), "w"), indent=2)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("CONFIG", type=str)
  args = parser.parse_args()

  config = ConfigFactory.parse_file(args.CONFIG)
  solver = CPCSegmenter(config)
  solver.predict()
  
if __name__ == "__main__":
  main()
