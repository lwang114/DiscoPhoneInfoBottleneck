import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import fairseq
import os
import json
import time
import pyhocon
from itertools import groupby
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans
from utils.utils import cuda
from utils.evaluate import compute_token_f1, compute_boundary_f1
from datasets.datasets import return_data
from datasets.librispeech import *


EPS = 1e-10
class Solver(object):

  def __init__(self, config):
    self.debug = config['debug']
    self.config = config 
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.get_feature_config(config)
    self.get_dataset_config(config)
    self.ckpt_dir = Path(config['ckpt_dir'])
    if not self.ckpt_dir.exists():
      self.ckpt_dir.mkdir()

  def get_feature_config(self, config):
    self.audio_feature = config['audio_feature']
    if config['audio_feature'] in ['wav2vec', 'wav2vec2', 'vq-wav2vec']:
      cp = config['wav2vec_path']
      self.audio_feature_net = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])[0][0].to(self.device)
      if config['audio_feature'] == 'wav2vec2':
        self.up_ratio = 2
      else:
        self.up_ratio = 1
    else:
      raise ValueError('Invalid audio feature {config["audio_feature"]}')

  def get_dataset_config(self, config):
    self.use_segment = config['use_segment']
    self.data_loader = return_data(config) 

  @torch.no_grad()
  def cluster(self, filename):
    testset = self.data_loader['test'].dataset
    n_clusters = self.config.get('n_clusters', 39)
    batch_size = self.data_loader['train'].batch_size
    
    utt_ids_dict = dict()
    X_dict = dict()
    segment_dict = dict()
    pred_segment_dict = dict()
    
    for split in ['train', 'test']:
      num_batches = len(self.data_loader[split])
      utt_ids_dict[split] = []
      X_dict[split] = []
      segment_dict[split] = dict()
      pred_segment_dict[split] = dict()
      progress = tqdm(total=num_batches, ncols=160, desc=f'Collecting segmental features for the {split} set')
      for b_idx, batch in enumerate(self.data_loader[split]):
        if self.debug and len(X_dict[split]) > 60:
          break
        audio_inputs = batch[0].to(self.device)
        audio_mask = batch[3].to(self.device)
        indices = batch[-1]

        z = self.audio_feature_net.feature_extractor(audio_inputs)
        if self.audio_feature == 'wav2vec':
          z = self.audio_feature_net.feature_aggregator(z)
        x = z.permute(0, 2, 1)
        if self.use_segment:
          x = torch.matmul(audio_mask, x)
        audio_lens = torch.div(
          audio_mask.sum(-1),
          160 * self.up_ratio,
          rounding_mode='trunc'
        ).long()

        for idx, global_idx in enumerate(indices):
          utt_id = os.path.basename(self.data_loader[split].dataset.dataset[global_idx][0]).split('.')[0]
          embed = x[idx, :audio_lens[idx]].cpu().detach().numpy()
          if np.isnan(embed).any() or np.isinf(embed).any():
            print(f'NaN or inf detected for {utt_id}, audio_lens, np.isnan(x[idx])',\
                  audio_lens[idx], np.isnan(x[idx].detach().cpu().numpy()).any())
          X_dict[split].extend(embed.tolist())
          utt_ids_dict[split].extend([utt_id]*embed.shape[0])
          segment_dict[split][utt_id] = self.data_loader[split].dataset.dataset[global_idx][-1]
          pred_segment_dict[split][utt_id] = self.data_loader[split].dataset.dataset[global_idx][-2]
        progress.update(1)
      progress.close()
      X_dict[split] = np.asarray(X_dict[split])

    begin_time = time.time()   
    clusterer = KMeans(n_clusters=n_clusters).fit(X_dict['train'])
    print(f'Clustering takes {time.time()-begin_time}s to finish')
 
    for split in ['train', 'test']:
        num_examples = len(self.data_loader[split].dataset)
        segment_dir = Path(self.ckpt_dir / f'predicted_segmentations/{split}') 
        if not segment_dir.exists():
            segment_dir.mkdir(parents=True, exist_ok=True)

        phone_file = self.ckpt_dir / f'{split}_{filename}'
        gold_phone_file = os.path.join(
            testset.data_path,
            f'{testset.splits[0]}/{testset.splits[0]}_nonoverlap.item' 
        )
        phone_f = open(phone_file, 'w')
        ys = clusterer.predict(X_dict[split])
        utt_ids = utt_ids_dict[split]
        segments = segment_dict[split] 
        pred_segments = pred_segment_dict[split]
        
        peaks = []
        gold_peaks = []
        progress = tqdm(total=num_examples, ncols=160, desc=f'Segmenting the {split} set')
        for utt_id, group in groupby(list(zip(utt_ids, ys)), lambda x:x[0]):
          y = torch.LongTensor([g[1]+1 for g in group])
          if self.use_segment:
            y_unseg = testset.unsegment(y, pred_segments[utt_id]).long().cpu().detach().numpy().tolist()
            utt_begin = segments[utt_id][0]['begin']
            pred_utt_begin = pred_segments[utt_id][0]['begin']
            peaks.append(
              [seg['end'] - pred_utt_begin for seg in pred_segments[utt_id][:-1]]
            )
            gold_peaks.append(
              [seg['end'] - utt_begin for seg in segments[utt_id][:-1]]
            )
          else:
            utt_begin = segments[utt_id][0]['begin']
            dur = segments[utt_id][-1]['end'] - utt_begin
            y_unseg = y.long().detach().cpu().numpy().tolist()
            peak = []
            for i, (y_cur, y_next) in enumerate(zip(y_unseg, y_unseg[1:]+[-1])):
              if y_cur != y_next:
                t = i * self.up_ratio / 100
                if t > dur:
                  break
                peak.append(t)
            peak[0] = 0
            if peak[-1] != dur:
              peak[-1] = round(dur, 3)
            peaks.append(peak[1:-1])

            gold_peak = [round(seg['end']-utt_begin, 3) for seg in segments[utt_id]]
            gold_peaks.append(gold_peak[:-1])
            if self.debug:
              print(utt_id, 'gold: ', gold_peaks[-1])
              print(utt_id, 'pred: ', peaks[-1])            
            with open(segment_dir / f'{utt_id}.txt', 'w') as seg_f:
              seg_f.write('Predicted: '+' '.join([str(p) for p in peak])+'\n')
              seg_f.write('Gold: '+' '.join([str(g) for g in gold_peak]))
                
          y_str = ' '.join([str(l) for l in y_unseg])
          phone_f.write(f'{utt_id} {y_str}\n') 
          progress.update(1)
        phone_f.close()
        progress.close()
        if split == 'test':
          token_f1,\
          token_prec,\
          token_recall = compute_token_f1(
            phone_file, gold_phone_file,
            self.ckpt_dir / 'confusion.png'
          )
          info = f'{split} set result\tToken precision: {token_prec*100:.2f}\tToken recall: {token_recall*100:.2f}\tToken F1: {token_f1*100:.2f}'
          with open(self.ckpt_dir / 'results.txt', 'a') as f_result:
            f_result.write(info+'\n')
          print(info)
        boundary_prec, boundary_rec, boundary_f1 = compute_boundary_f1(peaks, gold_peaks)
        info = f'{split} set result\tBoundary precision: {boundary_prec*100:.2f}\tBoundary recall: {boundary_rec*100:.2f}\tBoundary F1: {boundary_f1*100:.2f}'
        with open(self.ckpt_dir / 'results.txt', 'a') as f_result:
          f_result.write(info+'\n')
        print(info)

  @torch.no_grad()
  def predict(self, filename):
    testset = self.data_loader['test'].dataset
    peaks = []
    gold_peaks = []
    if self.audio_feature in ['wav2vec', 'wav2vec2']:
      self.cluster(filename)
    else:
      group2int = dict()  
      for split in ['train', 'test']:
          if not self.debug:
            f_out = open(self.ckpt_dir / f'{split}_{filename}', 'w')
          segment_dir = Path(self.ckpt_dir / f'predicted_segmentations/{split}') 
          if not segment_dir.exists():
            segment_dir.mkdir(parents=True, exist_ok=True)

          progress = tqdm(total=len(self.data_loader[split].dataset), ncols=160, desc=f'Segmenting the {split} set')
          for b_idx, batch in enumerate(self.data_loader[split]): 
            if b_idx > 1 and self.debug:
              break
            audio_inputs = batch[0].to(self.device)
            audio_mask = batch[3].to(self.device)
            indices = batch[-1]

            z = self.audio_feature_net.feature_extractor(audio_inputs) 
            _, cluster_idxs = self.audio_feature_net.vector_quantizer.forward_idx(z)
           
            for i, (groups, global_idx) in enumerate(zip(cluster_idxs, indices)):
              audio_id = testset.dataset[global_idx][0].split('/')[-1].split('.')[0] 
              segments = testset.dataset[global_idx][-1]
              group_keys = groups.detach().cpu().numpy().tolist()
              group_keys = group_keys[:audio_lens[i]]
              k_idxs = []
              for k in group_keys:
                if not tuple(k) in group2int:
                  group2int[tuple(k)] = len(group2int)
                k_idxs.extend([str(group2int[tuple(k)])]*self.up_ratio)
              
              peak = []
              for t, (k_cur, k_next) in enumerate(zip(k_idxs, k_idxs[1:]+[-1])):
                if k_cur != k_next:
                  peak.append(float(t * self.up_ratio) / 100)
              gold_peak = [round(seg['end']-segments[0]['begin'], 3) for seg in segments]
              peaks.append(peak[1:-1])
              gold_peaks.append(gold_peak[:-1])
              if self.debug:  
                print(audio_id, 'gold: ', gold_peaks[-1])
                print(audio_id, 'pred: ', peaks[-1])
              with open(segment_dir / f'{audio_id}.txt', 'w') as f_seg:
                f_seg.write('Predicted: '+' '.join([str(p) for p in peak])+'\n')
                f_seg.write('Gold: '+' '.join([str(g) for g in gold_peak]))
              f_out.write(f'{audio_id} '+' '.join(k_idxs)+'\n')
              progress.update(1)
          f_out.close() 
          progress.close()
          boundary_prec, boundary_rec, boundary_f1 = compute_boundary_f1(peaks, gold_peaks)
          info = f'{split} set result\tBoundary precision: {boundary_prec*100:.2f}\tBoundary recall: {boundary_rec*100:.2f}\tBoundary F1: {boundary_f1*100:.2f}'
          if not self.debug:
            with open(self.ckpt_dir / 'results.txt', 'a') as f_result:
              f_result.write(info+'\n')
      print(info)

def main():
  parser = ArgumentParser()
  parser.add_argument('config')
  parser.add_argument('--setting', '-s', default='39clusters')
  parser.add_argument('--filename', default='prediction.txt')
  args = parser.parse_args()

  config = pyhocon.ConfigFactory.parse_file(args.config)[args.setting]
  if config.debug:
    ckpt_dir = f'checkpoints/debug'
  else:
    ckpt_dir = f'checkpoints/{config.dataset}_{config.audio_feature}_{args.setting}'
  config.ckpt_dir = ckpt_dir
  config['ckpt_dir'] = ckpt_dir
  for seed in config.get('seeds', []):
    config.seed = seed
    config['seed'] = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    
    print()
    print('[CONFIG]')
    print(config)
    print()
    solver = Solver(config)
    solver.predict(args.filename)

if __name__ == '__main__':
  main() 
