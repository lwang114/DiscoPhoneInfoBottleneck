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
from itertools import groupby
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import hydra
import hydra.utils as utils

from datasets.librispeech import *
from vqcpc_model import Encoder


EPS = 1e-10
class Solver(object):
  def __init__(self, cfg):
    self.debug = cfg.debug
    self.cfg = cfg
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.get_model_cfg(cfg)
    self.get_dataset_cfg(cfg)
    self.ckpt_dir = Path(cfg.ckpt_dir)
    if not self.ckpt_dir.exists():
      self.ckpt_dir.mkdir()

  def get_model_cfg(self, cfg):
    print(f"Load checkpoint from {cfg['checkpoint']}")
    self.encoder = Encoder(**cfg.model.encoder).to(self.device)
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    self.encoder.load_state_dict(checkpoint["encoder"])
    self.encoder.eval()

  def get_dataset_cfg(self, cfg):
    preprocessor = LibriSpeechPreprocessor(
                          cfg.dset_dir, 80,
                          splits=cfg.splits,
                          audio_feature=cfg.audio_feature,
                          phone_label=cfg.phone_label,
                          ignore_index=cfg.ignore_index,
                          debug=cfg.debug)
    
    train_data = LibriSpeechDataset(
                       cfg.dset_dir, 
                       preprocessor,
                       'train',
                       splits=cfg.splits, 
                       augment=True,
                       audio_feature=cfg.audio_feature,
                       phone_label=cfg.phone_label,
                       use_segment=cfg.use_segment,
                       debug=cfg.debug) 

    test_data = LibriSpeechDataset(
                       cfg.dset_dir, 
                       preprocessor,
                       'test',
                       splits=cfg.splits, 
                       augment=True,
                       audio_feature=cfg.audio_feature,
                       phone_label=cfg.phone_label,
                       use_segment=cfg.use_segment,
                       debug=cfg.debug)
    self.data_loader = {'train': DataLoader(
                                  train_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=8),
                        'test': DataLoader(
                                  test_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=8)} 

  @torch.no_grad()
  def predict(self, filename): 
    testset = self.data_loader['test'].dataset
    f_out = open(self.ckpt_dir / filename, 'w')
    
    for b_idx, batch in tqdm(enumerate(self.data_loader['test'])): 
      if b_idx > 1 and self.debug:
        break
      audio_inputs = batch[0].to(self.device).permute(0, 2, 1)
      audio_mask = batch[3].to(self.device)
      audio_lens = audio_mask.sum(-1).long()

      _, _, cluster_idxs  = self.encoder.encode(audio_inputs) 
      cluster_idxs = cluster_idxs.cpu().numpy()
      for i, k_idxs in enumerate(cluster_idxs):
        global_idx = b_idx * self.cfg.batch_size + i
        audio_id = testset.dataset[global_idx][0].split('/')[-1].split('.')[0]
        k_str = [str(k) for k in k_idxs[:audio_lens[i]].tolist()]
        f_out.write(f'{audio_id} '+' '.join(k_str)+'\n')
    f_out.close()


@hydra.main(config_path="configs/vqcpc", config_name="encode")
def main(cfg):
  parser = ArgumentParser()
  parser.add_argument('--filename', default='prediction.txt')
  args = parser.parse_args()
  
  solver = Solver(cfg)
  solver.predict(args.filename)

if __name__ == '__main__':
  main() 
