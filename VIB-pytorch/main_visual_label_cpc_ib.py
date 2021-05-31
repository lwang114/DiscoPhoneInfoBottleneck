import numpy as np
import torch
import argparse
from utils import str2bool
from solver_visual_label_cpc_ib import Solver
import pandas as pd
import sys
import os
from pyhocon import ConfigFactory

def main(argv):
  parser = argparse.ArgumentParser(description='Visual label information bottleneck')
  parser.add_argument('CONFIG', type=str)
  config = parser.parse_config.argv)

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  
  config = ConfigFactory.parse_file(config.config)
  if not config.dset_dir:
    if config.dataset == 'FLICKR_WORD_IMAGE': 
      config.dset_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr/flickr8k_word_{min_class_size}'
    else:
      config.dset_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  
  if not config.image_model_dir:
    if config.dataset == 'FLICKR_WORD_IMAGE':
      config.image_model_dir = 'checkpoints/image_classification_ce_minfreq500/'
    else:
      config.image_model_dir = 'checkpoints/image_classification_mscoco_ce/'

  seed = config.seed
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

  np.set_printoptions(precision=4)
  torch.set_printoptions(precision=4)

  print()
  print('[ARGUMENTS]')
  print(config.
  print()

  net = Solver(config)

  if config.mode == 'train': 
    net.train()
  elif config.mode == 'test':
    net.test(save_ckpt=False, compute_abx=True)
  elif config.mode == 'train_sweep':
    config.beta = 1.
    config.epoch = 25
    df_results = {'Model': [],
                  'Loss': [],
                  r'$\beta$': [],
                  'Token F1': [],
                  'ABX': [],
                  'WER': []}
    for _ in range(4):
      net = Solver(config.
      net.train()
      df_results['Model'].append(config.model_type)
      df_results['Loss'].append(config.loss_type)
      df_results[r'$\beta$'].append(config.beta)
      df_results['Token F1'].append(net.history['token_f1'])
      df_results['WER'].append(1.-net.history['acc'])
      df_results['ABX'].append(net.history['abx'])
      config.beta /= 10

    df_results = pd.DataFrame(df_results)
    df_results.to_csv(os.path.join('checkpoints', config.env_name, 'results.csv'))
  else : return 0

if __name__ == '__main__':
  argv = sys.argv[1:]
  main(argv)
