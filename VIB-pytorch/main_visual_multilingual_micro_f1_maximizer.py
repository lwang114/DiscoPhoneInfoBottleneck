import numpy as np
import torch
import argparse
from utils.utils import str2bool
from solver_visual_multilingual_micro_f1_maximizer import Solver
from pyhocon import ConfigFactory
import sys
import os

def main(argv):
  parser = argparse.ArgumentParser(description='Visual multilingual micro token F1 maximizer')
  parser.add_argument('CONFIG', type=str)
  args = parser.parse_args(argv)

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

  config = ConfigFactory.parse_file(args.CONFIG)
  if not config.dset_dir:
    config.dset_dir = "/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic" 
  
  seed = config.seed
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

  np.set_printoptions(precision=4)
  torch.set_printoptions(precision=4)

  print()
  print('[CONFIGS]')
  print(config)
  print()

  net = Solver(config)
  save_embedding = config.get('save_embedding', False)
  if config.mode == 'train':
    net.train(save_embedding=save_embedding)
  elif config.mode == 'test':
    net.test(save_embedding=save_embedding) 
  else:
    return 0

if __name__ == '__main__':
  argv = sys.argv[1:]
  main(argv)
