import numpy as np
import torch
import argparse
from utils import str2bool
from solver_image import Solver
import pandas as pd
import os


def main(args):
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

  seed = args.seed
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

  np.set_printoptions(precision=4)
  torch.set_printoptions(precision=4)

  print()
  print('[ARGUMENTS]')
  print(args)
  print()

  net = Solver(args)
  train_loader = net.train_loader
  test_loader = net.test_loader

  if args.mode == 'train': 
    net.train(train_loader, test_loader)
  elif args.mode == 'test': 
    net.test(test_loader, out_prefix='train_predictions')
    net.test(train_loader, out_prefix='test_predictions')
  else: Exception(f'Mode can only be from {train, test}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Image classification')
  parser.add_argument('--seed', default=1, type=int)
  parser.add_argument('--data_path', default='/home/lwang114/data/flickr/')
  parser.add_argument('--exp_dir', default='checkpoints/image_classification')
  parser.add_argument('--feature_type', choices={'res34', 'rcnn'})
  parser.add_argument('--loss_type', choices={'ce', 'bce'}, default='bce')
  parser.add_argument('--mode', choices={'train', 'test'}, default='train')
  parser.add_argument('--epoch', default=10, type=int)
  parser.add_argument('--min_class_size', default=50, type=int)
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--pos_weight', default=1., type=int)
  args = parser.parse_args()
  main(args)
