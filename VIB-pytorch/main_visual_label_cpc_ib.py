import numpy as np
import torch
import argparse
from utils import str2bool
from solver_cpc_ib import Solver
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

  if args.mode == 'train': 
    net.train()
  elif args.mode == 'test':
    net.test(save_ckpt=False, compute_abx=True)
  elif args.mode == 'train_sweep':
    args.beta = 1.
    args.epoch = 25
    df_results = {'Model': [],
                  'Loss': [],
                  r'$\beta$': [],
                  'Token F1': [],
                  'ABX': [],
                  'WER': []}
    for _ in range(4):
      net = Solver(args)
      net.train()
      df_results['Model'].append(args.model_type)
      df_results['Loss'].append(args.loss_type)
      df_results[r'$\beta$'].append(args.beta)
      df_results['Token F1'].append(net.history['token_f1'])
      df_results['WER'].append(1.-net.history['acc'])
      df_results['ABX'].append(net.history['abx'])
      args.beta /= 10

    df_results = pd.DataFrame(df_results)
    df_results.to_csv(os.path.join('checkpoints', args.env_name, 'results.csv'))
  else : return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPC Pretraining')
    parser.add_argument('--epoch', default = 200, type=int, help='epoch size')
    parser.add_argument('--beta', default=1e-3, type=float, help='beta')
    parser.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
    parser.add_argument('--K', default = 256, type=int, help='dimension of encoding Z')
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--dataset', default='MSCOCO2K', choices={'MSCOCO2K', 'FLICKR'}, type=str, help='dataset name')
    parser.add_argument('--loss_type', choices={'IB-only', 'IB+CPC', 'IB+CPC+VQ', 'CPC-only'})
    parser.add_argument('--model_type', choices={'gumbel_blstm', 'pyramidal_blstm', 'gumbel_markov_blstm'}, default='gumbel_blstm')
    parser.add_argument('--cpc_feature', choices={'rnn', 'bottleneck'}, default='rnn')
    parser.add_argument('--image_feature', choices={'label', 'soft_label', 'multi_label', 'rcnn', 'res34'}, default='label')
    parser.add_argument('--image_model_dir', type=str, default='flickr_image_classification')
    parser.add_argument('--dset_dir', default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    parser.add_argument('--ckpt_dir', default='checkpoints/phone_discovery_visual_label_cpc_ib', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
    parser.add_argument('--mode',default='train', type=str, choices={'train', 'test', 'train_sweep'}, help='train or test')
    parser.add_argument('--n_predicts', type=int, default=6, help='number of prediction samples for CPC')
    parser.add_argument('--n_negatives', type=int, default=17, help='number of prediction samples for CPC')
    parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard')
    args = parser.parse_args()

    main(args)
