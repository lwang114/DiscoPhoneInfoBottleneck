import numpy as np
import torch
import argparse
from utils import str2bool
from solver_retrieval_cpc_ib import Solver

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

  if args.mode == 'train' : net.train()
  elif args.mode == 'test' : net.test(save_ckpt=False, compute_abx=True)
  else : return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPC Pretraining')
    parser.add_argument('--epoch', default = 200, type=int, help='epoch size')
    parser.add_argument('--beta', default=1e-3, type=float, help='beta')
    parser.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
    parser.add_argument('--K', default = 256, type=int, help='dimension of encoding Z')
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--batch_size', default = 32, type=int, help='batch size')
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--dataset', default='MSCOCO2K_SEGMENT_IMAGE', type=str, help='dataset name')
    parser.add_argument('--model_type', choices={'gumbel_blstm', 'pyramidal_blstm', 'gumbel_markov_blstm', 'blstm', 'vq_blstm'}, default='gumbel_blstm')
    parser.add_argument('--dset_dir', default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
    parser.add_argument('--cuda',default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--mode',default='train', type=str, help='train or test')
    parser.add_argument('--n_predicts', type=int, default=12, help='number of prediction samples for CPC')
    parser.add_argument('--n_negatives', type=int, default=128, help='number of prediction samples for CPC')
    parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard')
    args = parser.parse_args()

    main(args)
