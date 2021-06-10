import numpy as np
import torch

# TODO
def main(argv):
  parser = argparse.ArgumentParser(description='Visual multilingual information bottleneck')
  parser.add_argument('CONFIG', type=str)
  args = parser.parse_args(argv)

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

  config = ConfigFactory.parse_file(args.CONFIG)
  if not config.dset_dir:
    if config.dataset == 'LIBRISPEECH':


