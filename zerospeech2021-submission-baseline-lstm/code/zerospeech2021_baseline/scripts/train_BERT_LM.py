import argparse


def # TODO 


def parse_args(argv):
  # Run parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('CONFIG', type=str)
  return parser.parse_args(argv)

def main(argv):
  # Args parser
  args = parse_args(argv)
  print(model) # XXX


if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)
