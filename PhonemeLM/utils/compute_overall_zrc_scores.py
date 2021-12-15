import argparse


def compute_overall_zrc_scores(in_path, out_path, task):
  n_lines = 0
  scores = []
  with open(in_path, 'r') as f_in:
    for line in f_in:
      if n_lines == 0:
        n_lines += 1
        continue
      n_lines += 1
      scores.append(float(line.rstrip('\n').split(',')[-1]))

  info = f'Overall {task} score: {sum(scores)/len(scores):.4f}'
  print(info)
  with open(out_path, 'w') as f_out:
    f_out.write(info)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--in_path')
  parser.add_argument('--out_path')
  parser.add_argument('--task')
  args = parser.parse_args()
  compute_overall_zrc_scores(in_path=args.in_path,
                             out_path=args.out_path, 
                             task=args.task)


if __name__ == '__main__':
  main()
