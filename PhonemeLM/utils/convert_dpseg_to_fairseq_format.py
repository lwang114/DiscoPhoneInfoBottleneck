import argparse
import os
from tqdm import tqdm
import json


def convert_dpseg_to_fairseq_format(in_path, file_id_paths, out_paths):
    """ 
    Args:
      in_path: str, segmentation file in dpseg format
      file_id_paths: list of str, paths containing file_ids for the splits
      out_paths: list of str, paths containing paths for the output files in fairseq format
    """
    assert len(file_id_paths) == len(out_paths)
    end_idxs = []
    lengths = []
    for file_id_path in file_id_paths:
        with open(file_id_path, 'r') as f_id:
            lens = [int(line.rstrip('\n').split()[1]) for line in f_id]
            lengths.extend(lens)
            end_idxs.append(len(lengths)-1)
            
    path_idx = 0
    f_out = open(out_paths[path_idx], 'w')
    with open(in_path, 'r') as f_in:
        for line_idx, line in tqdm(enumerate(f_in)):
            seq = line.rstrip('\n').split()
            length = lengths[line_idx]
            assert len(''.join(seq)) == length
            if len(line) and line_idx > end_idxs[path_idx]:
                path_idx += 1
                f_out.close()
                f_out = open(out_paths[path_idx], 'w')
            f_out.write(' '.join(seq)+'\n')
    f_out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path')
    parser.add_argument('--file_id_paths')
    parser.add_argument('--out_paths')
    args = parser.parse_args()
    file_id_paths = args.file_id_paths.split(',')
    out_paths = args.out_paths.split(',')
    convert_dpseg_to_fairseq_format(args.in_path, file_id_paths, out_paths)


if __name__ == '__main__':
    main()
    
            
            
