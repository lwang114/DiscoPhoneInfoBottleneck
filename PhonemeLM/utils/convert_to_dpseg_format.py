import argparse
import os
import sys
from copy import deepcopy
from tqdm import tqdm
import json


symbol_list = [chr(ord('a')+i) for i in range(26)]+\
              [chr(ord('A')+i) for i in range(26)]+\
              [str(i) for i in range(10)]+\
              [chr(i) for i in range(593, 670)] # 139 distinct symbols in total 

def convert_to_dpseg_format(in_paths, out_prefix):
  """ Convert output file from clustering_quantization.py to input format of 
  adaptor grammar model from dpseg so that every unit gets mapped to a single char, 
   e.g., 5865-34629-0032 12,23,23 -> abc   
  """
  int2sym = dict()
  with open(out_prefix+'.txt', 'w') as f_out:
    for in_path in in_paths:
      with open(in_path, 'r') as f_in:
        for line in tqdm(f_in):
          seq = line.strip().split()
          seq_str = []
          for i in seq:
            if not i in int2sym:
              if int(i) < len(symbol_list): 
                int2sym[i] = symbol_list[int(i)]
              else:
                raise ValueError('Number of symbols exceeds maximum')
            seq_str.append(int2sym[i])
          f_out.write(''.join(seq_str)+'\n')

  with open(out_prefix+'.json', 'w') as f_dict:
    json.dump(int2sym, f_dict, indent=2)

def convert_to_dpseg_format_with_gold_segments(in_paths, segment_files, out_prefix):
  segment_dict = dict()
  int2sym = dict()
  out_text = []
  
  for in_path, segment_file in zip(in_paths, segment_files):
    with open(segment_file, 'r') as f_seg:
      for line in tqdm(f_seg):
        sent_dict = json.loads(line.rstrip('\n'))
        if 'audio_id' in sent_dict:
          audio_id = sent_dict['audio_id']
        else:
          audio_id = sent_dict['utterance_id']
        segment_dict[audio_id] = []
        phn_idx = -1
        for word in sent_dict['words'][:-1]:
          phn_idx += len(word['phonemes'])
          if phn_idx > 0:
            segment_dict[audio_id].append(phn_idx)

    with open(in_path.replace('.txt', '_ids.txt'), 'r') as f_id:
      audio_ids = [line.strip().split()[0] for line in tqdm(f_id)]

    with open(in_path, 'r') as f_in:
      for audio_id, line in tqdm(zip(audio_ids, f_in)):
        seq = line.strip().split()
        segments = segment_dict[audio_id]
          
        seq_str = []
        for end, i in enumerate(seq):
          if not i in int2sym:
            if int(i) < len(symbol_list): 
              int2sym[i] = symbol_list[int(i)]
            else:
              raise ValueError('Number of symbols exceeds maximum')
          seq_str.append(int2sym[i])
          if end and end in segments:
            seq_str.append(' ')
        out_text.append(''.join(seq_str)+'\n')
  
  with open(out_prefix+'.json', 'w') as f_dict,\
       open(out_prefix+'.txt', 'w') as f_out:
    json.dump(int2sym, f_dict, indent=2)
    f_out.write(''.join(out_text))

    
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--in_paths')
  parser.add_argument('--out_prefix')
  parser.add_argument('--segment_paths')
  args = parser.parse_args()

  in_paths = args.in_paths.split(',')
  if not args.segment_paths:
    convert_to_dpseg_format(in_paths, args.out_prefix)
  else:
    segment_paths = args.segment_paths.split(',')
    assert len(in_paths) == len(segment_paths)
    convert_to_dpseg_format_with_gold_segments(in_paths, segment_paths, args.out_prefix)

if __name__ == '__main__':
  main()
