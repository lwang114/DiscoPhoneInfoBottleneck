import torch
from torch import nn
from torch.autograd import Variable
import json
import re
import argparse
import itertools
import sys

SIL = "SIL"
def str2bool(v):
    """
    codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
            #state_dict[key] = (1-self.decay)*state_dict[key] + (self.decay)*new_state_dict[key]

        self.model.load_state_dict(state_dict)

def convert_item_to_output_format(item_file, 
                                  output_file, 
                                  frame_rate=0.01,
                                  max_len=2048):
  """
  Args :
      item_file : str, path to file of format
          line 0 : whatever
          line > 0 : {audio id} {onset} {offset} {phn} {prev phn} {next phn} {spk}
          onset : beginning of the triplet (in s)
          offset : end of the triplet (in s)
          frame_rate : float, frame rate per feature frame (in s)
      output_file : str, path to output file of format
          {audio id} {phone idxs separated by commas}
  """
  tokens = set() 
  triplets = dict()
  audio_ids = []
  with open(item_file, 'r') as in_f,\
       open(output_file, 'r') as out_f:
    out_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")
    for idx, line in enumerate(in_f):
      if idx == 0:
        continue
      audio_id, begin, end, phn, _, _, _ = line.rstrip().split()
      if not phn in tokens:
        tokens.add(phn)
      
      if not audio_id in triplets:
        triplets[audio_id] = []
        audio_ids.append(audio_id)
      begin = int(float(begin) / frame_rate)
      end = int(float(end) / frame_rate)
      triplets[audio_id].append((begin, end, phn))

    tokens = ['BLANK']+sorted(tokens)
    stoi = {t:i for i, t in enumerate(tokens)}

    for audio_id in audio_ids:
      print(audio_id)
      sequence = ['0']*max_len
      for i, tri in enumerate(sorted(triplets[audio_id])):
        if i == len(triplets[audio_id]) - 1:
          if i > 1:
            begin = max(triplets[audio_id][i-2][1], tri[0])
          else:
            begin = tri[0]
        else:
          begin = max(triplets[audio_id][i+1][0], tri[0])

        if i == 0:
          if i < len(triplets[audio_id]) - 2: 
            end = min(triplets[audio_id][2][0], tri[1])
          else:
            end = tri[1]
        else:
          end = min(triplets[audio_id][i-1][1], tri[1])
        if max(begin, end) > max_len:
          continue
        
        for t in range(begin, end): 
          sequence[t] = str(stoi[tri[2]])
      sequence_str = ','.join(sequence)
      out_f.write(f'{audio_id} {sequence_str}\n')


def convert_json_to_item(json_path, output_path):
  """
  Args :
      json_path : str, .json file, each line storing a dict of
          {"utterance_id" : str,
           "words" : a list of list of word boundary info}
 
      output_path : str, path to .item file of format
          line 0 : whatever
          line > 0 : {audio id} {onset} {offset} {phn} {prev phn} {next phn} {spk}
          onset : beginning of the triplet (in s)
          offset : end of the triplet (in s)
          frame_rate : float, frame rate per feature frame (in s)
  """
  with open(json_path, 'r') as in_f,\
       open(output_path, 'w') as out_f:
    out_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")
    for line in in_f:
      label_dict = json.loads(line.rstrip('\n'))
      if 'utterance_id' in label_dict:
        utt_id = label_dict['utterance_id']
      else:
        utt_id = label_dict['audio_id']
      
      phonemes_with_stress = [phn for w in label_dict['words'] for phn in w['phonemes']]
      for phn_idx, phn in enumerate(phonemes_with_stress):
        token = re.sub(r'[0-9]', '', phn['text'])
        if phn_idx == 0:
          prev_token = SIL
        else:
          prev_token = re.sub(r'[0-9]', '', phonemes_with_stress[phn_idx-1]['text']) 
        
        if phn_idx == len(phonemes_with_stress) - 1:
          next_token = SIL
        else:
          next_token = re.sub(r'[0-9]', '', phonemes_with_stress[phn_idx+1]['text']) 

        begin = round(phn["begin"], 3)
        end = round(phn["end"], 3)
        out_f.write(f'{utt_id} {begin} {end} {token} {prev_token} {next_token} 0\n') 

def convert_item_to_beer_format(item_file, 
                                beer_file, 
                                frame_rate=0.01,
                                pred_file=None):
  used_utt_ids = None
  if pred_file:
      used_utt_ids = []
      with open(pred_file, 'r') as f:
          for line in f:
              if not line.strip().split()[0] in used_utt_ids:
                  used_utt_ids.append(line.strip().split()[0])
                  
  with open(item_file, 'r') as in_f,\
       open(beer_file, 'w') as out_f:
    cur_utt_str = ['']
    begin = 0
    for idx, line in enumerate(in_f):
      if idx == 0:
          continue
      tokens = line.split()
      
      if used_utt_ids is not None:
          if not tokens[0] in used_utt_ids:
              continue

      if tokens[0] != cur_utt_str[0]:
        if cur_utt_str[0]:
          out_f.write(' '.join(cur_utt_str)+'\n')
        cur_utt_str = [tokens[0]]
        begin = 0

      cur_begin = int(float(tokens[1]) / frame_rate)
      cur_end = int(float(tokens[2]) / frame_rate)
      for _ in range(begin, cur_begin):
        cur_utt_str.append(SIL)
      begin = cur_end

      dur = cur_end - cur_begin 
      cur_utt_str.extend([tokens[3]]*dur)
    out_f.write(' '.join(cur_utt_str))

def convert_output_to_beer_format(output_file, beer_file):
  with open(output_file, 'r') as in_f,\
       open(beer_file, 'w') as out_f:
    for line in in_f:
      tokens = line.strip().split()
      phns = ' '.join(tokens[1].split(','))
      out_f.write(f'{tokens[0]} {phns}\n')

if __name__ == '__main__':
  argv = sys.argv[1:]  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('TASK', type=int)
  args, _ = parser.parse_known_args(argv)

  if args.TASK == 0:
    convert_item_to_output_format('../../../../data/zerospeech2021-dataset/phonetic/dev-clean/dev-clean.item',
                                 'gold_quantized_outputs.txt',
                                 frame_rate=0.01,
                                 max_len=2048)
  elif args.TASK == 1:
    if len(argv) > 1:
      input_file = argv[1]
      output_file = argv[2]
    else:
      input_file = '../../../../data/zerospeech2021-dataset/phonetic/dev-clean/dev-clean.json'
      output_file = 'dev-clean_nonoverlap.item'
    convert_json_to_item(input_file, output_file)
  elif args.TASK == 2:
    if len(argv) > 1:
      item_file = argv[1]
      gold_beer_file = argv[2]
      pred_file = argv[3]
    else:
      item_file = '../../../../data/zerospeech2021-dataset/phonetic/dev-clean/dev-clean_nonoverlap.item'
      gold_beer_file = '../../../../data/zerospeech2021-dataset/phonetic/dev-clean/dev-clean.ali'
      pred_file = '../checkpoints/phone_discovery_librispeech_wav2vec2/predictions_phoneme.14.txt'
    convert_item_to_beer_format(item_file,
                                gold_beer_file,
                                frame_rate=0.01,
                                pred_file=pred_file)
  elif args.TASK == 3:
    if len(argv) > 1:
      output_file = argv[1]
      pred_beer_file = argv[2]
    else:
      output_file = '../checkpoints/phone_discovery_librispeech_wav2vec2/predictions_phoneme.14.txt'
      pred_beer_file = '../checkpoints/phone_discovery_librispeech_wav2vec2/pred_dev-clean.ali'
      # output_file = '../../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/outputs_quantized/quantized_outputs.txt' 
      # pred_beer_file = '../../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/outputs_quantized/pred_dev-clean.ali'
    convert_output_to_beer_format(output_file, pred_beer_file)
