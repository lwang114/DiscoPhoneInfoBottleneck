import torch
from torch import nn
from torch.autograd import Variable
import json
import re

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
  

if __name__ == '__main__':
  # convert_item_to_output_format('../../../../data/zerospeech2021-dataset/phonetic/dev-clean/dev-clean.item',
  #                              'gold_quantized_outputs.txt',
  #                              frame_rate=0.01,
  #                              max_len=2048)
  convert_json_to_item('../../../../data/zerospeech2021-dataset/phonetic/dev-clean/dev-clean.json', 'dev-clean_nonoverlap.item')        
