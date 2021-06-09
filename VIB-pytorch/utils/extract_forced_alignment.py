import os
import json
import argparse

def extract_forced_alignment(exp_dir):
  """
  Extract forced alignment from Kaldi based on this tutorial 
  https://www.eleanorchodroff.com/tutorial/kaldi/forced-alignment.html
  
  Args:
    exp_dir: str, path to the forced alignment directory, assuming the 
      following files exist
        data/lang/phone.txt : a file with each line 
          {phone in BIES fmt}\t{int phone id}
        data/lang/phone/align_lexicon.txt : a file with each line
          {word}\t{word}\t{phones in BIES fmt} 
  
  Returns: 
    utterance_info.json : file storing the forced alignment in the format
      { "utterance_id" : str,
        "words" : a list of dicts with keys 
          "begin" : float,
          "end" : float,
          "text" : str,
          "phonemes" : a list of dicts with keys
            "begin" : float,
            "end" : float,
            "text" : str
      } 
  """
  # Convert time marks and phone ids
  with open('data/lang/phones.txt', 'r') as phn_f,\
       open('data/lang/phones/silence.txt', 'r') as sil_f,\
       open(os.path.join(exp_dir, 'merged_alignment.txt'), 'r') as merged_f,\
       open(os.path.join(exp_dir, 'final_ali.txt'), 'w') as final_f:
    SIL = [line.split('_')[0] for line in sil_f]

    id_to_phone = dict()
    for line in phn_f:
      phn, idx = line.rstrip('\n').split()
      id_to_phone[idx] = phn

    cur_token = None  
    for line in merged_f:
      utt_id, channel, start, dur, phn_id = line.rstrip('\n').split()
      phn = id_to_phone[phn_id]
      final_f.write(f'{utt_id}\t{channel}\t{start}\t{dur}\t{phn}\n')

  # Convert phone alignment to word alignment
  with open('data/lang/phones/align_lexicon.txt', 'r') as pron_f,\
       open(os.path.join(exp_dir, 'final_ali.txt'), 'r') as final_f,\
       open(os.path.join(exp_dir, 'utterance_info.json'), 'w') as word_f:
    pron_to_word = dict()
    for line in pron_f:
      parts = line.split()
      pron = tuple(phn.split('_')[0] for phn in parts[2:])
      pron_to_word[pron] = parts[0]

    cur_utt_id = ''
    cur_utt = None
    cur_word = {'begin': None,
                'end': None,
                'phonemes': [],
               }
    start_word = 0
    for line in final_f:
      utt_id, _, begin, dur, phn = line.rstrip('\n').split('\t')
      if utt_id != cur_utt_id:
        print(utt_id)
        if cur_utt_id:
          word_f.write(json.dumps(cur_utt)+'\n')
        cur_utt_id = utt_id
        cur_utt = dict()
        cur_utt['utterance_id'] = utt_id
        cur_utt['words'] = []  

      parts = phn.split('_')
      token = parts[0]
      if token in SIL:
        continue
      boundary = parts[1]
      cur_word['phonemes'].append({'begin': float(begin),
                                   'end': float(begin)+float(dur),
                                   'text': token})

      if boundary in ['E', 'S']:
        pron = tuple(phn['text'] for phn in cur_word['phonemes'])
        cur_word['begin'] = cur_word['phonemes'][0]['begin']
        cur_word['end'] = cur_word['phonemes'][-1]['end']
        cur_word['text'] = pron_to_word[pron] 
        cur_utt['words'].append(cur_word)
        cur_word = {'begin': None,
                    'end': None,
                    'phonemes': []}
    word_f.write(json.dumps(cur_utt)+'\n')

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='exp/tri4b_train_clean_100')
  args = parser.parse_args()
  extract_forced_alignment(args.exp_dir)

if __name__ == '__main__':
  main()
