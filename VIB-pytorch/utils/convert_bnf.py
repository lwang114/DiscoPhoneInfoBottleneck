import sys
import os
import json
import argparse
import numpy as np
from kaldiio import ReadHelper, WriteHelper

def extract_utt2wav(wav_scp):
  utt2wav = dict()
  with open(wav_scp, 'r') as f:
    for line in f:
      utt_id, fn = line.split()
      audio_id = os.path.basename(fn).split('.')[0]
      utt2wav[utt_id] = audio_id
  return utt2wav

def separate_bnf_by_ids(feat_scp, out_dir, wav_scp=None):
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  
  utt2wav = None
  if wav_scp:
    utt2wav = extract_utt2wav(wav_scp)

  with ReadHelper(f'scp:{feat_scp}') as reader:
    for key, arr in reader:
      if utt2wav:
        audio_id = utt2wav[key]
      else:
        audio_id = key
      print(key, audio_id)
      # out_path = os.path.join(out_dir, f'{audio_id}.ark.gz')
      np.savetxt(os.path.join(out_dir, f'{audio_id}.txt'), arr)
      # with WriteHelper(f'ark:| gzip -c > {out_path}') as writer:
      #   writer(str(0), arr)

def separate_bnf_by_words(feat_scp, word_info_file, out_dir, frame_rate=10, wav_scp=None):
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  utt2wav = None
  if wav_scp:
    utt2wav = extract_utt2wav(wav_scp)

  sent_to_word = dict()
  with open(word_info_file, 'r') as f:
    for line in f:
      word = json.loads(line.rstrip('\n')) 
      audio_id = word['audio_id']
      word_id = word['word_id']
      begin = word['begin']
      end = word['end']
      if not audio_id in sent_to_word:
        sent_to_word[audio_id] = dict()
      audio_word_id = f'{audio_id}_{word_id}'
      sent_to_word[audio_id][audio_word_id] = [begin, end]

  with ReadHelper(f'scp:{feat_scp}') as reader:
    for k, arr in reader:
      if utt2wav:
        audio_id = utt2wav[k]
      else:
        audio_id = k
      print(k, audio_id)
      if not audio_id in sent_to_word:
        continue
      for audio_word_id, segment in sent_to_word[audio_id].items():
        begin_frame = int(round(segment[0] * 1000 / frame_rate, 3))
        end_frame = int(round(segment[1] * 1000 / frame_rate, 3))
        word_arr = arr[begin_frame:end_frame]
        # out_path = os.path.join(out_dir, f'{audio_word_id}.ark.gz')
        np.savetxt(os.path.join(out_dir, f'{audio_word_id}.txt'), word_arr)
        # with WriteHelper(f'ark:| gzip -c > {out_path}') as writer:
        #   writer(str(0), word_arr)

          
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('TASK', type=int)

  args, argv = parser.parse_known_args(sys.argv[1:])

  if args.TASK == 0:
    if len(argv) > 1:
      separate_bnf_by_ids(*argv)
    else:
      feat_scp = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/liming_iclr2022/siyuan_multi13lang/to_liming/bnf_output/feats.scp' 
      wav_scp = None
      out_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/full_corpus_newsplit/all_bnf_txt/'
      separate_bnf_by_ids(feat_scp, out_dir, wav_scp=wav_scp)
  elif args.TASK == 1:
    if len(argv) > 1:
      separate_bnf_by_words(*argv)
    else:
      word_info_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/full_corpus_newsplit/mboshi_word/mboshi_word_top300-600.json' 
      feat_scp = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/liming_iclr2022/siyuan_multi13lang/to_liming/bnf_output/feats.scp' 
      wav_scp = None
      out_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/full_corpus_newsplit/mboshi_word_bnf_txt/'
      separate_bnf_by_words(feat_scp, word_info_file, out_dir, wav_scp=wav_scp)
