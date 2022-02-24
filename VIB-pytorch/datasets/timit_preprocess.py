import json
import os
import argparse
from copy import deepcopy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.io import wavfile
import re
import sys
import shutil
from itertools import groupby 

stop_words = stopwords.words("english")
SILS = ['h#', 'pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'pau', 'epi']
SIL = 'SIL'
MAPPING = {'aa': 'aa',
           'ao': 'aa',
           'ah': 'ah', 
           'ax': 'ah', 
           'ax-h': 'ah',
           'er': 'er',
           'axr': 'er',
           'hh': 'hh', 
           'hv': 'hh',
           'ih': 'ih', 
           'ix': 'ih',
           'l': 'l', 
           'el': 'l',
           'm': 'm', 
           'em': 'm',
           'n': 'n',
           'en': 'n', 
           'nx': 'n',
           'ng': 'ng', 
           'eng': 'ng',
           'sh': 'sh', 
           'zh': 'sh',
           'uw': 'uw',
           'ux': 'uw',
           'q': SIL}

def is_overlapped(a, b):
  if (a['begin'] >= b['end']) or (a['end'] <= b['begin']):
    return False
  return True
  
def align_phones_with_words(phones, words):
  phones = sorted(phones, key=lambda x:x['begin'])
  for wrd_idx in range(len(words)):
    words[wrd_idx]['phonemes'] = []

  phn_idx = 0
  wrd_idx = 0
  while phn_idx < len(phones) and wrd_idx < len(words):
    phn = phones[phn_idx]
    wrd = words[wrd_idx]
    if phn['text'] in SILS:
      phn['text'] = SIL
      words[wrd_idx]['phonemes'].append(deepcopy(phn))
      phn_idx += 1
    elif is_overlapped(phn, wrd):
      words[wrd_idx]['phonemes'].append(deepcopy(phn))
      phn_idx += 1
    elif phones[phn_idx]['end'] <= words[wrd_idx]['begin']:
      phn_idx += 1
    else:
      wrd_idx += 1

  if not phn_idx == len(phones) or wrd_idx == len(words):
    print(words, phn_idx, wrd_idx, len(phones), len(words)-1)
  return words

def convert_dataset(data_path):
  drs = os.listdir(data_path)
  for dr in drs:
    if not os.path.isdir(os.path.join(data_path, dr)):
      continue
    for spk in os.listdir(os.path.join(data_path, dr)):
      if not os.path.isdir(os.path.join(data_path, dr, spk)):
        continue
      for fn in os.listdir(os.path.join(data_path, dr, spk)):
        if fn.split('.')[-1] == 'wav':
          print(fn, os.path.join(data_path, f'{dr}_{spk}_{fn}'))
          shutil.move(os.path.join(data_path, dr, spk, fn), os.path.join(data_path, f'{dr}_{spk}_{fn}')) 

def extract_meta_data(data_path,
                      out_prefix,
                      debug=False):
  """
  Args :
      data_path : str, path to the TIMIT root with the file structure
            {dialect name}/
                {speaker name}/
                    {audio_id}.WRD : word alignment file,
                    {audio_id}.PHD : phone alignment file
  """
  fns = os.listdir(data_path)
  word_dict = dict()
  phn_dict = dict()
  phn_alis = dict()
  num_files = 0
  for fn in fns:
    if fn.split('.')[-1] != 'wav':
      continue
    dr, spk, ftype = fn.split('.')[0].split('_')
    wrd_ali_file = os.path.join(data_path, dr, spk, ftype+'.WRD')
    phn_ali_file = os.path.join(data_path, dr, spk, ftype+'.PHN')
    
    num_files += 1
    if num_files > 20 and debug:
      break
    
    audio_id = fn.split('.')[0]
    word_dict[audio_id] = []
    with open(wrd_ali_file, 'r') as f:
      for line in f:
        begin_sample, end_sample, label = line.split() 
        begin = float(begin_sample) / 16000.
        end = float(end_sample) / 16000.
        word_dict[audio_id].append({'text': label,
                                    'begin': begin,
                                    'end': end})
    
    phn_dict[audio_id] = []
    with open(phn_ali_file, 'r') as f:
      for line in f:
        begin_sample, end_sample, label = line.split() 
        if label in SILS:
          label = SIL
        if label in MAPPING:
          label = MAPPING[label]
        begin = float(begin_sample) / 16000.
        end = float(end_sample) / 16000.
        phn_dict[audio_id].append({'text': label,
                                   'begin': begin,
                                   'end': end})
      
  # Create meta data and ABX file
  out_f = open(f'{out_prefix}.json', 'w')
  abx_f = open(f'{out_prefix}_nonoverlap.item', 'w') 
  abx_f.write('#file_ID onset offset #phone prev-phone next-phone speaker\n')
  for audio_id in word_dict:
    spk = audio_id.split('_')[-2]
    words = word_dict[audio_id]
    phones = phn_dict[audio_id]
    words = align_phones_with_words(phones, words)
    sent_dict = {'audio_id': audio_id,
                 'spk': spk,
                 'words': words,
                 'visual_words': []} 
    out_f.write(json.dumps(sent_dict)+'\n')
    
    for phn_idx, phn in enumerate(phones):
      if phn['text'] == SIL:
        continue
      prev_phn = SIL
      next_phn = SIL
      if phn_idx > 0:
        prev_phn = phones[phn_idx-1]['text']
      if phn_idx < len(phones) - 1:
        next_phn = phones[phn_idx+1]['text']
      abx_f.write(f'{audio_id} {phn["begin"]} {phn["end"]} {phn["text"]} {prev_phn} {next_phn} {spk}\n') 
    print(audio_id)
  out_f.close()
  abx_f.close()
  print(f'Number of audios: {num_files}')

def extract_vocab(data_path, out_file, word_type='noun', splits=['TRAIN']):
  lemmatizer = WordNetLemmatizer() 
  vocab = dict()
  word_dict = dict()
  for split in splits:
    sent_file = os.path.join(data_path, split, f'{split}.json')
    with open(sent_file, 'r') as f:
      for line in f:
        sent_dict = json.loads(line.rstrip('\n'))
        word_dict[sent_dict['audio_id']] = []
        if word_type.rfind('gram'):
            for n in word_type.rstrip('gram'):
                n = int(n)
                for w in sent_dict['words']:
                    phns = [(phn_idx, phn) for phn_idx, phn in enumerate(w['phonemes']) if phn['text'] != 'SIL']
                    for i in range(len(phns)-n+1):
                        ngram = ' '.join([phn['text'] for phn_idx, phn in phns[i:i+n]])
                        word_dict[sent_dict['audio_id']].append(
                            {'begin': w['phonemes'][phns[i][0]]['begin'],
                             'end': w['phonemes'][phns[i+n-1][0]]['end'],
                             'text': ngram,
                             'phonemes': deepcopy(w['phonemes'][phns[i][0]:phns[i+n-1][0]+1])}
                        )
                        if not ngram in vocab:
                            vocab[ngram] = 0
                        vocab[ngram] += 1 
        else:
            sent = [word['text'] for word in sent_dict['words']]
            postags = [token[1] for token in nltk.pos_tag(sent)]
            for w_idx, w in enumerate(sent):
              w = lemmatizer.lemmatize(w.lower())
              if word_type == 'noun' and postags[w_idx][0] == 'N':
                if not w in vocab:
                  vocab[w] = 0
                vocab[w] += 1
                word_dict[sent_dict['audio_id']].append(deepcopy(w))
              elif word_type == 'any':
                if not w in vocab:
                  vocab[w] = 0
                vocab[w] += 1
              else: raise ValueError(f'Unknown word type: {word_type}')
  print(f'Number of distinct vocab: {len(vocab)} stored in {out_file}')
  json.dump(vocab, open(out_file, 'w'), indent=2)
  return word_dict

def extract_word_dataset(data_path, out_path, vocab_file=None, dataset_name='train_timit', word_type='noun', k=300, debug=False):
  lemmatizer = WordNetLemmatizer()
  dataset_name = dataset_name
  out_path = os.path.join(out_path, dataset_name)
  splits = ["FULL"]
  if vocab_file:
    vocab = json.load(open(vocab_file))
    top_words = set()
    for w in sorted(vocab, key=lambda x:vocab[x], reverse=True):
      if w in vocab and not w in stop_words:
        if vocab[w] > 50:
          top_words.add(w)
    print(f'Number of chosen words: {len(top_words)}')
  else:
    vocab_path = os.path.join(data_path, f'TIMIT_{word_type}s.json')
    chosen_path = os.path.join(data_path, f'TIMIT_top{k}_{word_type}s.json')
    extract_vocab(data_path, vocab_path, word_type=word_type, splits=splits)
    vocab = json.load(open(vocab_path))
    top_words = sorted(vocab, key=lambda x:vocab[x], reverse=True)[:k]
    top_word_tuples = [(w, vocab[w]) for w in top_words]
    top_words = set(top_words)
    json.dump(top_word_tuples, open(chosen_path, 'w'), indent=2)
    print(f'Number of chosen words: {len(top_words)} stored in {chosen_path}')

  if not os.path.exists(out_path):
    os.makedirs(out_path)
  
  word_file = os.path.join(out_path, f'../{dataset_name}.json')
  abx_file = os.path.join(out_path, f'{dataset_name}_nonoverlap.item')
  counts = dict()
  word_f = open(word_file, 'w')
  abx_f = open(abx_file, 'w')
  abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

  global_idx = 0
  for split in splits:         
    sent_file = os.path.join(data_path, split, f'{split}.json') 
    sent_f = open(sent_file, 'r')
    for line in sent_f:
      sent_dict = json.loads(line.rstrip('\n'))
      audio_id = sent_dict['audio_id']
      audio_path = os.path.join(data_path, split, f'{audio_id}.wav')
      spk = sent_dict['spk']
      if not os.path.exists(audio_path):
        continue
      fs, audio = wavfile.read(audio_path)
      
      for word_idx, word in enumerate(sent_dict['words']):
        if debug and global_idx > 20:
          break
        if word['text'] in top_words:
          global_idx += 1
          if not word['text'] in counts:
            counts[word['text']] = 1
          elif counts[word['text']] >= 500:
            continue
          else:
            counts[word['text']] += 1
          
          word_info = {'audio_id': audio_id,
                       'word_id': str(word_idx),
                       'label': word['text'],
                       'begin': word['begin'],
                       'end': word['end'],
                       'spk': spk,
                       'split': dataset_name,
                       'phonemes': word['phonemes']}
          word_f.write(json.dumps(word_info)+'\n')

          # Extract wav file
          word_audio = audio[int(word['begin']*fs):int(word['end']*fs)]
          word_audio_id = f'{word_info["audio_id"]}_{word_info["word_id"]}'
          word_audio_path = os.path.join(out_path, word_audio_id+'.wav')
          wavfile.write(word_audio_path, fs, word_audio)
          
          # Extract ABX info
          for phn_idx, phn in enumerate(word['phonemes']):
            prev_phn = SIL
            next_phn = SIL
            if phn_idx > 0:
              prev_phn = word['phonemes'][phn_idx-1]['text']
            if phn_idx < len(word['phonemes']) - 1:
              next_phn = word['phonemes'][phn_idx+1]['text']
            abx_f.write(f'{word_audio_id} {phn["begin"]} {phn["end"]} {phn["text"]} {prev_phn} {next_phn} {spk}\n')         
    sent_f.close()
  print(f'Number of audio files: {global_idx+1}, number of chosen words used: {len(counts)}')
  word_f.close()
  abx_f.close()

def extract_ngram_dataset(data_path, out_path, vocab_file=None, dataset_name='train_timit', word_type='3gram', k=300, debug=False):
  lemmatizer = WordNetLemmatizer()
  dataset_name = dataset_name
  out_path = os.path.join(out_path, dataset_name)
  splits = ["FULL"]
  if vocab_file:
    vocab = json.load(open(vocab_file))
    top_words = set()
    for w in sorted(vocab, key=lambda x:vocab[x], reverse=True):
      if w in vocab and not w in stop_words:
        if vocab[w] > 50:
          top_words.add(w)
    print(f'Number of chosen words: {len(top_words)}')
  else:
    vocab_path = os.path.join(data_path, f'TIMIT_{word_type}s.json')
    chosen_path = os.path.join(data_path, f'TIMIT_top{k}_{word_type}s.json')
    ngram_dict = extract_vocab(data_path, vocab_path, word_type=word_type, splits=splits)
    vocab = json.load(open(vocab_path))
    top_words = sorted(vocab, key=lambda x:vocab[x], reverse=True)[:k]
    top_word_tuples = [(w, vocab[w]) for w in top_words]
    top_words = set(top_words)
    json.dump(top_word_tuples, open(chosen_path, 'w'), indent=2)
    print(f'Number of chosen words: {len(top_words)} stored in {chosen_path}')

  if not os.path.exists(out_path):
    os.makedirs(out_path)
  
  word_file = os.path.join(out_path, f'../{dataset_name}.json')
  abx_file = os.path.join(out_path, f'{dataset_name}_nonoverlap.item')
  counts = dict()
  word_f = open(word_file, 'w')
  abx_f = open(abx_file, 'w')
  abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

  global_idx = 0
  for split in splits:
    sent_file = os.path.join(data_path, split, f'{split}.json') 
    sent_f = open(sent_file, 'r')
    for line in sent_f:
      sent_dict = json.loads(line.rstrip('\n'))
      audio_id = sent_dict['audio_id']
      audio_path = os.path.join(data_path, split, f'{audio_id}.wav')
      spk = sent_dict['spk']
      if not os.path.exists(audio_path):
        continue
      fs, audio = wavfile.read(audio_path)
      
      # Extract ngrams in sentences
      for word_idx, word in enumerate(ngram_dict[audio_id]):
        if debug and global_idx > 20:
          break 
        if word['text'] in top_words:
          global_idx += 1
          if not word['text'] in counts:
            counts[word['text']] = 1
          elif counts[word['text']] >= 500:
            continue
          else:
            counts[word['text']] += 1
          
          word_info = {'audio_id': audio_id,
                       'word_id': str(word_idx),
                       'label': word['text'],
                       'begin': word['begin'],
                       'end': word['end'],
                       'spk': spk,
                       'split': dataset_name,
                       'phonemes': word['phonemes']}
          word_f.write(json.dumps(word_info)+'\n')

          # Extract wav file
          word_audio = audio[int(word['begin']*fs):int(word['end']*fs)]
          word_audio_id = f'{word_info["audio_id"]}_{word_info["word_id"]}'
          word_audio_path = os.path.join(out_path, word_audio_id+'.wav')
          wavfile.write(word_audio_path, fs, word_audio)
          
          # Extract ABX info
          for phn_idx, phn in enumerate(word['phonemes']):
            prev_phn = SIL
            next_phn = SIL
            if phn_idx > 0:
              prev_phn = word['phonemes'][phn_idx-1]['text']
            if phn_idx < len(word['phonemes']) - 1:
              next_phn = word['phonemes'][phn_idx+1]['text']
            abx_f.write(f'{word_audio_id} {phn["begin"]} {phn["end"]} {phn["text"]} {prev_phn} {next_phn} {spk}\n')         
    sent_f.close()
  print(f'Number of audio files: {global_idx+1}, number of chosen words used: {len(counts)}')
  word_f.close()
  abx_f.close()


def remove_SA_files(in_path, out_path):
  with open(in_path, 'r') as f_in, \
       open(out_path, 'w') as f_out:
    for line in f_in:
      if not 'SA' in line.split()[0].split('_')[-1]:
        print(line.rstrip('\n'), file=f_out)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('TASK', type=int)
  parser.add_argument('--debug', action='store_true')
  argv = sys.argv[1:]
  args, _ = parser.parse_known_args(argv)
  data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/TIMIT'
  if args.TASK == 0:
    extract_meta_data(f'{data_path}/TEST',
                      out_prefix=os.path.join(f'{data_path}/TEST', 'TEST'),
                      debug=False)
  if args.TASK == 1:
    extract_word_dataset(data_path,
                         out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                         vocab_file='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_visual_nouns.json', # '/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_nouns.json',
                         debug=False)
  if args.TASK == 2:
    convert_dataset(f'{data_path}/TEST')
  if args.TASK == 3:
    extract_word_dataset(data_path,
                         out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                         vocab_file=None,
                         dataset_name='train_timit_top300',
                         debug=False)
  if args.TASK == 4:
    if len(argv) > 1:
      in_path = argv[1]
      out_path = argv[2]
    else:
      in_path = f'{data_path}/TRAIN/TRAIN.ali'
      out_path = f'{data_path}/TRAIN/TRAIN_wo_sa.ali'
    remove_SA_files(in_path, out_path)
  if args.TASK == 5:
    extract_word_dataset(data_path,
                         out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                         vocab_file='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_nouns_top0-200.json',
                         dataset_name='train_timit_top0-200',
                         debug=args.debug)
  if args.TASK == 6:
    extract_word_dataset(data_path,
                         out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                         vocab_file='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_nouns_top200-600.json',
                         dataset_name='train_timit_top200-600',
                         debug=args.debug)
  if args.TASK == 7:
    extract_word_dataset(data_path,
                         out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                         vocab_file='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_visual_nouns.json',
                         dataset_name='train_timit_visual',
                         debug=args.debug)
  if args.TASK == 8:
    word_type = 'any'
    extract_word_dataset(data_path,
                         out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                         vocab_file=None,
                         dataset_name=f'train_timit_{word_type}_top300',
                         word_type=word_type,
                         debug=args.debug)
  if args.TASK == 9:
    word_type = '3gram'
    extract_ngram_dataset(data_path,
                          out_path='/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/librispeech_word',
                          vocab_file=None,
                          dataset_name=f'train_timit_{word_type}_top300',
                          word_type=word_type,
                          debug=args.debug)
