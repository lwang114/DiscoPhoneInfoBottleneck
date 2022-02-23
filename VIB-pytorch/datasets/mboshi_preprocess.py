import json
import os
from itertools import groupby
from scipy.io import wavfile
from copy import deepcopy
import re
import argparse
import numpy as np
import functools
from tqdm import tqdm

SIL = 'SIL'
TONES = [chr(i) for i in range(741, 748)]
SPECIAL_TOKENS = ['ʰ', 'ʔ'] 
TONE_MAPPING = {'ώ':'ω', 'á':'a', 'é':'e', 'ú':'u', 'í':'i', 'έ':'ε', 'ó':'o'}
def is_overlapped(a, b):
    return False if a[1] <= b[0] or a[0] >= b[1] else True
    
def align_phones_with_words(phones, words):
    """
    Args:
        phones: a list of dict of
            {'begin': float, begin sec,
             'end': float, end sec,
             'text': str}
        words: a list of dict of
            {'begin': float, begin sec,
             'end': float, end sec,
             'text': str}
    
    Returns:
        aligned_words: a list of dict of 
            {'begin': float, begin sec,
             'end': float, end_sec,
             'text': str,
             'phonemes': a list of dict of
                {'begin': float, begin sec,
                 'end': float, end sec,
                 'text': str}}
    """
    word_idx = 0
    phn_idx = 0
    n_words = len(words)
    n_phns = len(phones)
    for word in words:
        if not 'phonemes' in word:
            word['phonemes'] = []

    while (word_idx < n_words) and (phn_idx < n_phns):
        phone = phones[phn_idx]
        word = words[word_idx]
        
        phn_label = phone['text']
        wrd_label = word['text']
        if (phn_label == SIL):
            phn_idx += 1
            continue
        
        if (wrd_label == SIL):
            word_idx += 1
            continue
            
        phn_begin = phone['begin']
        phn_end = phone['end']
        word_begin = word['begin']
        word_end = word['end']
        
        if is_overlapped((phn_begin, phn_end), (word_begin, word_end)):
            word['phonemes'].append(deepcopy(phone))
            phn_idx += 1
        else:
            word_idx += 1
    return words


def extract_vocab(data_path):
    vocab = dict()
    for split in ['all']:
        sent_f = open(os.path.join(data_path, split, f'{split}.json'), 'r')
        for line in sent_f:
            sent_info = json.loads(line)
            for word in sent_info['words']:
                label = word['text']
                if label == SIL:
                    continue
                if not label in vocab:
                    vocab[label] = 1
                else:
                    vocab[label] += 1
    print(f'Vocab size: {len(vocab)}')
    return vocab


def extract_meta_data(word_ali_file, 
                      phone_ali_file, 
                      out_prefix,
                      debug=False):
    """
    Args:
        word_ali_file: str, a text file with each line containing
            {audio_id} {begin sec} {end sec} {word label}
        phone_ali_file: str, a text file with each line containing
            {audio_id} {begin sec} {end sec} {phone label}
    """
    word_f = open(word_ali_file, 'r')
    phn_f = open(phone_ali_file, 'r')
    out_f = open(out_prefix+'.json', 'w')
    abx_f = open(out_prefix+'_nonoverlap.item', 'w')
    abx_f.write('#file_ID onset offset #phone prev-phone next-phone speaker\n')
    
    word_dict = dict()
    lines = word_f.read().strip().split('\n')
    idx = 0
    for audio_id, sent_info in groupby(lines, lambda x:x.split()[0]):
        if debug and idx > 20:
            break
        idx += 1
        word_dict[audio_id] = []
        for word_info in sent_info:
            _, begin, end, label = word_info.split()
            begin = float(begin)
            end = float(end)
            word_dict[audio_id].append({'begin': begin,
                                        'end': end,
                                        'text': label})
    word_f.close()
    
    phn_dict = dict()
    lines = phn_f.read().strip().split('\n')
    idx = 0
    for audio_id, sent_info in groupby(lines, lambda x:x.split()[0]):
        if debug and idx > 20:
            break
        idx += 1
        phn_dict[audio_id] = []
        for phn_info in sent_info:
            _, begin, end, label = phn_info.split()
            begin = float(begin)
            end = float(end)
            phn_dict[audio_id].append({'begin': begin,
                                       'end': end,
                                       'text': label})
    phn_f.close()
    
    # Align words with phones
    for audio_id in sorted(word_dict):
        spk = audio_id.split('_')[0] 
        words = word_dict[audio_id]
        phns = phn_dict[audio_id]
        aligned_words = align_phones_with_words(phns, words)
        for w in aligned_words:
          if len(w['phonemes']) == 0:
            print(audio_id, w, ' does not match any phonemes')

        align_info = {'audio_id': audio_id,
                      'spk': spk,
                      'words': aligned_words,
                      'visual_words': []}
        
        print(audio_id) # XXX
        for word in aligned_words:
          for phn_idx, phn in enumerate(word['phonemes']):
            if phn['text'] == SIL:
              continue
            prev_phn = SIL
            next_phn = SIL
            if phn_idx > 0:
              prev_phn = word['phonemes'][phn_idx-1]['text']
            if phn_idx < len(word['phonemes']) - 1:
              next_phn = word['phonemes'][phn_idx+1]['text']
            abx_f.write(f'{audio_id} {phn["begin"]} {phn["end"]} {phn["text"]} {prev_phn} {next_phn} {spk}\n')
        out_f.write(json.dumps(align_info)+'\n')
    abx_f.close()
    out_f.close()


def split_metadata(in_path, wavs, out_path):
  with open(in_path, 'r') as f_in,\
       open(out_path, 'w') as f_out:
    for line in tqdm(f_in):
      sent_dict = json.loads(line.rstrip('\n'))
      if sent_dict['audio_id']+'.wav' in wavs:
        f_out.write(json.dumps(sent_dict)+'\n')


def extract_word_dataset(data_path, order=[0, 100], debug=False):
    """
    Args:
        data_path: str, directory containing following files:
        
            {split}.json: file with each line containing
                {"audio_id": audio_id,
                 "word_id": str,
                 "label": str,
                 "begin": float, begin_sec,
                 "end": float, end_sec,
                 "spk": str,
                 "split": str,
                 "phonemes": a list of dicts}
            
            {split}.item: file with each line continaing
                #file_ID onset offset #phone prev-phone next-phone speaker
        
        top_k: int
    """
    dataset_name = f'mboshi_word'
    dataset_path = os.path.join(data_path, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Extract top vocab
    vocab = extract_vocab(data_path)
    top_words = sorted(vocab, key=lambda x:vocab[x], reverse=True)[order[0]:order[1]]
    with open(os.path.join(data_path, f'mboshi_vocab_top{order[0]}-{order[1]}.json'), 'w') as f:
      json.dump(top_words, f, indent=2)
    
    word_f = open(os.path.join(dataset_path, f'{dataset_name}_top{order[0]}-{order[1]}.json'), 'w')
    counts = dict()
    word_lens = dict()
    total_dur = 0
    for split in ['all']:
        sent_f = open(os.path.join(data_path, split, f'{split}.json'), 'r')
        new_split = f'{split}_top{order[0]}-{order[1]}'
        split_path = os.path.join(dataset_path, new_split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        abx_f = open(os.path.join(split_path, f'{new_split}_nonoverlap.item'), 'w')
        abx_f.write('#file_ID onset offset #phone prev-phone next-phone speaker\n')
        
        idx = 0
        for line in sent_f:
            if debug and idx > 20:
                break
            idx += 1
            sent_info = json.loads(line.rstrip('\n'))
            audio_id = sent_info['audio_id']
            spk = sent_info['spk']
            # print(audio_id) # XXX
            
            for word_id, word_info in enumerate(sent_info['words']):
                if not word_info['text'] in counts:
                  counts[word_info['text']] = 1
                  if word_info['text'] in top_words: 
                    word_lens[word_info['text']] = len(word_info['phonemes'])
                else:
                  counts[word_info['text']] += 1

                if word_info['text'] in top_words and (counts[word_info['text']] < 200):
                    # Create .wav file
                    audio_path = os.path.join(data_path, split, f'{audio_id}.wav')
                    if not os.path.exists(audio_path):
                      continue
                    begin_sec = word_info['phonemes'][0]['begin']
                    end_sec = word_info['phonemes'][-1]['end'] 

                    fs, audio = wavfile.read(audio_path)
                    begin = int(begin_sec * 16000) 
                    end = int(end_sec * 16000)
                    total_dur += end_sec - begin_sec


                    word_audio = audio[begin:end]
                    word_audio_path = os.path.join(split_path, f'{audio_id}_word{word_id}.wav')
                    wavfile.write(word_audio_path, fs, word_audio)

                    # Update meta data file
                    word_info['audio_id'] = audio_id
                    word_info['word_id'] = f'word{word_id}'
                    word_info['label'] = word_info['text']
                    word_info['spk'] = spk
                    word_info['split'] = new_split
                    word_f.write(json.dumps(word_info)+'\n')
                                
                    # Update abx item file
                    for phn_idx, phn in enumerate(word_info["phonemes"]):
                        prev_phn = SIL
                        next_phn = SIL
                        if phn_idx > 0:
                            prev_phn = word_info["phonemes"][phn_idx-1]["text"]
                            #prev_phn = re.sub(r"[0-9]", "", prev_phn)
                        if phn_idx < len(word_info["phonemes"]) - 1:
                            next_phn = word_info["phonemes"][phn_idx+1]["text"]
                            #next_phn = re.sub(r"[0-9]", "", next_phn)
                        phn_label = phn["text"] #re.sub(r"[0-9]", "", phn["text"])
          
                        begin_phn = round(phn["begin"] - begin_sec, 3) 
                        end_phn = round(phn["end"] - begin_sec, 3)
                        abx_f.write(f"{audio_id}_{word_id} {begin_phn} {end_phn} {phn_label} {prev_phn} {next_phn} {spk}\n")
                    
        sent_f.close()
        abx_f.close()
    word_f.close()
    avg_word_len = sum(word_lens.values()) / len(word_lens)
    print(f'Total duration {total_dur / 3600:.4f} hrs, average word length {avg_word_len:.3f}')


def count_ngrams(phone_align_file, n_min=2, n_max=4, debug=False):
    sentences = dict()
    with open(phone_align_file, 'r') as f:
        # Read sentences
        for line in f:
            if debug and len(sentences) > 20:
                break
            audio_id, begin, end, phn = line.split()
            begin = float(begin)
            end = float(end)
            if not audio_id in sentences:
                sentences[audio_id] = []
            sentences[audio_id].append({'begin': begin,
                                        'end': end,
                                        'text': phn})
    
    # Compute n-grams
    ngram_counts = dict()
    for audio_id in sentences:
        sent = sentences[audio_id]
        for n in range(n_min, n_max+1):
            for i in range(len(sent)-n+1):
                ngram = ' '.join([phn['text'] for phn in sent[i:i+n]])
                if not ngram in ngram_counts:
                    ngram_counts[ngram] = 0
                ngram_counts[ngram] += 1
    
    # Filter n-grams that occur at least 200 times
    top_ngrams = dict()
    for ngram in ngram_counts:
        if ngram_counts[ngram] > 100:
            top_ngrams[ngram] = ngram_counts[ngram]
    
    print(f'Number of ngrams with frequency >= 100: {len(top_ngrams)}')
    ngram_num = sum(list(top_ngrams.values()))
    print(f'Total number of ngrams: {ngram_num}')
    
    # Extract ngrams in sentences
    ngram_dict = dict()
    for audio_id in sentences:
        sent = sentences[audio_id]
        for n in range(n_min, n_max+1):
            for i in range(len(sent)-n+1):
                ngram_text = ' '.join([phn['text'] for phn in sent[i:i+n]])
                if ngram_text in top_ngrams:
                    if not audio_id in ngram_dict:
                        ngram_dict[audio_id] = []
                    
                    ngram = {'begin': sent[i]['begin'],
                             'end': sent[i+n-1]['end'],
                             'text': ngram_text,
                             'phonemes': deepcopy(sent[i:i+n])} 
                    ngram_dict[audio_id].append(ngram)
    return ngram_dict, top_ngrams
    
    
def extract_ngram_dataset(phone_ali_file, data_path, n_min=3, n_max=10, debug=False):
    dataset_name = 'mboshi_word'
    new_split = f'all_{n_min}-{n_max}grams'
    split_path = os.path.join(data_path, dataset_name, new_split)
    if not os.path.isdir(split_path):
        os.makedirs(split_path)
    
    ngram_dict, top_ngrams = count_ngrams(phone_ali_file, n_min=n_min, n_max=n_max)
    
    word_f = open(os.path.join(data_path, dataset_name, f'{dataset_name}_{n_min}-{n_max}gram.json'), 'w') 
    # XXX abx_f = open(os.path.join(data_path, dataset_name, new_split, f'{new_split}_nonoverlap.item'), 'w')
    # XXX abx_f.write('#file_ID onset offset #phone prev-phone next-phone speaker\n')
    for i, audio_id in enumerate(sorted(ngram_dict)):
        print(audio_id)
        if debug and i > 20:
          break

        # Save the audio segments, save the metadata, save the item file
        audio_path = os.path.join(data_path, 'all', f'{audio_id}.wav')
        # XXX fs, audio = wavfile.read(audio_path)
        for word_id, ngram in enumerate(ngram_dict[audio_id]):
            begin_sec = ngram['begin']
            end_sec = ngram['end']
            begin = int(begin_sec * 16000) 
            end = int(end_sec * 16000)
            # XXX word_audio = audio[begin:end]
            word_audio_path = os.path.join(split_path, f'{audio_id}_{n_min}-{n_max}gram{word_id}.wav')
            # XXX wavfile.write(word_audio_path, fs, word_audio)
            
            # Update word info
            spk = audio_id.split('_')[0]
            word_info = {'audio_id': audio_id,
                         'word_id': f'{n_min}-{n_max}gram{word_id}',
                         'begin': ngram['begin'],
                         'end': ngram['end'],
                         'label': ngram['text'],
                         'spk': spk,
                         'split': new_split,
                         'phonemes': deepcopy(ngram['phonemes'])}
            word_f.write(json.dumps(word_info)+'\n')
            
            # Update abx file
            for phn_idx, phn in enumerate(word_info["phonemes"]):
                prev_phn = SIL
                next_phn = SIL
                if phn_idx > 0:
                    prev_phn = word_info["phonemes"][phn_idx-1]["text"]
                if phn_idx < len(word_info["phonemes"]) - 1:
                    next_phn = word_info["phonemes"][phn_idx+1]["text"]
                 
                phn_label = phn["text"] 
          
                begin_phn = round(phn["begin"] - begin_sec, 3) 
                end_phn = round(phn["end"] - begin_sec, 3)
                # XXX abx_f.write(f"{audio_id}_{word_id} {begin_phn} {end_phn} {phn_label} {prev_phn} {next_phn} {spk}\n")
    word_f.close()
    # XXX abx_f.close()
    
            
def extract_kaldi_multilingual_phones(data_path):
  """
  Args :
    data_path : str, path to the Flickr8k word root
    split : str,     pseudo_phone_file : str, storing a dict of
      {"utts" : 
          {audio_id} : 
              "output" : [{"rec_text" : str,
                           "rec_token" : str}],
              "utt2spk" : str
          }
      }
  """ 
  def _remove_special_tokens(phn):
    for t in TONES+SPECIAL_TOKENS:
      phn = re.sub(t, '', phn)
    return phn

  in_file = os.path.join(data_path, f"all/all.json")
  out_file = os.path.join(data_path, f"all/all_with_multilingual_phones.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  tokens = set()
  multilingual_phones = dict()
  for split in ["Mboshi_train_decoded", "Mboshi_dev_decoded", "Mboshi_eval_decoded"]:
    multilingual_phone_file = f"/ws/ifp-53_1/hasegawa/tools/kaldi/egs/discophone/v1_multilang/exp/gmm/tri5_{split}/merged_alignment_sym_new.txt"
    with open(multilingual_phone_file, "r") as f:
      phone_list = [line.rstrip('\n').split() for line in f]

    for utt_id, group in groupby(phone_list, lambda x:x[0]):
      for phn_info in group:
        _, spk, begin, end, phn = phn_info
        if not utt_id in multilingual_phones:
          multilingual_phones[utt_id] = []

        phn = _remove_special_tokens(phn)
        if not phn in tokens:
          tokens.add(phn)
        multilingual_phones[utt_id].append({"begin": float(begin),
                                            "end": float(begin)+float(end),
                                            "text": phn})

  for line in in_f:
    sent_dict = json.loads(line.rstrip("\n"))
    audio_id = sent_dict["audio_id"].split("/")[-1]
    sent_dict["multilingual_phones"] = []

    if not audio_id in multilingual_phones:
      print(f"{audio_id} not found")
      continue 
    sent_dict["multilingual_phones"].extend(multilingual_phones[audio_id])
    out_f.write(json.dumps(sent_dict)+"\n")
  in_f.close()
  out_f.close()
  print(tokens)
  print(f"Multilingual phone set size: {len(tokens)}") 


def extract_kaldi_multilingual_phones_for_words(data_path):
  """
  Args :
    data_path : str, path to the Flickr8k word root
    split : str,     pseudo_phone_file : str, storing a dict of
      {"utts" : 
          {audio_id} : 
              "output" : [{"rec_text" : str,
                           "rec_token" : str}],
              "utt2spk" : str
          }
      }
  """ 
  def _remove_special_tokens(phn):
    for t in TONES+SPECIAL_TOKENS:
      phn = re.sub(t, '', phn)
    return phn

  in_file = os.path.join(data_path, f"mboshi_word/mboshi_word.json")
  out_file = os.path.join(data_path, f"mboshi_word/mboshi_word_with_multilingual_phones.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  tokens = set()
  multilingual_phones = dict()
  for split in ["mboshi_word_all_decoded"]:
    multilingual_phone_file = f"/ws/ifp-53_1/hasegawa/tools/kaldi/egs/discophone/v1_multilang/exp/gmm/tri5_{split}/merged_alignment_sym.txt"
    with open(multilingual_phone_file, "r") as f:
      phone_list = [line.rstrip('\n').split() for line in f]

    for utt_id, group in groupby(phone_list, lambda x:x[0]):
      for phn_info in group:
        _, spk, begin, end, phn = phn_info
        if not utt_id in multilingual_phones:
          multilingual_phones[utt_id] = []

        phn = _remove_special_tokens(phn)
        if not phn in tokens:
          tokens.add(phn)
        multilingual_phones[utt_id].append({"begin": float(begin),
                                            "end": float(begin)+float(end),
                                            "text": phn})

    for line in in_f:
      sent_dict = json.loads(line.rstrip("\n"))
      audio_id = sent_dict["audio_id"].split("/")[-1]
      spk = sent_dict["spk"]
      word_idx = int(sent_dict["word_id"])
      sent_dict["multilingual_phones"] = []

      utt_id = f"{audio_id}_{word_idx}"
      if not utt_id in multilingual_phones:
        print(f"{utt_id} not found")
        continue
      print(utt_id)  
      sent_dict["multilingual_phones"].extend(multilingual_phones[utt_id])
      out_f.write(json.dumps(sent_dict)+"\n")
  in_f.close()
  out_f.close()
  print(tokens)
  print(f"Multilingual phone set size: {len(tokens)}")
 

def extract_kaldi_multilingual_phones_for_words_wo_alignment(data_path):
  """
  Args :
    data_path : str, path to the Flickr8k word root
    split : str,     pseudo_phone_file : str, storing a dict of
      {"utts" : 
          {audio_id} : 
              "output" : [{"rec_text" : str,
                           "rec_token" : str}],
              "utt2spk" : str
          }
      }
  """
  def _remove_special_tokens(phn):
    for t in TONES+SPECIAL_TOKENS:
      phn = re.sub(t, '', phn)
    return phn

  in_file = os.path.join(data_path, f"mboshi_word/mboshi_word.json")
  out_file = os.path.join(data_path, f"mboshi_word/mboshi_word_with_multilingual_phones.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  tokens = set()
  multilingual_phones = dict()
  multilingual_phone_file = f"/ws/ifp-53_1/hasegawa/tools/kaldi/egs/discophone/v1_multilang/exp/gmm/tri5/merged_transcript.txt"
  with open(multilingual_phone_file, "r") as f:
    phone_list = [line.rstrip('\n').split() for line in f]

  for utt_id, group in groupby(phone_list, lambda x:x[0]):
    if utt_id in multilingual_phones:
      if len(multilingual_phones[utt_id]) > 0:
        continue # Only use the first best path
      print(utt_id)
    
    if not utt_id in multilingual_phones:
      multilingual_phones[utt_id] = []

    for phn_info in group:
      for phn in phn_info[1:]:
        if phn[0] == '<':
          continue
        phn = _remove_special_tokens(phn)
        if not phn:
          continue
        for token in phn:
          if token == 'ː':
            multilingual_phones[utt_id][-1]['text'] = multilingual_phones[utt_id][-1]['text']+token
            continue
          if not token in tokens:
            tokens.add(token) 
          multilingual_phones[utt_id].append({"begin": -1,
                                              "end": -1,
                                              "text": token})
  i = 0
  for line in in_f:
    sent_dict = json.loads(line.rstrip("\n"))
    audio_id = sent_dict["audio_id"].split("/")[-1]
    spk = sent_dict["spk"]
    word_idx = int(sent_dict["word_id"])
    sent_dict["multilingual_phones"] = []

    utt_id = f"{audio_id}_{word_idx:04d}"
    if not utt_id in multilingual_phones:
      print(f"{utt_id} not found")
    else:
      print(i, utt_id)  
      i += 1
      sent_dict["multilingual_phones"].extend(multilingual_phones[utt_id])
    out_f.write(json.dumps(sent_dict)+"\n")
  in_f.close()
  out_f.close()
  print(tokens)
  print(f"Multilingual phone set size: {len(tokens)}") 


def extract_multilingual_phones(in_path, multilang_path, out_path, key_name='predicted_segments_multilingual'):
  """
  Args :
    in_path : str, path to the Mboshi dataset metadata 
    multilang_path : str, path to the multilingual labels in beer format
  """
  sentences = dict()
  with open(multilang_path, 'r') as f:
      for line in f:
          parts = line.split()
          audio_id = parts[0]
          print(audio_id)
          phns = []
          phn_text = parts[1:]
          begin = 0
          for phn_idx in range(1, len(phn_text)):
              if phn_text[phn_idx] != phn_text[phn_idx-1]:
                  cur_phn = phn_text[phn_idx-1]
                  phns.append(Phone(cur_phn, begin, phn_idx-1))
                  begin = phn_idx
          cur_phn = phn_text[-1]
          phns.append(Phone(cur_phn, begin, phn_idx))
          
          phn_text_re = []
          for phn in phns:
              phn_text_re.extend([phn.text]*(phn.end-phn.begin+1))

          is_eq = functools.reduce(lambda x,y:x and y, map(lambda x,y:x==y, phn_text, phn_text_re))
          if not is_eq:
              print(f'Original:\n{phn_text}')
              print(f'Reconstructed:\n{phn_text_re}') # XXX
              print('\n')
          sentences[audio_id] = Sentence([Word(phns, 0)])

  with open(in_path, 'r') as in_f, \
       open(out_path, 'w') as out_f:
      for line in in_f:
          sent_dict = json.loads(line.rstrip('\n'))
          audio_id = sent_dict['audio_id']
          print(audio_id)
          sent_dict[key_name] = [{'text': phn.text,
                                  'begin': phn.begin_sec,
                                  'end': phn.end_sec} for phn in sentences[audio_id].phones]
          out_f.write(json.dumps(sent_dict)+'\n')


def extract_multilingual_phones_for_words(in_path, multilang_path, out_path, key_name='predicted_segments_multilingual'):
  """
  Args :
    in_path : str, path to the Mboshi word dataset metadata 
    multilang_path : str, path to the multilingual labels in beer format
  """
  sentences = dict()
  with open(multilang_path, 'r') as f:
      for line in f:
          parts = line.split()
          audio_id = parts[0]
          print(audio_id)
          phns = []
          phn_text = parts[1:]
          begin = 0
          for phn_idx in range(1, len(phn_text)):
              if phn_text[phn_idx] != phn_text[phn_idx-1]:
                  cur_phn = phn_text[phn_idx-1]
                  phns.append(Phone(cur_phn, begin, phn_idx-1))
                  begin = phn_idx
          cur_phn = phn_text[-1]
          phns.append(Phone(cur_phn, begin, phn_idx))
          
          phn_text_re = []
          for phn in phns:
              phn_text_re.extend([phn.text]*(phn.end-phn.begin+1))

          is_eq = functools.reduce(lambda x,y:x and y, map(lambda x,y:x==y, phn_text, phn_text_re))
          if not is_eq:
              print(f'Original:\n{phn_text}')
              print(f'Reconstructed:\n{phn_text_re}') # XXX
              print('\n')

          sentences[audio_id] = Sentence([Word(phns, 0)])

  with open(in_path, 'r') as in_f, \
       open(out_path, 'w') as out_f:
      for line in in_f:
          word_dict = json.loads(line.rstrip('\n'))
          audio_id = word_dict['audio_id']
          print(audio_id)
          begin = word_dict['begin']
          end = word_dict['end']
          word_dict[key_name] = [{'text': phn.text,
                                  'begin': phn.begin_sec,
                                  'end': phn.end_sec} for phn in sentences[audio_id].find_phones(begin, end)]
          out_f.write(json.dumps(word_dict)+'\n')


class Phone:
    def __init__(self, text, begin=0, end=0, frame_sec=0.01):
        self.text = text
        self.begin = begin
        self.end = end
        self.frame_sec = 0.01
    
    @property
    def begin_sec(self):
        begin_sec = round(self.begin * self.frame_sec, 3)
        begin_re = int(round(begin_sec / self.frame_sec, 3))
        if begin_re != self.begin:
            print(f'{begin_re} (reconstructed) != {self.begin} (orig)')
        return begin_sec
    
    @property
    def end_sec(self):
        end_sec = round((self.end + 1) * self.frame_sec, 3)
        end_re = int(round(end_sec / self.frame_sec, 3)) - 1 
        if end_re != self.end:
            print(f'{end_re} (reconstructed) != {self.end} (orig)') 
        return end_sec
                
class Word:
    def __init__(self, phones, begin):
        self.phones = phones
        self.begin = begin

    @property
    def end(self):
        return self.begin + len(self.phones) - 1
        
    @property
    def text(self):
        return [phn.text for phn in self.phones]

    @property
    def begin_sec(self):
        return self.phones[0].begin_sec
    
    @property
    def end_sec(self):
        return self.phones[-1].end_sec
    
    def add(self, phone):
        self.phones.append(phone)
        self.phones = sorted(self.phones, key=lambda x:x.end)
      
    def __len__(self):
        return len(self.phones)
    
    def contains(self, phone):
        for phn in self.phones:
            if phn.begin == phone.begin and phn.end == phone.end:
                return True
        return False
    
class Sentence:
    def __init__(self, words):
        self.words = words

    @property
    def text(self):
        return [phn for w in self.words for phn in w.text]
    
    @property
    def phones(self):
        return [phn for word in self.words for phn in word.phones]
    
    def add(self, word):
        self.words.append(word)
    
    def add_phone(self, i, phone):
        self.words[i].add(phone)
    
    def __len__(self):
        return len(self.text)
    
    def contains_phone(self, phn):
        for word in self.words:
            if word.contains(phn):
                return True
        return False
    
    def find_phones(self, begin_sec, end_sec):
        phns = []
        for phn in self.phones:
          if phn.end_sec <= begin_sec or phn.begin_sec >= end_sec:
            continue

          begin = int(round(begin_sec / phn.frame_sec, 3))
          end = int(round(end_sec / phn.frame_sec, 3))
          cur_begin = max(phn.begin, begin)
          cur_end = min(phn.end, end)
          phns.append(Phone(phn.text, cur_begin, cur_end))
        return phns

class Corpus:
    def __init__(self, 
                 wavscp_file,
                 text_file,
                 beer_file,
                 phone_itos_file,
                 phoneme_itos_file,
                 phone2phoneme_file):
        self.phone_itos, \
        self.phoneme_itos, \
        self.phone2phoneme = self.create_phone_mapping(phone_itos_file,
                                                       phoneme_itos_file,
                                                       phone2phoneme_file)
        self.phoneme_stoi = {phn:i for i, phn in self.phoneme_itos.items()}
        self.utt2wav = self.create_utt_to_wav_mapping(wavscp_file)
        self.orig_sents = self.read_original_sentences(text_file)
        self.beer_sents = self.read_beer_sentences(beer_file)
        
    def parse(self, word):
        max_phn_len = 3
        phn_list = [] 
        begin = 0
        while begin < len(word):
          for i in range(max_phn_len):
            end = begin + max_phn_len - i - 1
            phn = word[begin:end+1]
            found = 0
            if not phn in self.phone2phoneme:
              if begin == end:
                print(f'{phn} at ({begin}, {end}, {len(word)}) not found') 
                phn_list.append(Phone(phn))
                begin += max_phn_len - i
              else:
                continue
            else:
              phn_list.append(Phone(phn))
              begin += max_phn_len - i
              break
        return phn_list

    def create_utt_to_wav_mapping(self, wavscp_file):
        utt2wav = dict()
        with open(wavscp_file, 'r') as f:
            for line in f:
                utt_id, wav = line.rstrip('\n').split()
                audio_id = os.path.basename(wav).split('.')[0]
                utt2wav[utt_id] = audio_id
        return utt2wav
    
    def read_original_sentences(self, text_file):
        sentences = dict()
        tokens = set()
        with open(text_file, 'r') as text_f:
            for line in text_f:
                parts = line.strip().split()
                audio_id = self.utt2wav[parts[0]]
                words = []
                begin = 0
                for word in parts[1:]:
                    phone_list = self.parse(word)
                    words.append(Word(phone_list, begin))
                    begin += len(phone_list)
                    tokens.update([phn.text for phn in phone_list]) 
                sentences[audio_id] = Sentence(words)
        print('Original phoneme set: ', tokens)
        return sentences

    def read_beer_sentences(self, beer_file):
        sentences = dict()
        with open(beer_file, 'r') as beer_f:
            for line in beer_f:
                parts = line.split()
                audio_id = parts[0]
                phns = []
                phn_text = parts[1:]
                begin = 0
                for phn_idx in range(1, len(phn_text)):
                    if phn_text[phn_idx] != phn_text[phn_idx-1]:
                        cur_phn = self.phoneme_itos[phn_text[phn_idx-1].lstrip('phoneme')]
                        phns.append(Phone(cur_phn, begin, phn_idx-1))
                        begin = phn_idx
                cur_phn = self.phoneme_itos[phn_text[-1].lstrip('phoneme')]
                phns.append(Phone(cur_phn, begin, phn_idx))
                
                phn_text_re = []
                for phn in phns:
                    phn_text_re.extend([f'phoneme{self.phoneme_stoi[phn.text]}' if phn.text != SIL.lower() else phn.text]*(phn.end-phn.begin+1))
 
                is_eq = functools.reduce(lambda x,y:x and y, map(lambda x,y:x==y, phn_text, phn_text_re))
                if not is_eq:
                    print(f'Original:\n{phn_text}')
                    print(f'Reconstructed:\n{phn_text_re}') # XXX
                    print('\n')

                sentences[audio_id] = Sentence([Word(phns, 0)])

        return sentences
        
    def create_phone_mapping(self, 
                             phone_itos_file, 
                             phoneme_itos_file,
                             phone2phoneme_file):
        phone_itos = json.load(open(phone_itos_file))
        phoneme_itos = json.load(open(phoneme_itos_file))
        phone2phoneme = dict()
        with open(phone2phoneme_file, 'r') as f:
            for line in f:
                phn_idx, phm_idx = line.rstrip('\n').split()
                phn = phone_itos[phn_idx]
                phm = phoneme_itos[phm_idx]
                phone2phoneme[phn] = phm        
        return phone_itos, phoneme_itos, phone2phoneme
    
    def remove_repeated(self, sent):
        new_sent = Sentence([])
        begin = 0
        for w_idx, word in enumerate(sent.words):
            new_word = Word([], begin)
            for phn_idx, phn in enumerate(word.phones):
                if not phn_idx and not w_idx:
                    new_word.add(deepcopy(phn))
                    continue
                elif not phn_idx:
                    prev_phn = sent.words[w_idx].phones[-1]
                else:
                    prev_phn = word.phones[phn_idx-1]

                if prev_phn.text != phn.text:
                    new_word.add(deepcopy(phn))
            if len(new_word) > 0:
              new_sent.add(new_word)
              begin += len(new_word.phones)
        return new_sent

    def remove_silence(self, sent):
        new_sent = Sentence([])
        for w_idx, word in enumerate(sent.words): 
            new_word = Word([], w_idx)
            for phn in word.phones:
                if phn.text != SIL.lower():
                    new_word.add(Phone(phn.text, phn.begin, phn.end))
            if len(new_word) > 0:
                new_sent.add(new_word)
        return new_sent
    
    def map_phones_to_phonemes(self, sent):
        new_sent = Sentence([])
        for w_idx, word in enumerate(sent.words):
            new_word = Word([], w_idx)
            for phn_idx, phn in enumerate(word.phones):
                new_word.add(Phone(self.phone2phoneme[phn.text], 
                                   phn.begin, 
                                   phn.end))
            new_sent.add(new_word)
        return new_sent
    
    def align(self, src_sent, trg_sent):
        """
        Align with the target sentence based on minimizing edit distance
        """
        w_ins, w_del, w_sub = 1.0, 1.0, 1.0
        src_len = len(src_sent)
        trg_len = len(trg_sent)
    
        D = np.zeros((src_len+1, trg_len+1))
        back_ptr = dict()
        alignment = dict()
        for i in range(src_len+1):
            D[i, 0] = w_del * i
            back_ptr[(i-1, -1)] = (i-2, -1)
        
        for j in range(trg_len+1):
            D[0, j] = w_ins * j
            back_ptr[(-1, j-1)] = (-1, j-2)

        for i in range(1, src_len+1):
            for j in range(1, trg_len+1):
                if src_sent.text[i-1] == trg_sent.text[j-1]: # Exact match
                    D[i, j] = D[i-1, j-1]
                    back_ptr[(i-1, j-1)] = (i-2, j-2)
                else:
                    D_ins = D[i, j-1] + w_ins
                    D_del = D[i-1, j] + w_del
                    D_sub = D[i-1, j-1] + w_sub
                    D_min = min(D_ins, D_del, D_sub)
                    D[i, j] = D_min
                    if D_ins == D_min: # Insert
                        back_ptr[(i-1, j-1)] = (i-1, j-2)
                    elif D_del == D_min: # Delete
                        back_ptr[(i-1, j-1)] = (i-2, j-1)
                    else: # Substitute
                        back_ptr[(i-1, j-1)] = (i-2, j-2)

        cur = (src_len-1, trg_len-1)
        while cur != (-1, -1):
            if not cur[0] in alignment: 
                alignment[max(cur[0], 0)] = [max(cur[1], 0)]
            else:
                alignment[max(cur[0], 0)].append(max(cur[1], 0))
            cur = back_ptr[cur]
        
        alignment = [alignment[src_idx] for src_idx in sorted(alignment)]
        i = 0 
        word_idx = 0
        aligned_sent = Sentence([Word([], 0)])
        while i < len(alignment):
            js = alignment[i]
            if i <= src_sent.words[word_idx].end:
                for j in js:
                    if not aligned_sent.contains_phone(trg_sent.phones[j]):
                        aligned_sent.add_phone(word_idx, deepcopy(trg_sent.phones[j]))
                i += 1
            else:
                word_idx += 1
                aligned_sent.add(Word([], i))
        return aligned_sent
    
    def extract_forced_alignments(self, phone_ali_path, word_ali_path):          
        f_word_ali = open(word_ali_path, 'w')
        f_phn_ali = open(phone_ali_path, 'w')
        n_match = 0
        for audio_id in sorted(self.orig_sents):
            beer_sent = self.remove_silence(self.beer_sents[audio_id])
            orig_phone_sent = self.orig_sents[audio_id]
            orig_sent = self.map_phones_to_phonemes(orig_phone_sent)
            orig_sent = self.remove_repeated(orig_sent)
            orig_words = orig_sent.words
            
            #if len(orig_sent) != len(beer_sent):
            #    print(f'orig_sent: {orig_sent.text} ({len(orig_sent.text)}), beer_sent: {beer_sent.text} ({len(beer_sent.text)})')
            if len(orig_sent) == len(beer_sent):
                n_match += 1
            
            aligned_sent = self.align(orig_sent,
                                      beer_sent)
            if len(beer_sent) != len(aligned_sent):
              print(audio_id, len(beer_sent), len(aligned_sent)) # XXX
        
            # Save alignments
            for w in aligned_sent.words:
                if len(w) > 0:
                    w_str = ''.join(w.text)
                    f_word_ali.write(f'{audio_id} {w.begin_sec} {w.end_sec} {w_str}\n')
                    for phn in w.phones:
                        f_phn_ali.write(f'{audio_id} {phn.begin_sec} {phn.end_sec} {phn.text}\n')

        f_word_ali.close()
        f_phn_ali.close()
        print(f'Number of matching sentences: {n_match}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('TASK', type=int)
    args = parser.parse_args()
    data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/full_corpus_newsplit/'
    word_ali_file = os.path.join(data_path, '../ZRC_scoring/mboshi/mboshi.wrd')
    phone_ali_file = os.path.join(data_path, '../ZRC_scoring/mboshi/mboshi.phn')
    out_prefix = os.path.join(data_path, 'all/all')
    
    if args.TASK == 0:
        extract_meta_data(word_ali_file,
                          phone_ali_file,
                          out_prefix,
                          debug=False)
    elif args.TASK == 1:
        extract_word_dataset(data_path, debug=False)
        #extract_word_dataset(data_path, debug=False, order=[300, 600])
        #extract_word_dataset(data_path, debug=False, order=[600, 900])
    elif args.TASK == 2:
        extract_vocab(data_path)
    elif args.TASK == 3:
        extract_kaldi_multilingual_phones_for_words_wo_alignment(data_path)
    elif args.TASK == 4:
        extract_kaldi_multilingual_phones_for_words(data_path)
    elif args.TASK == 5:
        extract_kaldi_multilingual_phones(data_path)
    elif args.TASK == 6:
        beer_file = '/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud/local/mboshi/all_phoneme'
        wavscp_file = '/ws/ifp-53_1/hasegawa/tools/kaldi/egs/discophone/s5/data/full/wav.scp' 
        text_file = '/ws/ifp-53_1/hasegawa/tools/kaldi/egs/discophone/s5/data/full/text'
        phone_itos_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/mboshi_phoneme_mappings/phone_itos.json'
        phoneme_itos_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/mboshi_phoneme_mappings/phoneme_itos.json'
        phone2phoneme_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/mboshi_phoneme_mappings/phone2phoneme_sym_SF_draft.txt'
        corpus = Corpus(wavscp_file, 
                        text_file, 
                        beer_file, 
                        phone_itos_file, 
                        phoneme_itos_file, 
                        phone2phoneme_file)
        corpus.extract_forced_alignments('mboshi.phn', 'mboshi.wrd')
    elif args.TASK == 7:
        in_path = os.path.join(data_path, 'mboshi_word/mboshi_word.json')
        multilang_path = os.path.join(data_path, '../liming_iclr2022/siyuan_multi13lang/to_liming/aud_output/without_gold_segmentation/output_feats.ark.txt_seg_clusters60_rand_0')
        out_path = os.path.join(data_path, 'mboshi_word/mboshi_word_with_multilang_pred_segments.json')
        extract_multilingual_phones_for_words(in_path, multilang_path, out_path)
    elif args.TASK == 8:
        in_path = os.path.join(data_path, 'all/all.json')
        multilang_path = os.path.join(data_path, '../liming_iclr2022/siyuan_multi13lang/to_liming/aud_output/without_gold_segmentation/output_feats.ark.txt_seg_clusters60_rand_0')
        out_path = os.path.join(data_path, 'all/all_with_multilang_pred_segments.json')
        extract_multilingual_phones(in_path, multilang_path, out_path)
    elif args.TASK == 9:
        in_path = os.path.join(data_path, 'all/all.json')
        multilang_path = os.path.join(data_path, '../liming_iclr2022/siyuan_multi13lang/to_liming/aud_output/with_gold_segmentation/output_feats.ark.txt.gold_ali_seg_clusters30_rand_0')
        out_path = os.path.join(data_path, 'all/all_with_multilang_phones.json')
        extract_multilingual_phones(in_path, multilang_path, out_path, key_name='multilingual_phones')
    elif args.TASK == 10:
        in_path = os.path.join(data_path, 'mboshi_word/mboshi_word.json')
        multilang_path = os.path.join(data_path, '../liming_iclr2022/siyuan_multi13lang/to_liming/aud_output/with_gold_segmentation/output_feats.ark.txt.gold_ali_seg_clusters30_rand_0')
        out_path = os.path.join(data_path, 'mboshi_word/mboshi_word_with_multilang_phones.json')
        extract_multilingual_phones_for_words(in_path, multilang_path, out_path, key_name='multilingual_phones')
    elif args.TASK == 11:
        extract_ngram_dataset(phone_ali_file, data_path, debug=False)
    elif args.TASK == 12:
        in_path = os.path.join(data_path, 'mboshi_word/mboshi_word_3-10gram.json')
        multilang_path = os.path.join(data_path, '../liming_iclr2022/siyuan_multi13lang/to_liming/aud_output/without_gold_segmentation/output_feats.ark.txt_seg_clusters60_rand_0')
        out_path = os.path.join(data_path, 'mboshi_word/mboshi_word_3-10gram_with_multilang_pred_segments.json')
        extract_multilingual_phones_for_words(in_path, multilang_path, out_path)
    elif args.TASK == 13:
        extract_ngram_dataset(phone_ali_file, data_path, n_min=2, n_max=2, debug=False)
    elif args.TASK == 14:
        in_path = os.path.join(data_path, 'mboshi_word/mboshi_word_2-2gram.json')
        multilang_path = os.path.join(data_path, '../liming_iclr2022/siyuan_multi13lang/to_liming/aud_output/without_gold_segmentation/output_feats.ark.txt_seg_clusters60_rand_0')
        out_path = os.path.join(data_path, 'mboshi_word/mboshi_word_2-2gram_with_multilang_pred_segments.json')
        extract_multilingual_phones_for_words(in_path, multilang_path, out_path)
    elif args.TASK == 15:
        for split in ['train', 'dev']:
            wavs = os.listdir(os.path.join(data_path, split))
            split_metadata(os.path.join(data_path, 'all/all.json'), wavs, os.path.join(data_path, f'{split}/{split}.json'))
