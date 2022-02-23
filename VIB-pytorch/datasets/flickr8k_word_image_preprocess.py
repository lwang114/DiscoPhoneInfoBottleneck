import sys
import os
import json
import argparse
import re
from itertools import groupby
UNK = "###UNK###"
NULL = "###NULL###"
BLANK = "###BLANK###"
IGNORED_TOKENS = ["SIL", "GARBAGE"]
TONES = [chr(i) for i in range(741, 748)]
SPECIAL_TOKENS = ['ʰ', 'ʔ', 'ʼ', 'ˈ'] 
print(TONES)

def extract_pseudo_phones(data_path):
  """
  Args :
    data_path : str, path to the LibriSpeech root
    split : str, {train-clean-100, train-clean-360, train-other-500}
    pseudo_phone_file : str, storing a dict of
      {"utts" : 
          {audio_id} : 
              "output" : [{"rec_text" : str,
                           "rec_token" : str}],
              "utt2spk" : str
          }
      }
  """
  pseudo_phones = dict()
  for split in ["train_flickr_audio", "val_flickr_audio", "test_flickr_audio"]:
    pseudo_phone_file = f"/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/exp/train_pytorch_train_li10/decode_flickr/{split}_decode_li10/data.json"
    pseudo_phones.update(json.load(open(pseudo_phone_file))["utts"])
  in_file = os.path.join(data_path, f"flickr8k_word_50.json")
  out_file = os.path.join(data_path, f"flickr8k_word_50_with_pseudo_phones.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  tokens = set()

  for line in in_f:
    sent_dict = json.loads(line.rstrip("\n"))
    audio_id = '_'.join(sent_dict["audio_id"].split('/')[-1].split('_')[1:])
    spk = sent_dict["spk"]
    sent_dict["pseudo_phones"] = []

    utt_id = f'{int(spk):05d}_{audio_id}'
    if not utt_id in pseudo_phones:
      print(f'{utt_id} not found')
      continue

    rec_tokens = pseudo_phones[utt_id]["output"][0]["rec_token"]
    for phn in rec_tokens.split():
      if phn in IGNORED_TOKENS or (phn[0] == '<'):
        continue
      if not phn in tokens:
        tokens.add(phn)
      sent_dict["pseudo_phones"].append(phn)
    out_f.write(json.dumps(sent_dict)+"\n") 
  in_f.close()
  out_f.close()
  print(tokens)
  print(f"Pseudo-phone set size: {len(tokens)}")

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
  in_file = os.path.join(data_path, f"flickr8k_word_50.json")
  out_file = os.path.join(data_path, f"flickr8k_word_50_with_multilingual_phones.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  tokens = set()
  multilingual_phones = dict()
  for split in ["train_flickr8k_word_50", "val_flickr8k_word_50", "test_flickr8k_word_50"]:
    multilingual_phone_file = f"/ws/ifp-53_1/hasegawa/tools/kaldi/egs/discophone/v1_multilang/exp/gmm/tri5_{split}/merged_alignment.txt"
    with open(multilingual_phone_file, "r") as f:
      phone_list = [line.rstrip('\n').split() for line in f]

    for utt_id, group in groupby(phone_list, lambda x:x[0]):
      for phn_info in group:
        _, spk, begin, end, phn = phn_info
        if not utt_id in multilingual_phones:
          multilingual_phones[utt_id] = []

        if not phn in tokens:
          tokens.add(phn)
        multilingual_phones[utt_id].append({"begin": float(begin),
                                            "end": float(begin)+float(end),
                                            "text": phn})

    for line in in_f:
      sent_dict = json.loads(line.rstrip("\n"))
      audio_id = "_".join(sent_dict["audio_id"].split("/")[-1].split("_")[1:])
      spk = sent_dict["spk"]
      word_idx = int(sent_dict["word_id"])
      sent_dict["multilingual_phones"] = []

      utt_id = f"{int(spk):05d}_{audio_id}_{word_idx:04d}"
      if not utt_id in multilingual_phones:
        # print(f"{utt_id} not found")
        continue
      print(utt_id)  
      sent_dict["multilingual_phones"].extend(multilingual_phones[utt_id])
      out_f.write(json.dumps(sent_dict)+"\n")
  in_f.close()
  out_f.close()
  print(tokens)
  print(f"Multilingual phone set size: {len(tokens)}")
      
def extract_kaldi_multilingual_phones_wo_alignment(data_path):
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

  in_file = os.path.join(data_path, f"flickr8k_word_50.json")
  out_file = os.path.join(data_path, f"flickr8k_word_50_with_multilingual_phones.json")
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
      print(f'Filling in the empty phone for {utt_id}')
    
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
    audio_id = "_".join(sent_dict["audio_id"].split("/")[-1].split("_")[1:])
    spk = sent_dict["spk"]
    word_idx = int(sent_dict["word_id"])
    sent_dict["multilingual_phones"] = []

    utt_id = f"{int(spk):05d}_{audio_id}_{word_idx:04d}"
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


def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  parser.add_argument("CONFIG", type=str)
  args = parser.parse_args(argv)
  config = json.load(open(args.config))

  if args.TASK == 0:
    extract_pseudo_phones(config["data_path"])
  elif args.TASK == 1:
    extract_kaldi_multilingual_phones(config["data_path"])
  elif args.TASK == 2:
    extract_kaldi_multilingual_phones_wo_alignment(config["data_path"])

if __name__ == "__main__":
  argv = sys.argv[1:]
  main(argv)
