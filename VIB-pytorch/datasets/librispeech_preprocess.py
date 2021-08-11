import sys
import os
import re
import json
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.io import wavfile
import re

stop_words = stopwords.words("english")
IGNORE_TOKENS = ["ʰ", "ʼ", "ˈ"]
SIL = "SIL" 
def extract_visual_words(data_path, split, visual_word_file):
  """
  Args :
    data_path : str, path to the LibriSpeech root
        transcription format: {dir1-dir2-sent_id} {word 1 in cap} ... {word last in cap}  
    split : str, {train-clean-100, train-clean-369, train-other-500}
    visual_word_file : str, filename storing the dictionary of visual words 

  Returns :
    {split}/{split}_with_visual_words.json : str, storing in each line dict in the form
        {"audio_id": str, 
         "text": str, transcript of the audio, 
         "visual_words": str, visual words of the audio}
  """
  lemmatizer = WordNetLemmatizer()
  visual_words = json.load(open(visual_word_file))
  in_file = os.path.join(data_path, split, f"{split}.json")
  out_file = os.path.join(data_path, split, f"{split}_with_visual_words.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  n_sents = 0
  n_words = 0
  vocab = set()

  for line in in_f:
    sent_dict = json.loads(line.rstrip("\n"))
    sent_dict["visual_words"] = []
    sent_dict["visual_lemmas"] = []
    for w_idx, w in enumerate(sent_dict["words"]):
      lemma = lemmatizer.lemmatize(w["text"].lower())
      if lemma in visual_words and not lemma in stop_words:
        if visual_words[lemma] > 50:
          vocab.add(lemma)
          sent_dict["visual_words"].append(w_idx)
          sent_dict["visual_lemmas"].append(lemma)
    if len(sent_dict["visual_lemmas"]) > 0:
      n_sents += 1
      n_words += len(sent_dict["visual_lemmas"])
      print(sent_dict["utterance_id"], sent_dict["visual_lemmas"])
    out_f.write(json.dumps(sent_dict)+"\n")
  in_f.close()
  out_f.close()
  print(f"Visual vocabulary size: {len(vocab)}")
  print(f"Number of visual words: {n_words}")
  print(f"Number of sentences with visual words: {n_sents}")

def extract_pseudo_phones(data_path, split, pseudo_phone_file):
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
  pseudo_phones = json.load(open(pseudo_phone_file))["utts"]
  in_file = os.path.join(data_path, split, f"{split}.json")
  out_file = os.path.join(data_path, split, f"{split}_with_pseudo_phones.json")
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  tokens = set()

  for line in in_f:
    sent_dict = json.loads(line.rstrip("\n"))
    audio_id = sent_dict["utterance_id"].split('/')[-1]
    sent_dict["pseudo_phones"] = []

    if not audio_id in pseudo_phones:
      print(f'{audio_id} not found')
      continue

    rec_tokens = pseudo_phones[audio_id]["output"][0]["rec_token"]
    for phn in rec_tokens.split():
      # if phn in IGNORE_TOKENS or (phn[0] == '<'):
      #   continue
      if not phn in tokens:
        tokens.add(phn)
      sent_dict["pseudo_phones"].append(phn)
    out_f.write(json.dumps(sent_dict)+"\n") 
  in_f.close()
  out_f.close()
  print(tokens)
  print(f"Pseudo-phone set size: {len(tokens)}")

def extract_word_dataset(data_path, debug=False):
  """
  Create a spoken word dataset organized as:
      root/
          librispeech_word.json
          train-clean-100/
              {audio_id}_{word_id}.wav
              ...
              train-clean-100.item
          train-clean-360/
              ...
              train-clean-360.item
          dev-clean/
              ...
              dev-clean.item
  """
  dataset_name = "librispeech_word"
  dataset_path = os.path.join(data_path, dataset_name)
  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
  word_info_file = os.path.join(dataset_path, f"{dataset_name}.json")
  word_f = open(word_info_file, 'w')

  for split in ["dev-clean", "train-clean-100", "train-clean-360"]: 
    sent_info_file = os.path.join(data_path, split, f"{split}.json") 
    sent_f = open(sent_info_file, 'r')
    dataset = os.path.join(dataset_path, split)
    if not os.path.exists(dataset):
      os.makedirs(dataset)

    abx_f = open(os.path.join(dataset, split+".item"), "w")
    abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

    global_idx = 0
    for line in sent_f:
      if debug and global_idx > 20:
        break
      sent_dict = json.loads(line.rstrip("\n"))
      audio_id = sent_dict["utterance_id"]
      audio_path = os.path.join(data_path, split, f"{audio_id}.wav")
      if not os.path.exists(audio_path):
        continue
      fs, audio = wavfile.read(audio_path)
      spk = audio_id.split("-")[0]

      visual_word_idxs = sent_dict["visual_words"]
      global_idx += len(visual_word_idxs)
      words = sent_dict["words"]
      print(split, audio_id)
      for word_idx in visual_word_idxs:
        word = words[word_idx]
        word_id = f"{audio_id}_{word_idx}"
        word_audio_path = os.path.join(dataset, f"{word_id}.wav")
        begin_sec = word["phonemes"][0]["begin"]
        end_sec = word["phonemes"][-1]["end"] 
        begin = int(begin_sec * 16000) 
        end = int(end_sec * 16000)
        word_audio = audio[begin:end]
      
        wavfile.write(word_audio_path, fs, word_audio)
        
        word_info = {"audio_id": audio_id,
                     "word_id": str(word_idx),
                     "label": word["text"],
                     "begin": begin_sec,
                     "end": end_sec,
                     "spk": spk,
                     "split": split,
                     "phonemes": word["phonemes"]}
        word_f.write(json.dumps(word_info)+"\n")

        # Extract ABX file
        for phn_idx, phn in enumerate(word["phonemes"]):
          prev_phn = SIL
          next_phn = SIL
          if phn_idx > 0:
            prev_phn = word["phonemes"][phn_idx-1]["text"]
            prev_phn = re.sub(r"[0-9]", "", prev_phn)
          if phn_idx < len(word["phonemes"]) - 1:
            next_phn = word["phonemes"][phn_idx+1]["text"]
            next_phn = re.sub(r"[0-9]", "", next_phn)
          phn_label = re.sub(r"[0-9]", "", phn["text"])
          
          begin_phn = round(phn["begin"] - begin_sec, 3) 
          end_phn = round(phn["end"] - begin_sec, 3)
          abx_f.write(f"{word_id} {begin_phn} {end_phn} {phn_label} {prev_phn} {next_phn} {spk}\n") 

    abx_f.close()
    sent_f.close()
    print(f"Number of words in {split}: {global_idx+1}")
  word_f.close()

def extract_noun(data_path, out_file):
  lemmatizer = WordNetLemmatizer()
 
  nouns = dict()
  for split in ["train-clean-100", "train-clean-360", "dev-clean"]:
    sent_info_file = os.path.join(data_path, f"{split}/{split}.json")
    sent_f = open(sent_info_file, "w")
    for line in sent_f:
      sent_info = json.loads(line.rstrip("\n"))
      sent = [word["text"] for word in sent_info["word"]]
      # POS Tagging
      sent_nouns = [lemmatizer.lemmatize(tagged[0]) for tagged in nltk.pos_tag(sent) if tagged[1][0] == "N"]
      for w in sent_nouns:
        if not w in nouns:
          nouns[w] = 1
        else:
          nouns[w] += 1
  json.dump(nouns, open(out_file, "w"), indent=2)

def extract_top_noun_dataset(data_path, order=[0, 99], debug=False):
  """
  Create a spoken word dataset with the top order[0]-order[1] nouns from LibriSpeech 
  """
  lemmatizer = WordNetLemmatizer()

  dataset_name = f"librispeech_word_top{order[0]}-{order[1]}"
  dataset_path = os.path.join(data_path, dataset_name)
  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
  word_info_file = os.path.join(dataset_path, f"{dataset_name}.json")
  word_f = open(word_info_file, "w")

  vocab_path = os.path.join(data_path, "librispeech_nouns.json")
  if not os.path.exists(vocab_path):
    extract_noun(data_path, vocab_path)
  vocab = json.load(open(vocab_path))
  top_words = sorted(vocab, key=lambda x:vocab[x], reverse=True)[order[0]:order[1]+1]

  for split in ["train-clean-100", "train-clean-360", "dev-clean"]:
    split_path = os.path.join(dataset_path, f"{split}_top{order[0]}-{order[1]}")
    if not os.path.exists(split_path):
      os.makedirs(split_path)

    abx_f = open(os.path.join(dataset, f"{split}_top{order[0]}-{order[1]}_nonoverlap.item"), "w")
    abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

    sent_info_file = os.path.join(data_path, f"{split}/{split}.json")
    sent_f = open(sent_info_file, "r")
    global_idx = 0
    for line in sent_f:
      if debug and global_idx > 20:
        break
      global_idx += 1

      sent_info = json.loads(line.rstrip("\n"))
      audio_id = sent_info["utterance_id"]
      spk = audio_id.split("-")[0]
      audio_path = os.path.join(data_path, split, audio_id+".wav")
      fs, audio = wavfile.read(audio_path)
      
      for word_id, word in enumerate(sent_info["words"]):
        lemma = lemmatizer.lemmatize(word["text"])  
        if word_id in sent_info["visual_words"]:
          continue

        if lemma in top_words:
          # Save meta info
          word_info = {"audio_id": audio_id,
                       "word_id": str(word_id),
                       "label": word["text"],
                       "begin": word["begin"],
                       "end": word["end"],
                       "spk": spk,
                       "split": split,
                       "phonemes": word["phonemes"]}
          word_f.write(json.dumps(word_info)+"\n")
        
          # Save ABX info
          for phn_idx, phn in enumerate(word["phonemes"]):
            prev_phn = SIL
            next_phn = SIL
            if phn_idx > 0:
              prev_phn = word["phonemes"][phn_idx-1]["text"]
              prev_phn = re.sub(r"[0-9]", "", prev_phn)
            if phn_idx < len(word["phonemes"]) - 1:
              next_phn = word["phonemes"][phn_idx+1]["text"]
              next_phn = re.sub(r"[0-9]", "", next_phn)
            phn_label = re.sub(r"[0-9]", "", phn["text"])
            
            begin_phn = round(phn["begin"] - begin_sec, 3) 
            end_phn = round(phn["end"] - begin_sec, 3)
            abx_f.write(f"{word_id} {begin_phn} {end_phn} {phn_label} {prev_phn} {next_phn} {spk}\n") 

          # Copy wav file
          word_audio = audio[int(word["begin"]*fs):int(word["end"]*fs)]
          word_audio_path = os.path.join(split_path, f"{audio_id}_{word_id}.wav")
          wavfile.write(word_audio_path, fs, word_audio)
    abx_f.close()
    sent_f.close()
  word_f.close()

def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  # parser.add_argument("--data_path", default="/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic")
  parser.add_argument("--config", default="../configs/librispeech_word_segment_cpc_info_quantizer.json")
  parser.add_argument("--split", default="train-clean-100")

  args = parser.parse_args(argv)
  config = json.load(open(args.config))
  data_path = config["data_path"]

  if args.TASK == 0:
    visual_word_file = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/phrase_classes.json"
    extract_visual_words(data_path, args.split, visual_word_file)
  elif args.TASK == 1:
    split = re.sub("-", "_", args.split)
    pseudo_phone_file = f"/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/exp/train_pytorch_train_li10/decode_librispeech/{split}_decode_li10/data.json"
    extract_pseudo_phones(data_path, args.split, pseudo_phone_file)
  elif args.TASK == 2:
    split = re.sub("-", "_", args.split)
    pseudo_phone_file = f"/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/exp/train_pytorch_train_li10/decode_flickr/{split}_decode_li10/data.json"
    extract_pseudo_phones(data_path, args.split, pseudo_phone_file)
  elif args.TASK == 3:
    extract_word_dataset(data_path, debug=config["debug"])
  elif args.TASK == 4:
    extract_top_noun_dataset(data_path, order=[0, 200], debug=config["debug"])

if __name__ == "__main__":
  args = sys.argv[1:]
  main(args) 
