import sys
import os
import re
import json
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.io import wavfile

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
          val-clean/
              ...
              val-clean.item
          test-clean/
              ...
              test-clean.item
  """
  dataset_name = "librispeech_word"
  sent_info_file = os.path.join(data_path, split, f"{split}.json") 
  word_info_file = os.path.join(data_path, dataset_name, "{dataset_name}.json")
  sent_f = open(sent_info_file, 'r')
  word_f = open(word_info_file, 'w')

  for split in ["train-clean-100", "val-clean", "test-clean"]:
    dataset = os.path.join(data_path, dataset_name, split)
    if not os.path.exists(dataset):
      os.makedirs(dataset)

    abx_f = open(os.path.join(data_path, dataset_name, split, split+".item"), "w")
    abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

    for line in sent_f:
      sent_dict = json.loads(line.rstrip("\n"))
      audio_id = sent_dict["audio_id"]
      audio_path = os.path.join(data_path, split, f"{audio_id}.wav")
      fs, audio = wavfile.read(audio_path) 
      spk = audio_id.split("_")[0]

      visual_word_idxs = sent_dict["visual_words"]
      words = sent_dict["words"]
      for word_idx in visual_word_idxs:
        word = words[word_idx]
        word_id = f"{audio_id}_{word_idx}"
        word_audio_path = os.path.join(data_path, dataset_name, split, f"{word_id}.wav")
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
          if phn_idx < len(word["phonemes"]) - 1:
            next_phn = word["phonemes"][phn_idx+1]["text"]
          abx_f.write("{audio_id}_{word_id} {phn['begin']} {phn['end']} {phn['text']} {prev_phn} {next_phn} {spk}\n") 
    abx_f.close()
  sent_f.close()
  word_f.close()
  
   
def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  parser.add_argument("--data_path", default="/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic")
  parser.add_argument("--split", default="train-clean-100")

  args = parser.parse_args(argv)
  if args.TASK == 0:
    visual_word_file = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/phrase_classes.json"
    extract_visual_words(args.data_path, args.split, visual_word_file)
  elif args.TASK == 1:
    split = re.sub("-", "_", args.split)
    pseudo_phone_file = f"/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/exp/train_pytorch_train_li10/decode_librispeech/{split}_decode_li10/data.json"
    extract_pseudo_phones(args.data_path, args.split, pseudo_phone_file)
  elif args.TASK == 2:
    split = re.sub("-", "_", args.split)
    pseudo_phone_file = f"/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/exp/train_pytorch_train_li10/decode_flickr/{split}_decode_li10/data.json"
    extract_pseudo_phones(args.data_path, args.split, pseudo_phone_file)

if __name__ == "__main__":
  args = sys.argv[1:]
  main(args) 
