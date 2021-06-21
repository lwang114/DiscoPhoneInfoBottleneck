import json
import os
from scipy.io import wavfile
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import argparse
import sys
import shutil
import re
from tqdm import tqdm
stop_words = stopwords.words("english")
SIL = "SIL"
IGNORED_TOKENS = ["SIL", "GARBAGE"]

def extract_word_dataset(data_path,
                         min_class_size=50,
                         debug=False):
  """
  Create a dataset organized as:
      root/
          word_info.json
          train/
              {audio_id}_{word_id}.wav
              ...
              train.item
          val/
              ...
              val.item
          test/
              ...
              test.item
  """
  utt2spk_file = os.path.join(data_path, "flickr_audio/wav2spk.txt")
  spk_f = open(utt2spk_file, "r")
  utt2spk = dict()
  for line in spk_f:
    audio_file, spk = line.split() 
    utt2spk[audio_file.split('.')[0]] = spk
   
  dataset_name = f"flickr8k_word_{min_class_size}"
  dataset_path = os.path.join(data_path, dataset_name)
  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
  
  class_sizes = json.load(open(os.path.join(data_path, "phrase_classes.json")))
  word_info_file = os.path.join(data_path, dataset_name, dataset_name+'.json')
  word_f = open(word_info_file, "w") 
  for split in ["train", "val", "test"]:
    dataset = os.path.join(data_path, dataset_name, split)
    if not os.path.exists(dataset):
      os.makedirs(dataset)

    with open(os.path.join(data_path, f"splits/flickr40k_{split}.txt"), "r") as f:
      filenames = [line.rstrip("\n").split("/")[-1] for line in f]
    
    abx_f = open(os.path.join(dataset_path, split, split+".item"), "w")
    abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")
    prev_id = ""
    word_idx = 0
    n_words = 0
    vocab = set()

    phrase_file = os.path.join(data_path, "flickr8k_phrases.json")
    phrase_f = open(phrase_file, "r")
    for line in phrase_f:
      if debug and n_words > 500:
        break
      phrase = json.loads(line.rstrip("\n"))
      label = phrase["label"]
      audio_id = phrase["utterance_id"]
      spk = utt2spk[audio_id]
      if audio_id != prev_id:
        print(audio_id, split)
        word_idx = 0
        prev_id = audio_id

      audio_file = audio_id + ".wav"
      audio_path = os.path.join(data_path, "flickr_audio/wavs", audio_file)

      if not label in stop_words and audio_file in filenames and (class_sizes[label] >= min_class_size):
        vocab.add(label)
        n_words += 1
        fs, audio = wavfile.read(audio_path)
        if len(phrase["children"]): 
          for word in phrase["children"]:
            word_idx += 1
            begin = word['begin']
            end = word["end"]
            begin_frame = int(begin * fs)
            end_frame = int(end * fs)
            audio_word = audio[begin_frame:end_frame+1]
            word_file = f"{audio_id}_{word_idx:04d}.wav"
            word_path = os.path.join(data_path, dataset_name, split, word_file)
            wavfile.write(word_path, fs, audio_word)
            word_info = {"audio_id": audio_id,
                         "image_id": "_".join(audio_id.split("_")[:-1]),
                         "word_id": str(word_idx),
                         "label": label,
                         "begin": begin,
                         "end": end,
                         "box": phrase["bbox"],
                         "box_id": phrase["feat_idx"],
                         "spk": spk,
                         "split": split,
                         "phonemes": word}
            word_f.write(json.dumps(word_info)+"\n")

            for phn_idx, phone in enumerate(word["children"]):
              if (phn_idx == 0) or (phn_idx == len(word["children"]) - 1):
                continue
              phn = phone["text"] # TODO Remove silence
              if phn[0] == "+": # If the unit is a disfluency such as breath, laughter, etc.
                continue
              begin_phn = round(phone["begin"] - begin, 3)
              end_phn = round(phone["end"] - begin, 3)
              prev_phn = word["children"][phn_idx-1]["text"]
              next_phn = word["children"][phn_idx+1]["text"]
              abx_f.write(f"{word_file.split('.')[0]} {begin_phn} {end_phn} {phn} {prev_phn} {next_phn} {spk}\n")
    abx_f.close()
    phrase_f.close()
  print(f"Number of words: {n_words}, vocab size: {len(vocab)}")
  word_f.close() 

def extract_zs_item_file(data_path, 
                         min_class_size, 
                         max_keep_size):
  dataset_name = f"flickr8k_word_{min_class_size}"
  for split in ["train", "val", "test"]:
    word_f = open(os.path.join(data_path,
                               dataset_name,
                               f"{dataset_name}.json"), "r")
    abx_f = open(os.path.join(data_path, 
                              dataset_name, 
                              split, 
                              f"{split}_{max_keep_size}.item"), "w")
    abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

    label_counts = dict()    
    for line in word_f:
      word = json.loads(line.rstrip("\n"))
      label = word["label"]
      if not label in label_counts:
        label_counts[label] = 1
      else:
        label_counts[label] += 1
      if label_counts[label] > max_keep_size:
        continue

      if word["split"] != split:
        continue
      audio_id = word["audio_id"]
      word_id = int(word["word_id"])
      audio_file_id = f"{audio_id}_{word_id:04d}"
      print(audio_file_id, split) # XXX
      spk = word["spk"]
      phonemes = word["phonemes"]["children"]
      begin = word["begin"]
      for phn_idx, phone in enumerate(phonemes):
        # XXX if (phn_idx == 0) or (phn_idx == len(phonemes) - 1):
        #   continue
        phn = phone["text"]
        if (phn[0] == "+") or (phn in SIL):
          continue
        begin_phn = round(phone["begin"] - begin, 3)
        end_phn = round(phone["end"] - begin, 3) 
        prev_phn = SIL # XXX phonemes[phn_idx-1]["text"]
        next_phn = SIL # XXX phonemes[phn_idx+1]["text"]
        abx_f.write(f"{audio_file_id} {begin_phn} {end_phn} {phn} {prev_phn} {next_phn} {spk}\n")
    abx_f.close()
  word_f.close()

def extract_zs_item_file_full_data(data_path):
  for split in ["test", "train", "val"]:
    sent_f = open(os.path.join(data_path,
                               f"{split}_flickr_audio/{split}_flickr_audio.json"), "r")
    abx_f = open(os.path.join(data_path,
                              f"{split}_flickr_audio/{split}_flickr_audio.item"), "w")
    abx_f.write("#file_ID onset offset #phone prev-phone next-phone speaker\n")

    for line in sent_f:
      sent = json.loads(line.rstrip("\n"))
      audio_id = sent["utterance_id"].split("/")[-1]
      # print(audio_id, split) # XXX
      spk = 0
      phonemes = []
      for word in sent["words"]:
        for phn in word["phonemes"]:
          phn["text"] = re.sub(r"[0-9]", "", phn["text"])
          phonemes.append(phn)

      for phn_idx, phone in enumerate(phonemes):
        phn = phone["text"]
        if (phn[0] == "+") or (phn in IGNORED_TOKENS):
          continue
        begin_phn = round(phone["begin"], 3)
        end_phn = round(phone["end"], 3) 
        if phn_idx == 0:
          prev_phn = SIL
        else:
          prev_phn = phonemes[phn_idx-1]["text"]
        
        if phn_idx == len(phonemes) - 1:
          next_phn = SIL
        else:
          next_phn = phonemes[phn_idx+1]["text"]
        abx_f.write(f"{audio_id} {begin_phn} {end_phn} {phn} {prev_phn} {next_phn} {spk}\n")
    sent_f.close()
    abx_f.close()

def extract_audio_for_concepts(data_path, 
                               min_class_size,
                               concepts,
                               out_dir="flickr_audio_by_concepts"):
  """
  Extract audio files for a list of concepts organized as
    concept 1/
      word_boundary.json
      *.wav
    ...
    concept n/
      word_boundary.json
      *.wav 
  """
  dataset_name = f"flickr8k_word_{min_class_size}"
  word_f = open(os.path.join(data_path,
                             dataset_name,
                             dataset_name+".json"))
  for c in concepts:
    if not os.path.exists(os.path.join(data_path,
                                       out_dir)):
      os.makedirs(os.path.join(data_path,
                               out_dir))
    if not os.path.exists(os.path.join(
                            data_path,
                            out_dir, c)):
      os.makedirs(os.path.join(data_path,
                               out_dir, c))
  
  concept_info = dict()
  for line in word_f:
    word = json.loads(line.rstrip("\n"))
    c = word["label"]
    if c in concepts:
      if not c in concept_info:
        concept_info[c] = [word]
      else:
        concept_info[c].append(word)

      audio_id = word["audio_id"]
      audio_file = audio_id+".wav"
      cur_dir = os.path.join(data_path,
                             out_dir, c)
      if not audio_file in cur_dir:
        print(audio_file)
        shutil.copyfile(os.path.join(data_path, "flickr_audio/wavs", audio_file),
                        os.path.join(cur_dir, audio_file))

  for c in concepts:
    segment_file = os.path.join(data_path, out_dir, c, "word_boundary.json")
    concept_info_str = "\n".join([json.dumps(c_info) for c_info in concept_info[c]])
    with open(segment_file, "w") as f:
      f.write(concept_info_str)

def match_words_with_phones(words, phones): 
  """   
  Returns :
      example : a dict of
        {"text" : str, transcript,
         "words" : a list of dicts}
  """
  w_idx = 0
  text = [w["text"] for w in words]
  for phone in phones:
    begin, end = phone["begin"], phone["end"]
    word = words[w_idx]
    if word["end"] <= begin:
      w_idx += 1
      if w_idx >= len(words):
        break
    words[w_idx]["phonemes"].append(phone)
  return {"words": words, "text": text}

def extract_sentence_info(data_path, split):
  """
  Returns :
      flickr_audio_{split}.json : file storing the dict with items
          {"utterance_id" : str,
           "text" : str, transcript of the audio,
           "box" : a list of tuples, 
           "words" : a list of dicts of
               "phonemes" : a list of dicts of 
                   "begin" : float,
                   "end" : float,
                   "text" : str}
  """
  lemmatizer = WordNetLemmatizer()
  word_dir = os.path.join(data_path, "word_segmentation") 
  phone_f = open(os.path.join(data_path, "flickr_labels.txt"), "r") 
  split_file = os.path.join(data_path, f"splits/flickr40k_{split}.txt")
  with open(split_file, "r") as f:
    audio_ids = [line.rstrip("\n").split("/")[-1].split(".")[0] for line in f]
  
  sent_f = open(os.path.join(data_path, f"{split}_flickr_audio.json"), "w")
  phones = []
  words = []
  audio_id = None
  idx = 0
  for line in phone_f:
    if "align" in line:      
      if audio_id in audio_ids:
        print(audio_id)
        example = match_words_with_phones(words, phones)
        example["utterance_id"] = os.path.join(data_path, f"flickr_audio/wavs/{audio_id}")
        sent_f.write(json.dumps(example)+"\n")

      audio_id = "_".join(line.rstrip("\n").split(".")[0].split("_")[1:])
      phones = []
      words = []
      if not audio_id in audio_ids:
        continue
      # XXX if idx > 10:
      #   break
      # idx += 1
      word_fn = os.path.join(word_dir, audio_id+".words")
      if not os.path.exists(word_fn):
        continue
      with open(word_fn, "r") as word_f:
        for line in word_f:
          w, begin, end = line.split()
          if ("$" in w) or ("+" in w):
            continue
          w = re.sub(r"[^\w]", "", w)
          w = re.sub(r"[0-9]", "", w)
          words.append({"text": w, 
                        "begin": float(begin), 
                        "end": float(end),
                        "phonemes": []})
    else:
      if not audio_id in audio_ids:
        continue
      phn, begin, end = line.split()
      phones.append({"text": phn,
                     "begin": float(begin),
                     "end": float(end)})
  example = match_words_with_phones(words, phones)
  example["utterance_id"] = os.path.join(data_path, f"flickr_audio/wavs/{audio_id}")

  sent_f.write(json.dumps(example)+"\n") 
  phone_f.close()
  sent_f.close()


def extract_visual_words(data_path,
                         split,
                         min_class_size):
  """
  Add the following keys :
    "visual_words" : a list of visual words idxs
    "bboxes" : a list of tuple of (left, upper, right, lower)
  """
  lemmatizer = WordNetLemmatizer()
  visual_classes = json.load(open(os.path.join(data_path, "phrase_classes.json")))  
  word_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  in_f = open(os.path.join(data_path, f"{split}.json"), "r")
  out_f = open(os.path.join(data_path, f"{split}_with_visual_words.json"), "w")

  visual_words = dict()
  for line in word_f:
    phrase = json.loads(line.rstrip("\n"))
    label = phrase["label"]
    if visual_classes[label] < min_class_size:
      continue 
    
    audio_id = phrase["utterance_id"]
    if not audio_id in visual_words:
      visual_words[audio_id] = dict()

    for word in phrase["children"]:
      visual_words[audio_id][(word["begin"], word["end"])] = {"text": word["text"],
                                                              "bbox": phrase["bbox"]}

  for line in in_f:
    sent = json.loads(line.rstrip("\n"))
    sent["visual_words"] = []
    sent["bboxes"] = []
    audio_id = sent["utterance_id"].split("/")[-1]
    if not audio_id in visual_words:
      out_f.write(json.dumps(sent)+"\n")
      continue
    print(audio_id)
    for w_idx, word in enumerate(sent["words"]):
      interval = (word["begin"], word["end"])
      # print(interval, visual_words[audio_id].keys()) # XXX
      if interval in visual_words[audio_id]:
        sent["visual_words"].append(w_idx)
        sent["bboxes"].append(visual_words[audio_id][interval]["bbox"])
    out_f.write(json.dumps(sent)+"\n")
  word_f.close()
  in_f.close()
  out_f.close()


def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  parser.add_argument("--config", type=str, default="../configs/flickr8k.json")
  args = parser.parse_args(argv)
  config = json.load(open(args.config))
  if args.TASK == 0:
    extract_word_dataset(config["data_path"],
                         config["min_class_size"],
                         config["debug"])
  elif args.TASK == 1:
    extract_zs_item_file(config["data_path"],
                         config["min_class_size"],
                         config["max_keep_size"])
  elif args.TASK == 2:
    classes = json.load(open(os.path.join(config["data_path"],
                                          "phrase_classes.json")))
    concepts = sorted(classes, key=lambda x:classes[x], reverse=True)[:2] 
    extract_audio_for_concepts(config["data_path"],
                               config["min_class_size"],
                               concepts)
  elif args.TASK == 3:
    extract_sentence_info(config["data_path"], "train")
    extract_sentence_info(config["data_path"], "val")
    extract_sentence_info(config["data_path"], "test")

    extract_visual_words(config["data_path"], "train_flickr_audio", 50)
    extract_visual_words(config["data_path"], "val_flickr_audio", 50)
    extract_visual_words(config["data_path"], "test_flickr_audio", 50)
  elif args.TASK == 4:
    extract_visual_words(os.path.join(config["data_path"], '../'), "train_flickr_audio/train_flickr_audio", 50)
    extract_visual_words(os.path.join(config["data_path"], '../'), "val_flickr_audio/val_flickr_audio", 50)
    extract_visual_words(os.path.join(config["data_path"], '../'), "test_flickr_audio/test_flickr_audio", 50)
  elif args.TASK == 5:
    extract_zs_item_file_full_data(os.path.join(config["data_path"], '../'))

if __name__ == "__main__":
  argv = sys.argv[1:]
  main(argv)
