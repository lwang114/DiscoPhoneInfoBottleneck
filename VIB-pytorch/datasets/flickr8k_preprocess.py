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
IGNORED_TOKENS = ["SIL", "GARBAGE", "ʰ", "ʼ", "ˈ"] 


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

def extract_sentence_info(data_path, out_path, split):
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
  
  sent_f = open(out_path, "w")
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
  if audio_id in audio_ids:
    example = match_words_with_phones(words, phones)
    example["utterance_id"] = os.path.join(data_path, f"flickr_audio/wavs/{audio_id}")

  sent_f.write(json.dumps(example)+"\n") 
  phone_f.close()
  sent_f.close()


def extract_visual_words(sentence_info_path,
                         phrase_info_path,
                         visual_class_path,
                         out_path,
                         min_class_size):
  """
  Add the following keys :
    "visual_words" : a list of visual words idxs
    "bboxes" : a list of tuple of (left, upper, right, lower)
  """
  lemmatizer = WordNetLemmatizer()
  visual_classes = json.load(open(visual_class_path))
  word_f = open(phrase_info_path, "r")
  in_f = open(sentence_info_path, "r")
  out_f = open(out_path, "w")
  visual_words = dict()
      
  for line in in_f:
    sent = json.loads(line.rstrip("\n"))
    sent["visual_words"] = []
    sent["bboxes"] = []
    audio_id = sent["utterance_id"].split("/")[-1]
    print(audio_id)
    for w_idx, word in enumerate(sent["words"]):
      label = lemmatizer.lemmatize(word['text'].lower())
      if label in visual_classes:
        if visual_classes[label] >= min_class_size:
          print(word['text'], label)
          sent["visual_words"].append(w_idx)
          sent["bboxes"].append([])
    out_f.write(json.dumps(sent)+"\n")
  word_f.close()
  in_f.close()
  out_f.close()

def extract_speaker_info(sentence_info_path, utt2spk_path, out_path):
  utt2spk = dict()
  with open(utt2spk_path, "r") as utt_f,\
       open(sentence_info_path, "r") as sent_f,\
       open(out_path, "w") as out_f:
    for line in utt_f:
      utt_id, spk = line.rstrip("\n").split()
      utt2spk[utt_id.split(".")[0]] = spk

    for line in sent_f:
      sent_dict = json.loads(line.rstrip("\n"))
      utt_id = sent_dict["utterance_id"].split("/")[-1]
      spk = utt2spk[utt_id]
      sent_dict["speaker"] = spk
      out_f.write(json.dumps(sent_dict)+"\n")

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
    audio_id = '_'.join(sent_dict["utterance_id"].split('/')[-1].split('_')[1:])
    spk = sent_dict["speaker"]
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
 
def copy_subset(subset_file, trg_dir):
  subset_f = open(subset_file, "r")
  for line in subset_f:
    src_info = json.loads(line.rstrip("\n"))
    src_file = src_info["utterance_id"] + ".wav"
    trg_file = os.path.join(trg_dir, os.path.basename(src_file))
    shutil.copyfile(src_file, trg_file)
  subset_f.close()

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
    old_data_root = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k" 
    new_data_root = os.path.join(config["data_path"], "../")
    visual_class_path = os.path.join(old_data_root, "phrase_classes.json")
    phrase_info_path = os.path.join(old_data_root, "flickr8k_phrases.json")
    for split in ["train", "val", "test"]: 
      sentence_info_path = os.path.join(new_data_root, f"{split}_flickr_audio/{split}_flickr_audio.json")
      out_path = os.path.join(new_data_root, f"{split}_flickr_audio/{split}_flickr_audio_with_visual_words.json")
      extract_sentence_info(old_data_root, 
                            out_path, 
                            split)
      extract_visual_words(sentence_info_path, 
                           phrase_info_path,
                           visual_class_path,
                           out_path, 50)
  elif args.TASK == 4:
    old_data_root = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k" 
    new_data_root = os.path.join(config["data_path"], "../zerospeech2021-dataset/phonetic")
    visual_class_path = os.path.join(old_data_root, "phrase_classes.json")
    phrase_info_path = os.path.join(old_data_root, "flickr8k_phrases.json")
    for split in ["train_flickr_audio", "val_flickr_audio", "test_flickr_audio"]:
      sentence_info_path = os.path.join(new_data_root, f"{split}/{split}.json")
      out_path = os.path.join(new_data_root, f"{split}/{split}_with_visual_words.json")
      extract_visual_words(sentence_info_path, 
                           phrase_info_path,
                           visual_class_path,
                           out_path, 50)
  elif args.TASK == 5:
    extract_zs_item_file_full_data(os.path.join(config["data_path"], '../'))
  elif args.TASK == 6: 
    new_data_root = os.path.join(config["data_path"], "../")
    utt2spk_path = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr_audio/wav2spk.txt" 
    for split in ["train_flickr_audio", "val_flickr_audio", "test_flickr_audio"]:
      sentence_info_path = os.path.join(new_data_root, f"{split}/{split}.json")
      out_path = os.path.join(new_data_root, f"{split}/{split}_with_spk_info.json")
      extract_speaker_info(sentence_info_path, 
                           utt2spk_path,
                           out_path)
  elif args.TASK == 7:
    new_data_root = os.path.join(config["data_path"], "../")
    for split in ["train_flickr_audio", "val_flickr_audio", "test_flickr_audio"]:
      pseudo_phone_file = f"/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/exp/train_pytorch_train_li10/decode_flickr/{split}_decode_li10/data.json"
      extract_pseudo_phones(new_data_root, split, pseudo_phone_file)
  elif args.TASK == 8:
    subset_file = "/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/test_flickr_audio/test_flickr_audio.json"
    trg_dir = "/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/test_flickr_audio"
    copy_subset(subset_file, trg_dir)

if __name__ == "__main__":
  argv = sys.argv[1:]
  main(argv)
