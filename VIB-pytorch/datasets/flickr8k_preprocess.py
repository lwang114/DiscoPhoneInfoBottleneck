import json
import os
from scipy.io import wavfile
from nltk.corpus import stopwords
import argparse
import sys
import shutil
stop_words = stopwords.words("english")

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
        print(audio_id)
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
        if (phn_idx == 0) or (phn_idx == len(phonemes) - 1):
          continue
        phn = phone["text"]
        if phn[0] == "+":
          continue
        begin_phn = round(phone["begin"] - begin, 3)
        end_phn = round(phone["end"] - begin, 3) 
        prev_phn = phonemes[phn_idx-1]["text"]
        next_phn = phonemes[phn_idx+1]["text"]
        abx_f.write(f"{audio_file_id} {begin_phn} {end_phn} {phn} {prev_phn} {next_phn} {spk}\n")
    abx_f.close()
  word_f.close()

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
        shutil.copyfile(os.path.join(data_path, "flickr_audio/wav", audio_file),
                        os.path.join(cur_dir, audio_file))

  for c in concepts:
    segment_file = os.path.join(data_path, out_dir, c, "word_boundary.json")
    concept_info_str = "\n".join([json.dumps(c_info) for c_info in concept_info[c]])
    with open(segment_file, "w") as f:
      f.write(concept_info_str)
              
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
                                          "flickr8k_phrases.json")))
    concepts = sorted(classes, key=lambda x:classes[x], reverse=True)[2] 
    extract_audio_for_concepts(config["data_path"],
                               config["min_class_size"],
                               concepts)

if __name__ == "__main__":
  argv = sys.argv[1:]
  main(argv)
