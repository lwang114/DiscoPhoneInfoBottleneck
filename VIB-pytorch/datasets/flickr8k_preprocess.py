import json
import os
from scipy.io import wavfile
from nltk.corpus import stopwords
import argparse
import sys
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
  
  phrase_file = os.path.join(data_path, "flickr8k_phrases.json")
  phrase_f = json.load(open(phrase_file))
  class_freq = os.path.join(data_path, "phrase_classes.json")

  word_info_file = os.path.join(data_path, dataset_name, dataset_name+'.json')
  word_f = open(word_info_file, "w") 
  for split in ["train", "val", "test"]:
    dataset = os.path.join(data_path, dataset_name, split)
    if not os.path.exists(dataset):
      os.makedirs(dataset)

    with open(os.path.join(data_path, f"splits/flickr40k_{split}.txt"), "r") as f:
      filenames = [line.rstrip("\n").split("/")[-1] for line in f]
    
    abx_f = open(os.path.join(dataset_path, split+".item"), "w")
    triplets = ["#file_ID onset offset #phone prev-phone next-phone speaker"] 
    prev_id = ""
    word_idx = 0
    n_words = 0
    vocab = set()
    for line in phrase_f:
      if debug and n_words > 500:
        break
      phrase = json.loads(line.rstrip("\n"))
      audio_id = phrase["utterance_id"]
      if audio_id != prev_id:
        word_idx = 0
        prev_id = audio_id
      else:
        word_idx += 1

      audio_file = audio_id + ".wav"
      audio_path = os.path.join(data_path, "flickr_audio/wavs", audio_file)
      label = phrase["label"]
      spk = utt2spk[audio_id]

      if label in stop_words:
        continue
      if audio_file in filenames and (class_freq[label] >= min_class_size):
        vocab.add(label)
        n_words += 1
        fs, audio = wavfile.read(audio_path)
        if len(phrase["children"]): 
          for word in phrase["children"]:
            begin = word['begin']
            end = word["end"]
            begin_frame = int(begin * fs)
            end_frame = int(end * fs)
            audio_word = audio[begin_frame:end_frame+1]
            print(audio_file, audio_word.shape) # XXX
            word_file = f"{audio_id}_{word_idx:04d}.wav"
            word_path = os.path.join(data_path, dataset_name, split, word_file)
            wavfile.write(word_path, fs, audio_word)
            word_info = {"audio_id": audio_id,
                         "word_id": str(word_idx),
                         "label": label,
                         "begin": begin,
                         "end": end,
                         "spk": spk,
                         "split": split,
                         "phonemes": word}
            word_f.write(json.dumps(word_info)+"\n")

            for phn_idx, phone in enumerate(word["children"][1:-1]):
              phn = phone["text"]
              begin_phn = phone["begin"] - begin
              end_phn = phone["end"] - begin
              prev_phn = word["children"][phn_idx-1]
              next_phn = word["children"][phn_idx+1]
              abx_f.write(f"{word_file.split('.')[0]} {begin_phn} {end_phn} {phn} {prev_phn} {next_phn} {spk}")
    abx_f.close()
  print(f"Number of words: {n_words}, vocab size: {len(vocab)}")
  word_f.close() 
     
def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  parser.add_argument("--config", type=str, default="configs/flickr8k.json")
  args = parser.parse_args(argv)
  config = json.load(open(args.config))
  if args.TASK == 0:
    extract_word_dataset(config["data_path"],
                         config["min_freq"],
                         config["debug"])

if __name__ == "__main__":
  argv = sys.argv[1:]
  main(argv)
