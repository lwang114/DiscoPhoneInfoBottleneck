import sys
import os
import json
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
def extract_visual_words(data_path, split, visual_word_file):
  """
  Parameters:
  ----------
    data_path : str, path to the LibriSpeech root
        transcription format: {dir1-dir2-sent_id} {word 1 in cap} ... {word last in cap}  
    split : str, {train-clean-100, train-clean-369, train-other-500}
    visual_word_file : str, filename storing the dictionary of visual words 

  Returns:
  -------
    librispeech_{split}.json : str, storing in each line dict in the form
        {"audio_id": str, 
         "text": str, transcript of the audio, 
         "labels": str, visual words of the audio}
  """
  lemmatizer = WordNetLemmatizer()
  visual_words = json.load(open(visual_word_file))
  dr1s = os.listdir(os.path.join(data_path, split))
  out_file = os.path.join(data_path, f"{split}_visual.json")
  out_f = open(out_file, "w")
  n_sents = 0
  n_words = 0
  vocab = set()
  for dr1 in dr1s:
    dr1_path = os.path.join(data_path, split, dr1)
    for dr2 in os.listdir(dr1_path):
      dr2_path = os.path.join(data_path, split, dr1, dr2)
      trans_file = os.path.join(dr2_path, f"{dr1}-{dr2}.trans.txt")
      with open(trans_file, "r") as trans_f:
        for line in trans_f:
          sent_dict = dict()
          parts = line.rstrip('\n').split()
          audio_id = parts[0]
          labels = []
          for w in parts[1:]:
            lemma = lemmatizer.lemmatize(w.lower())
            if lemma in visual_words and not lemma in stop_words:
              if visual_words[lemma] > 50:
                vocab.add(lemma)
                labels.append(lemma)
          
          if len(labels) > 0:
            print(audio_id, labels)
            n_sents += 1
            n_words += len(labels)
            sent_dict = {"audio_id": audio_id,
                         "text": parts[1:],
                         "labels": labels}
            out_f.write(json.dumps(sent_dict)+"\n")
  out_f.close()
  print(f"Visual vocabulary size: {len(vocab)}")
  print(f"Number of visual words: {n_words}")
  print(f"Number of sentences with visual words: {n_sents}")

# def extract_forced_alignment(): # TODO


def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  parser.add_argument("--data_path", default="/ws/ifp-53_2/hasegawa/lwang114/data/LibriSpeech/")
  parser.add_argument("--split", default="train-clean-100")

  args = parser.parse_args(argv)
  if args.TASK == 0:
    visual_word_file = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/phrase_classes.json"
    extract_visual_words(args.data_path, args.split, visual_word_file)

if __name__ == "__main__":
  args = sys.argv[1:]
  main(args) 
