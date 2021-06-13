import sys
import os
import json
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
special_symbols = ["ʰ", "ʼ"]
def extract_visual_words(data_path, split, visual_word_file):
  """
  Args:
    data_path : str, path to the LibriSpeech root
    split : str, {train-clean-100, train-clean-369, train-other-500}
    visual_word_file : str, filename storing the dictionary of visual words 

  Returns:
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
  out_f.close()
  print(f"Visual vocabulary size: {len(vocab)}")
  print(f"Number of visual words: {n_words}")
  print(f"Number of sentences with visual words: {n_sents}")

def extract_pseudo_phones(data_path, split, pseudo_phone_file):
  """
  Args:
    data_path : str, path to the LibriSpeech root
    split : str  
    pseudo_phone_file : str, path to the pseudo phone transcripts in the format
      "utts" : {
        {utt_id} : {
          "output" : {"rec_token" : str, recognized tokens separated by spaces,
                      "rec_token_id" : str, list of ints separated by spaces}        
        }
      }
  
  Returns:
    {split}/{split}_with_pseudo_phones.json : str, storing in each line dict containing
      {"audio_id" : str,
       "text" : str, transcript of the audio,
       "pseudo_phones" : str, pseudo phone sequence of the audio}
  """
  in_file = os.path.join(data_path, split, f"{split}.json")
  out_file = os.path.join(data_path, split, f"{split}_with_pseudo_phones.json")
  phone_info = json.load(open(pseudo_phone_file))["utts"]
  in_f = open(in_file, "r")
  out_f = open(out_file, "w")
  
  for line in in_f:
    sent_dict = json.loads(line.rstrip("\n"))
    utt_id = sent_dict["audio_id"]
    phone_tokens = phone_info[utt_id]["output"]["rec_token"]
    

     
  

def main(argv):
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("TASK", type=int)
  parser.add_argument("--data_path", default="/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic")
  parser.add_argument("--split", default="train-clean-100")

  args = parser.parse_args(argv)
  if args.TASK == 0:
    visual_word_file = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/phrase_classes.json"
    extract_visual_words(args.data_path, args.split, visual_word_file)

if __name__ == "__main__":
  args = sys.argv[1:]
  main(args) 
