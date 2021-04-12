import torch
import torchaudio
import torchvision
# import nltk
# from nltk.stem import WordNetLemmatizer
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.structured_prediction
import numpy as np
import re
import os
import json
from tqdm import tqdm
from itertools import combinations
from copy import deepcopy
# dep_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
# dep_parser._model = dep_parser._model.cuda()
# lemmatizer = WordNetLemmatizer()
UNK = "###UNK###"
NULL = "###NULL###"

def log_normalize(x):
    x.add_(1e-6).log_()
    mean = x.mean()
    std = x.std()
    return x.sub_(mean).div_(std + 1e-6)

def fix_embedding_length(emb, L):
  size = emb.size()[1:]
  if emb.size(0) < L:
    pad = [torch.zeros(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb

class FlickrSegmentImageDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, 
      preprocessor, split,
      splits = {
        "train": ["train"],
        "validation": ["test"],
        "test": ["test"],           
      },
      augment = True,
      sample_rate = 16000
  ):
    self.preprocessor = preprocessor
    self.splits = splits
    self.data_path = data_path
    self.sample_rate = sample_rate
    self.max_feat_len = 64
    if split == "train":
        self.max_class_size = 200
    elif split == "test":
        self.max_class_size = 50
    
    data = []
    for sp in self.splits[split]:
      # Load data paths to audio and visual features
      if self.preprocessor.balance_strategy:
        examples = load_data_split_balanced(data_path, split,
                                            balance_strategy=self.preprocessor.balance_strategy,
                                            max_class_size=self.max_class_size)
      else:
        examples = load_data_split(data_path, split)
      data.extend(examples)    
      print("Number of {} audio files = {}".format(split, len(examples)))

    # Set up transforms
    self.transforms = [
        torchaudio.transforms.MelSpectrogram(
          sample_rate=sample_rate, win_length=sample_rate * 25 // 1000,
          n_mels=preprocessor.num_features,
          hop_length=sample_rate * 10 // 1000,
        ),
        torchvision.transforms.Lambda(log_normalize),
    ]
  
    if augment:
        augmentation = [
                torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                torchaudio.transforms.TimeMasking(100, iid_masks=True),
                torchaudio.transforms.TimeMasking(100, iid_masks=True),
            ]
    self.transforms = torchvision.transforms.Compose(self.transforms)

    # Load each image-caption pairs
    audio = [example["audio"] for example in data]
    text = [example["text"] for example in data]
    duration = [example["duration"] for example in data]
    interval = [example["interval"] for example in data]
    image_ids = [example["image_id"] for example in data]
    feat_idxs = [example["feat_idx"] for example in data]
    self.dataset = list(zip(audio, text, duration, interval, image_ids, feat_idxs))

    # Create gold unit file
    if not os.path.exists(os.path.join(data_path, "flickr8k_segment_image_gold_units.json")) or not os.path.exists(os.path.join(data_path, "flickr8k_segment_image_abx_triplets.item")):
      if self.preprocessor.balance_strategy:
        create_gold_file_balanced(data_path, sample_rate,
                                  balance_strategy=self.preprocessor.balance_strategy,
                                  max_class_size=self.max_class_size)
      else:
        create_gold_file(data_path, sample_rate)
    self.gold_dicts = json.load(open(os.path.join(data_path, "flickr8k_segment_image_gold_units.json")))
    self.image_feats = np.load(os.path.join(data_path, "flickr8k_res34_finetuned.npz")) # XXX np.load(os.path.join(data_path, "flickr8k_res34.npz"))
    
  def sample_sizes(self):
    """
    Returns a list of tuples containing the input size
    (time, 1) and the output length for each sample.
    """
    return [((duration, 1), len(text)) for _, text, duration in self.dataset]

  def __getitem__(self, idx):
    audio_file, label, dur, interval, image_id, feat_idx = self.dataset[idx]
    begin = int(interval[0] * self.sample_rate)
    end = int(interval[1] * self.sample_rate)
    audio, _ = torchaudio.load(audio_file)

    try:
      inputs = self.transforms(audio[:, begin:end]).squeeze(0)
    except:
      inputs = self.transforms(audio)[:, :, int(begin // 160):int(end // 160)].squeeze(0)
    image_feat = self.image_feats[image_id][feat_idx]
    image_inputs = torch.FloatTensor(image_feat)

    nframes = inputs.size(-1)
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.

    inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t()
    outputs = self.preprocessor.to_index(label).squeeze(0)
    
    return inputs, image_inputs, outputs, input_mask 

  def __len__(self):
    return len(self.dataset)


class FlickrSegmentImagePreprocessor:
  """
  A preprocessor for the Flickr 8k dataset. Assume the existence of five files/directories:
      flickr_labels.txt : contains phone alignments of the format (in sec)
                          align_[#audio_file_id_1].txt 
                          #phone_1 onset_1 offset_1
                          ...
                          #phone_n onset_n offset_n
                          align_[#audio_file_id_2].txt 
                          ...  
      word_segmentation/ : contains word alignment files, each of the format (in sec)
                           #word onset offset

      flickr8k_phrases.txt : contains phrase info of the format
                          [#image_filename] [caption_id] #entity id #phrase onset offset
  
      flickr8k_bboxes.txt : contains bbox info of the format
                          [#image_filename] [caption_id] xmin ymin xmax ymax

      flickr30k_sentences.txt : contains captions of the format
                          #audio_filename caption_id #caption 

  Args:
      data_path (str) : Path to the top level data directory.
      num_features (int) : Number of audio features in transform.
  """
  def __init__(
    self,
    data_path,
    num_features,
    splits = {
        "train": ["train"],
        "validation": ["test"],
        "test": ["test"]
    },
    tokens_path=None,
    lexicon_path=None,
    use_words=False,
    prepend_wordsep=False,
    sample_rate=16000,
    balance_strategy="truncate"
  ):
    self.balance_strategy = balance_strategy
    self.wordsep = " "
    self._prepend_wordsep = prepend_wordsep
    self.num_features = num_features

    metadata_file = os.path.join(data_path, "flickr8k_phrases.json")
    if not os.path.exists(metadata_file):
      self.extract_phrase_info(data_path)
    
    data = []
    for _, spl in splits.items(): 
      for sp in spl:
          if sp == "train":
              self.max_class_size = 200
              self.min_class_size = 80
          elif sp == "test":
              self.max_class_size = 50
              self.min_class_size = 80
          if balance_strategy:
              data.extend(load_data_split_balanced(data_path, sp,
                                                   balance_strategy=balance_strategy,
                                                   max_class_size=self.max_class_size,
                                                   min_class_size=self.min_class_size))
          else:
              data.extend(load_data_split(data_path, sp))
   
    tokens = set()
    lexicon = {}
    for ex in data:
      tokens.add(ex["text"])
    self.tokens = sorted(tokens)
    self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}
    print(f"Number of types: {self.num_tokens}")
  
  @property
  def num_tokens(self):
    return len(self.tokens)

  def to_index(self, line):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in line.split(self.wordsep)])

  def union_of_boxes(self, b1, b2):
    return [min(b1[0], b2[0]),\
            min(b1[1], b2[1]),\
            max(b1[2], b2[2]),\
            max(b1[3], b2[3])]

  def extract_phrase_info(self, data_path):
    word_dir = os.path.join(os.path.join(data_path, "word_segmentation"))
    phone_f = open(os.path.join(data_path, "flickr_labels.txt"), "r")
    phrase_f = open(os.path.join(data_path, "flickr8k_phrases.txt"), "r")
    bbox_f = open(os.path.join(data_path, "flickr8k_bboxes.txt"), "r")
    sent_f = open(os.path.join(data_path, "flickr8k_sentences.txt"), "r")
    out_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "w")

    # Build a mapping form file id to caption text
    id_to_text = {} 
    for line in sent_f: 
      parts = line.strip("\n").split()
      image_id = parts[0].split(".")[0] 
      capt_id = parts[1]
      utterance_id = f"{image_id}_{int(capt_id)-1}"
      text = parts[2:]
      id_to_text[utterance_id] = text

    # Build a mapping from file id to phrase info
    id_to_phrase = dict()
    id_to_bbox = dict()
    for line_bbox in bbox_f:
      parts = line_bbox.split()
      entity_id = parts[1]
      bbox = line_bbox.strip("\n").split()[-4:]
      if entity_id in id_to_bbox:
        id_to_bbox[entity_id] = self.union_of_boxes(id_to_bbox[entity_id],
                                                    [int(x) for x in bbox])
      else:
        id_to_bbox[entity_id] = [int(x) for x in bbox]

    for line_phrase in phrase_f:
      parts = line_phrase.strip("\n").split()
      image_id = parts[0].split(".")[0]
      capt_id = parts[1] 
      entity_id = parts[2]
      utterance_id = f"{image_id}_{int(capt_id)-1}"
      phrase = " ".join(parts[3:-1])
      if not entity_id in id_to_bbox:
        continue

      begin = int(parts[-1]) - 1
      end = begin + len(parts[3:-1]) - 1
      if not utterance_id in id_to_phrase:
        id_to_phrase[utterance_id] = []

      bbox = line_bbox.strip("\n").split()[-4:]
      phrase_info = {"utterance_id": utterance_id,
                     "text": phrase, 
                     "begin": begin, 
                     "end": end,
                     "bbox": id_to_bbox[entity_id],
                     "entity_id": entity_id,
                     "feat_idx": len(id_to_phrase[utterance_id])}
      id_to_phrase[utterance_id].append(phrase_info)
    print(f"Number of audio: {len(id_to_phrase)}")

    # Iterate over each speech file to extract utterance info
    cur_utterance_id = None
    cur_phones = None
    cur_words = None
    idx = 0
    for line in tqdm(phone_f):
      if "align" in line:
        if cur_utterance_id and cur_utterance_id in id_to_phrase:
          utt = Utterance(cur_phones, cur_words,
                          id_to_phrase[cur_utterance_id],
                          id_to_text[cur_utterance_id])
          for phrase in utt.phrases:
            out_f.write(json.dumps(phrase)+"\n")
        cur_utterance_id = "_".join(line.strip("\n").split(".")[0].split("_")[1:])
        cur_phones = []
        cur_words = []
         
        word_fn = os.path.join(word_dir, cur_utterance_id+".words")
        if not os.path.exists(word_fn):
          continue
        with open(word_fn, "r") as word_f:
          for line in word_f:
            w, begin, end = line.split()
            if "$" in w:
              continue
            w = re.sub(r"[^\w]", "", w)
            w = re.sub(r"[0-9]", "", w)
            cur_words.append({"text": w, 
                              "begin": float(begin), 
                              "end": float(end)})
      else:
        phn, begin, end = line.split()
        cur_phones.append({"text": phn, 
                           "begin": float(begin), 
                           "end": float(end)})
      
    utt = Utterance(cur_phones, cur_words,
                    id_to_phrase[cur_utterance_id],
                    id_to_text[cur_utterance_id])
    for phrase in utt.phrases:
      out_f.write(json.dumps(phrase)+"\n")
    phone_f.close()
    phrase_f.close()
    sent_f.close()
    
class Utterance:
  def __init__(self, phones, words, phrases, rawtext):
    """
    Args:
        phones : a list of dicts of {"text": str, "begin": begin time in sec, "end": end time in sec}
        words : a list of dicts of {"text": str, "begin": begin time in sec, "end": end time in sec}
        phrases : a list of dicts of {"text": str, "begin": begin time in word tokens, "end": end time in word tokens}
        rawtext : a list of strs
    """
    phones = sorted(phones, key=lambda x:x['begin'])
    words = sorted(words, key=lambda x:x['begin'])
    phrases = sorted(phrases, key=lambda x:x['begin'])

    words, phrases = self.extract_char_offsets(words, phrases, rawtext)
    words = self.align_time(phones, words)
    phrases = self.extract_phrase_labels(phrases)
    self.phrases = self.match(words, phrases)
    
  def extract_char_offsets(self, words, phrases, rawtext):
    # Extract char offsets for text words
    text_offsets = []
    begin = 0
    for token in rawtext:
      text_offsets.append([begin, begin+len(token)-1])
      begin += len(token)
    
    # Convert token offsets to char offsets
    for idx in range(len(phrases)):
      begin_char = text_offsets[phrases[idx]["begin"]][0]
      end_char = text_offsets[phrases[idx]["end"]][1]
      phrases[idx]["begin"] = begin_char
      phrases[idx]["end"] = end_char 

    # Extract char offsets for acoustic words
    begin = 0
    for word in words:
      word["begin_char"] = begin
      word["end_char"] = begin+len(word["text"])-1
      begin += len(word["text"])
    
    return words, phrases

  def match(self, children, parents):
    for parent in parents:
        if not "children" in parent:
            parent["children"] = []

        for child in children:
            if lemmatizer.lemmatize(child["text"].lower()) == parent["label"]:
                parent["children"].append(child)
    return parents

  def align_char(self, children, parents):
    parent_idx = 0
    n_parents = len(parents)

    for child in children:
      parent = parents[parent_idx]
      if not "children" in parent:
        parent["children"] = []
      begin, end = child["begin_char"], child["end_char"]

      if begin > parent["end"]:
        parent_idx += 1
        if parent_idx >= n_parents:
          break
        parent = parents[parent_idx] 
        parent["children"] = []
      
      if end < parent["begin"]:
        continue
      parent["children"].append(child)
    return parents  

  def align_time(self, children, parents):
    parent_idx = 0
    n_parents = len(parents)

    for child in children:
      parent = parents[parent_idx]
      if not "children" in parent:
        parent["children"] = []
      begin, end = child["begin"], child["end"]

      if begin >= parent["end"]:
        parent_idx += 1
        if parent_idx >= n_parents:
          break
        parent = parents[parent_idx] 
        parent["children"] = []
      
      if end <= parent["begin"]:
        continue
      parent["children"].append(child)
    return parents

  def extract_phrase_labels(self, phrases):
    for phrase in phrases:
      text = phrase["text"].split()
      pos_tags = [t[1] for t in nltk.pos_tag(text, tagset="universal")]
      instance = dep_parser._dataset_reader.text_to_instance(text, pos_tags)
      parsed_text = dep_parser.predict_batch_instance([instance])[0]
      head_idx = np.where(np.asarray(parsed_text["predicted_heads"]) <= 0)[0][0]
      phrase["label"] = lemmatizer.lemmatize(text[head_idx])
    return phrases

def load_data_split(data_path, split):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "text" : a list of tokenized words for the class name,
            "full_text" : a list of tokenized words for the whole phrase, 
            "duration" : float,
            "interval": [begin of the word in ms, end of the word in ms],
            "image_id": str,
            "feat_idx": int, image feature idx
          }
  """
  with open(os.path.join(data_path, "splits/flickr40k_{}.txt".format(split)), "r") as f:
    filenames = [line.rstrip("\n").split("/")[-1] for line in f]

  image_feats = np.load(os.path.join(data_path, "flickr8k_res34_finetuned.npz")) # XXX
  utt_to_feat = {'_'.join(k.split('_')[:-1]):k for k in image_feats}
  
  examples = []

  class_freqs = json.load(open(os.path.join(data_path, "phrase_classes.json"), "r"))
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  for line in phrase_f:
    # if len(examples) > 800: # XXX
    #     break
    phrase = json.loads(line.rstrip("\n"))
    utterance_id = phrase["utterance_id"]
    fn = utterance_id + ".wav"
    filename = os.path.join(data_path, "flickr_audio/wavs", fn)

    if fn in filenames and "children" in phrase:
        if len(phrase["children"]) > 0 and class_freqs[phrase["label"]] >= 20: # Filter out low-frequency words
            example = {"audio": filename,
                       "text": phrase["label"],
                       "full_text": phrase["text"],
                       "duration": phrase["children"][-1]["end"] - phrase["children"][0]["begin"],
                       "interval": [phrase["children"][0]["begin"], phrase["children"][-1]["end"]],
                       "feat_idx": phrase["feat_idx"],
                       "image_id": utt_to_feat["_".join(phrase["utterance_id"].split("_")[:-1])]}
            examples.append(example)

  phrase_f.close()
  return examples

def load_data_split_balanced(data_path, split, balance_strategy="truncate", max_class_size=200, min_class_size=500):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "text" : a list of tokenized words for the class name,
            "full_text" : a list of tokenized words for the whole phrase, 
            "duration" : float,
            "interval": [begin of the word in ms, end of the word in ms],
            "image_id": str,
            "feat_idx": int, image feature idx
          }
  """
  with open(os.path.join(data_path, "splits/flickr40k_{}.txt".format(split)), "r") as f:
    filenames = [line.rstrip("\n").split("/")[-1] for line in f]

  image_feats = np.load(os.path.join(data_path, "flickr8k_res34_finetuned.npz")) # XXX
  utt_to_feat = {'_'.join(k.split('_')[:-1]):k for k in image_feats}
  
  examples = []

  class_freqs = json.load(open(os.path.join(data_path, "phrase_classes.json"), "r"))
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  class_counts = {c:0 for c in class_freqs}
  class_to_example = {c:[] for c in class_freqs}
  
  for line in phrase_f:
    # if len(examples) > 800: # XXX
    #     break
    phrase = json.loads(line.rstrip("\n"))
    utterance_id = phrase["utterance_id"]
    fn = utterance_id + ".wav"
    filename = os.path.join(data_path, "flickr_audio/wavs", fn)
    label = phrase["label"]
    
    if fn in filenames and "children" in phrase:
        if len(phrase["children"]) > 0 and class_freqs[label] >= min_class_size and class_counts[label] < max_class_size: # Filter out low-frequency words
            class_counts[label] += 1
            image_id = utt_to_feat["_".join(phrase["utterance_id"].split("_")[:-1])]
            feat_idx = phrase["feat_idx"]
            
            example = {"audio": filename,
                       "text": phrase["label"],
                       "full_text": phrase["text"],
                       "duration": phrase["children"][-1]["end"] - phrase["children"][0]["begin"],
                       "interval": [phrase["children"][0]["begin"], phrase["children"][-1]["end"]],
                       "feat_idx": feat_idx,
                       "image_id": image_id}
            class_to_example[label].append(example)
            examples.append(example)

  # Augment the dataset by reverse mismatch the audio and image of the same type
  '''
  if balance_strategy == "augment":
      for c in class_to_example:
          if len(class_to_example[c]) > 1 and class_counts[c] < max_class_size:
              n_augment_examples = max_class_size - class_counts[c] 
              example_list = class_to_example[c]
              num_speech = len(example_list)
              speech_idxs = [speech_idx for speech_idx in range(num_speech) for _ in range(1, num_speech)]
              image_idxs = [image_idx for speech_idx in range(num_speech) for image_idx in range(num_speech-1, -1, -1) if image_idx != speech_idx]
              assert len(speech_idxs) == len(image_idxs)
              
              for speech_idx, image_idx in zip(speech_idxs[:n_augment_examples], image_idxs[:n_augment_examples]):
                  example = deepcopy(example_list[speech_idx])
                  example["image_id"] = example_list[image_idx]["image_id"]
                  example["feat_idx"] = example_list[image_idx]["feat_idx"]
                  # print(example_list[speech_idx]["audio"], example_list[image_idx]["image_id"]) # XXX
                  examples.append(example)
  '''
  phrase_f.close()
  return examples
  
def create_gold_file(data_path, sample_rate):
  """
  Create the following files:
      gold_units.json : contains gold_dicts, a list of mappings 
          {"sentence_id" : str,
           "units" : a list of ints representing phoneme id for each feature frame,
           "text" : a list of strs representing phoneme tokens for each feature frame}
     abx_triplets.item : contains ABX triplets in the format
                         line 0 : whatever (not read)
                         line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
                         onset : begining of the triplet (in s)
                         offset : end of the triplet (in s)
  """
  class_freqs = json.load(open(os.path.join(data_path, "phrase_classes.json"), "r"))
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  filenames = [line.rstrip("\n").split("/")[-1] for line in open(os.path.join(data_path, "splits/flickr40k_test.txt"), "r")] 
  phone_to_index = {}
  gold_dicts = []
  triplets = ['#file_ID onset offset #phone prev-phone next-phone speaker']

  phrase_idx = 0
  for line in phrase_f:
    phrase = json.loads(line.rstrip("\n"))
    utterance_id = phrase["utterance_id"]
    fn = utterance_id + ".wav"
    filename = os.path.join(data_path, "flickr_audio/wavs", fn)
    if fn in filenames and "children" in phrase:
      if len(phrase["children"]) == 0 and class_freqs[phrase["label"]] < 20: # Filter out low-frequency words
          continue
      label = phrase["label"]
      nframes = int((phrase["children"][-1]["end"] - phrase["children"][0]["begin"])*100)+1

      gold_dict = {"sentence_id": fn,
                   "units": [-1]*nframes,
                   "phoneme_text": [NULL]*nframes,
                   "word_text": [label]*nframes,
                   "word_full_text": [NULL]*nframes
      }
      
      begin_phone = 0
      begin_word = 0
      example_id = f"{fn}_{phrase_idx}"
      phrase_idx += 1
      for word in phrase["children"]:
        dur_word = int((word["end"] - word["begin"])*100) 
        end_word = begin_word + dur_word
        for x in range(begin_word, end_word):
            gold_dict["word_full_text"][x] = word["text"]
        begin_word += dur_word
            
        for phn_idx, phone in enumerate(word["children"]):
          if "SIL" in phone["text"] or "+" in phone["text"]:
              continue
          if not phone["text"] in phone_to_index:
            phone_to_index[phone["text"]] = len(phone_to_index)
          dur_phone = int((phone["end"] - phone["begin"])*100)
          end_phone = begin_phone + dur_phone
          for t in range(begin_phone, end_phone):
              gold_dict["phoneme_text"][t] = phone["text"]
              gold_dict["units"][t] = phone_to_index[phone["text"]]
          
          if phn_idx == 0:
            prev_token = NULL
          else:
            prev_token = word["children"][phn_idx-1]["text"]

          if phn_idx == len(word["children"]) - 1:
            next_token = NULL
          else:
            next_token = word["children"][phn_idx+1]["text"]

          if len(gold_dicts) < 5000:
              triplets.append(f"{example_id} {begin_phone} {begin_phone + dur_phone} {phone['text']} {prev_token} {next_token} 0")

          begin_phone += dur_phone

      gold_dicts.append(gold_dict)
  
  with open(os.path.join(data_path, "gold_units.json"), "w") as gold_f:
    json.dump(gold_dicts, gold_f, indent=2)

  with open(os.path.join(data_path, "abx_triplets.item"), "w") as triplet_f:
    triplet_f.write('\n'.join(triplets))

def create_gold_file_balanced(data_path, sample_rate, balance_strategy="truncate", max_class_size=100, min_class_size=80):
  """
  Create the following files:
      gold_units.json : contains gold_dicts, a list of mappings 
          {"sentence_id" : str,
           "units" : a list of ints representing phoneme id for each feature frame,
           "text" : a list of strs representing phoneme tokens for each feature frame}
     abx_triplets.item : contains ABX triplets in the format
                         line 0 : whatever (not read)
                         line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
                         onset : begining of the triplet (in s)
                         offset : end of the triplet (in s)
  """
  class_freqs = json.load(open(os.path.join(data_path, "phrase_classes.json"), "r"))
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  filenames = [line.rstrip("\n").split("/")[-1] for line in open(os.path.join(data_path, "splits/flickr40k_test.txt"), "r")] 
  phone_to_index = {}
  class_counts = {c:0 for c in class_freqs}
  gold_dicts = []
  triplets = ['#file_ID onset offset #phone prev-phone next-phone speaker']

  phrase_idx = 0
  for line in phrase_f:
    phrase = json.loads(line.rstrip("\n"))
    utterance_id = phrase["utterance_id"]
    fn = utterance_id + ".wav"
    filename = os.path.join(data_path, "flickr_audio/wavs", fn)
    label = phrase["label"]

    if fn in filenames and "children" in phrase:
      if len(phrase["children"]) == 0 or class_freqs[label] < min_class_size or class_counts[label] >= max_class_size: # Filter out low-frequency words
          continue

      class_counts[label] += 1
      nframes = int((phrase["children"][-1]["end"] - phrase["children"][0]["begin"])*100)+1

      gold_dict = {"sentence_id": fn,
                   "units": [-1]*nframes,
                   "phoneme_text": [NULL]*nframes,
                   "word_text": [label]*nframes,
                   "word_full_text": [NULL]*nframes
      }
      
      begin_phone = 0
      begin_word = 0
      example_id = f"{fn}_{phrase_idx}"
      phrase_idx += 1
      for word in phrase["children"]:
        dur_word = int((word["end"] - word["begin"])*100) 
        end_word = begin_word + dur_word
        for x in range(begin_word, end_word):
            gold_dict["word_full_text"][x] = word["text"]
        begin_word += dur_word
            
        for phn_idx, phone in enumerate(word["children"]):
          if "SIL" in phone["text"] or "+" in phone["text"]:
              continue
          if not phone["text"] in phone_to_index:
            phone_to_index[phone["text"]] = len(phone_to_index)
          dur_phone = int((phone["end"] - phone["begin"])*100)
          end_phone = begin_phone + dur_phone
          for t in range(begin_phone, end_phone):
              gold_dict["phoneme_text"][t] = phone["text"]
              gold_dict["units"][t] = phone_to_index[phone["text"]]
          
          if phn_idx == 0:
            prev_token = NULL
          else:
            prev_token = word["children"][phn_idx-1]["text"]

          if phn_idx == len(word["children"]) - 1:
            next_token = NULL
          else:
            next_token = word["children"][phn_idx+1]["text"]

          if len(gold_dicts) < 5000:
              triplets.append(f"{example_id} {begin_phone} {begin_phone + dur_phone} {phone['text']} {prev_token} {next_token} 0")

          begin_phone += dur_phone

      gold_dicts.append(gold_dict)
  
  with open(os.path.join(data_path, "flickr8k_segment_image_gold_units.json"), "w") as gold_f:
    json.dump(gold_dicts, gold_f, indent=2)

  with open(os.path.join(data_path, "flickr8k_segment_image_abx_triplets.item"), "w") as triplet_f:
    triplet_f.write('\n'.join(triplets))

    
if __name__ == "__main__":
    preproc = FlickrSegmentImagePreprocessor(num_features=80, data_path="/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/")
