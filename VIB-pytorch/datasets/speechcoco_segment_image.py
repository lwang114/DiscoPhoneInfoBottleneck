import torch
import torchaudio
import torchvision
from torchvision import transforms
import numpy as np
import os
import json
from copy import deepcopy
from PIL import Image

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

class SpeechCOCODataset(torch.utils.data.Dataset):


  def __init__(
      self, data_path,
      preprocessor, split,
      splits={
        "train": ["train"],
        "test": ["val"]
        },
      sample_rate=16000,
      augment=True,
      image_feature="res34"
      ):
    self.preprocessor = preprocessor
    self.splits = splits
    self.data_path = data_path
    self.sample_rate = sample_rate
    self.max_feat_len = 256
    self.max_region_num = 10
    # self.min_word_freq = 500

    data = []
    for sp in self.splits[split]:
      examples = load_data_split(data_path, sp) # TODO Filter out rare words
      data.extend(examples)
      print("Number of {} audio files = {}".format(split, len(examples)))
    
    # Set up transforms
    self.audio_transforms = [
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
      self.audio_transforms = torchvision.transforms.Compose(self.audio_transforms)
    self.image_transforms = transforms.Compose(
          [transforms.Scale(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))]
        ) 

    # Load each image-caption pairs 
    audio = [example["audio"] for example in data]
    image = [example["image"] for example in data]
    boxes = [example["boxes"] for example in data]
    labels = [example["labels"] for example in data]
    intervals = [example["intervals"] for example in data]
    box_idxs = [example["box_idxs"] for example in data]
    self.dataset = list(zip(audio, image, labels, intervals, boxes, box_idxs))

    # Create gold unit file
    gold_file = os.path.join(data_path, "speechcoco_gold_units.json")
    if not os.path.exists(gold_file):
      create_gold_file(data_path, sample_rate)
    self.gold_dicts = json.load(open(gold_file))
   
    self.image_feats = None
    self.image_to_feat = None
    if image_feature.split("_")[-1] != "label": 
      self.image_feats = np.load(os.path.join(data_path,
                                              f"mscoco_{split}_{image_feature}.npz")) 
      self.image_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in self.image_feats}

  def load_audio(self, audio_file, intervals):
    raw_audio, _ = torchaudio.load(audio_file)
    feats = self.audio_transforms(raw_audio)
    
    segments = []
    for interval in intervals:
      begin = int(interval[0] // 10)
      end = int(interval[1] // 10)
      segment = feats[:, :, begin:end]
      segments.append(segment.squeeze(0))
    inputs = torch.cat(segments, dim=-1)
    nframes = inputs.size(-1)
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.
    
    inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t()
    return inputs, input_mask

  def load_image(self, image_file, boxes, box_idxs):
    if self.image_feats is not None:
      image_id = image_file.split('/')[-1].split('.')[0]
      feat_id = self.image_to_feat[image_id]
      image_feat = torch.FloatTensor(self.image_feats[feat_id])
      regions = torch.cat([image_feat[box_idx] for box_idx in box_idxs])
    else:
      image = Image.open(image_file).convert("RGB")
      if len(np.asarray(image).shape) == 2:
        print(f"Gray scale image {image_file}, convert to RGB".format(image_file))
        image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
      
      for box_idx, box in enumerate(boxes):
        if (box[2] <= box[0]) or (box[3] <= box[1]):
          print(f"Bad box: {image_file} {box}")
          box[2] = max(box[0] + 1, box[2])
          box[3] = max(box[1] + 1, box[3])
        boxes[box_idx] = deepcopy(box)
      
      regions = [self.image_transforms(image.crop(box=box)) for box in boxes]
      regions = torch.stack(regions)
    
    regions = fix_embedding_length(regions, self.max_region_num)
    region_mask = torch.zeros(self.max_region_num)
    region_mask[:len(boxes)] = 1
    return regions, region_mask

  def __getitem__(self, idx):
    audio_file, image_file, labels, intervals, boxes, box_idxs = self.dataset[idx]
    audio_feats, audio_mask = self.load_audio(audio_file, intervals)
    region_feats, region_mask = self.load_image(image_file, boxes, box_idxs)
    outputs = self.preprocessor.to_index(labels)
    outputs = fix_embedding_length(outputs, self.max_region_num)
    return audio_feats, region_feats, outputs, audio_mask, region_mask

  def __len__(self):
    return len(self.dataset)

class SpeechCOCOPreprocessor:


  def __init__(
    self,
    data_path,
    num_features,
    splits={
      "train": ["train"],
      "test": ["val"]
      },
    tokens_path=None,
    lexicon_path=None,
    image_feature="res34",
    sample_rate=16000
    ):
    self.num_features = num_features

    for sp in splits:
      for split in splits[sp]:
        metadata_file = os.path.join(data_path, f"{split}2014/speechcoco_{split}.json")
        if not os.path.exists(metadata_file):
          extract_sentence_info(data_path, split)

    tokens = set()
    for _, spl in splits.items():
      for sp in spl:
        data = load_data_split(data_path, sp)
        for ex in data:
          tokens.update(ex["labels"])
    
    self.tokens = sorted(tokens)
    self.tokens_to_index = {t:i for i, t in enumerate(self.tokens)}

  @property
  def num_tokens(self):
    return len(self.tokens)

  def to_index(self, line):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in line])
    
def extract_sentence_info(data_path, split):
  """
  Assume the existence of the following files:
    {split}2014/mscoco_{split}_word_segments.txt: in the format
       {audio_id} {label} {begin} {end}

    {split}2014/mscoco_{split}_phone_info.json (for val only):
        "audio_id" : str,
        "word" : str,
        "begin" : float,
        "end" : float,
        "phonemes" : list of dicts with items
          "begin" : float,
          "end" : float,
          "text" : str
    {split}2014/mscoco_{split}_bboxes.txt: in the format
        {image_id} {label} {x} {y} {width} {height}
          
  Returns:
      sentences : containing the following items:
        "audio_id" : str, 
        "image_id" : str,
        "text": list of strs, e.g., ["a", "little", "girl"],
        "labels" : list of strs, e.g., ["girl"], 
        "boxes" : list of 4-tuples, e.g., [[8, 152, 108, 340]] in [xmin, ymin, xmax, ymax] format, 
        "box_idxs": list of ints,
        "audio" : a list of dicts with items
            "text" : str,
            "begin" : float,
            "end" : float,
            "phonemes" : a list of dicts with items 
              "text" : int, e.g., "G", 
              "begin" : int, e.g., 1.166, 
              "end" : int, e.g., 1.256
  """
  word_info_file = os.path.join(
                     data_path, 
                     f'{split}2014/mscoco_{split}_word_segments.txt'
                   )
  box_info_file = os.path.join(
                    data_path,
                    f'{split}2014/mscoco_{split}_bboxes.txt'
                  )
  phone_info_file = os.path.join(
                      data_path,
                      f'{split}2014/mscoco_{split}_phone_info.json'
                    )
  sentence_info_file = os.path.join(
                         data_path,
                         f'{split}2014/speechcoco_{split}.json'
                       )
  word_f = open(word_info_file, 'r') 
  words = dict()
  for line in word_f:
    audio_id, label = line.split()[:2]
    image_id = audio_id.split('_')[0]
    interval = [float(t) for t in line.split()[2:]]
    if not image_id in words:
      # if len(words) >= 100: # XXX
      #   break
      words[image_id] = dict()
    
    if not audio_id in words[image_id]:
      words[image_id][audio_id] = [{'label': label,
                                    'interval': interval}]  
    else:
      words[image_id][audio_id].append({'label': label,
                                        'interval': interval})

  box_f = open(box_info_file, 'r')
  boxes = dict() 
  for line in box_f:
    fn, label = line.split()[:2]
    image_id = str(int(fn.split('_')[-1]))
    box = [int(x) for x in line.split()[2:]]
    x, y, w, h = box
    if not image_id in boxes:
      if not image_id in words:
        continue
      if len(boxes) == len(words): # XXX
        break

      boxes[image_id] = [{'image_id': fn,
                          'label': label,
                          'box': [x, y, x+w, y+h],
                          'box_idx': 0}]
    else:
      boxes[image_id].append({'image_id': fn,
                              'label': label,
                              'box': [x, y, x+w, y+h],
                              'box_idx': len(boxes[image_id])}) 
  
  phones = dict()
  if os.path.exists(phone_info_file):
    phone_f = open(phone_info_file, 'r')
    for line in phone_f:
      phone_info = json.loads(line.rstrip('\n'))
      audio_id = phone_info['audio_id'] 
      image_id = audio_id.split('_')[0]
      interval = (phone_info['begin'], phone_info['end'])
      
      if not image_id in phones:
        if not image_id in words:
          continue
        if len(phones) == len(words): # XXX
          break
        phones[image_id] = dict()
      
      phns = [{'begin': phn['begin'],
               'end': phn['end'],
               'text': phn['value']} for phn in phone_info['phonemes']]
      if not audio_id in phones[image_id]:
        phones[image_id][audio_id] = {interval: phns}
      else:
        phones[image_id][audio_id][interval] = phns

  word_f = open(word_info_file, 'r')
  sent_f = open(sentence_info_file, 'w')
  sent_info = dict()
  for ex, image_id in enumerate(sorted(words)):
    box_info = boxes[image_id]

    for audio_id in sorted(words[image_id]):
      print(ex, audio_id)
      image_id = audio_id.split('_')[0]
      word_info = words[image_id][audio_id]
      if len(phones):
        phns = phones[image_id][audio_id]
        phns_info = [phns[tuple(w['interval'])] for w in word_info]
      else:
        phns_info = [[] for _ in word_info]

      sent_info = {'audio_id': audio_id,
                   'image_id': box_info[0]['image_id'],
                   'text': [w['label'] for w in word_info],
                   'labels': [b['label'] for b in box_info],
                   'boxes': [b['box'] for b in box_info],
                   'box_idxs': [b['box_idx'] for b in box_info],
                   'audio': [
                        {'begin': w['interval'][0],
                         'end': w['interval'][1],
                         'text': w['label'],
                         'phonemes': phn_info}
                        for w, phn_info in zip(word_info, phns_info)
                        ]
                  }      
      sent_f.write(json.dumps(sent_info)+'\n')
  sent_f.write(json.dumps(sent_info))
  word_f.close()
  sent_f.close()

def load_data_split(data_path, split):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "image" : filename of image, 
            "labels" : a list of tokenized words for the class name,
            "intervals": list of [begin of the word in ms, end of the word in ms],
            "boxes": list of 4-tuples,
            "box_idxs": list of ints, image feature indices,
          }
  """
  sent_f = open(os.path.join(data_path, f"{split}2014/speechcoco_{split}.json"), "r")
  examples = []
  idx = 1
  for line in sent_f:
    #if idx > 600: # XXX
    #  break
    idx += 1
    sent = json.loads(line.rstrip("\n"))
    audio_id = sent["audio_id"]
    image_id = sent["image_id"]
    audio_file = audio_id + ".wav"
    audio_path = os.path.join(data_path, f"{split}2014/wav/", audio_file)
    image_file = image_id + ".jpg"
    image_path = os.path.join(data_path, f"{split}2014/imgs/{split}2014/", image_file)

    labels = sent["labels"]
    intervals = [[word["begin"], word["end"]] for word in sent["audio"]] 
    example = {"audio": audio_path,
               "image": image_path,
               "labels": labels,
               "intervals": intervals,
               "boxes": sent["boxes"],
               "box_idxs": sent["box_idxs"]} 
    examples.append(example) 
  sent_f.close()
  return examples

def create_gold_file(data_path, sample_rate):
  """
  Create the following files:
      gold_units.json : contains a list of dicts
        {"sentence_id" : str,
         "units" : a list of ints,
         "text" : a list of strs}
      abx_triplets.item : contains ABX triplets in the format
                          line 0 : whatever (not read)
                          line > 0 : #file_ID onset offset #phone prev-phone next-phone speaker
                          onset : begining of the triplet (in s)
                          offset : end of the triplet (in s) 
  """
  sent_f = open(os.path.join(data_path, "val2014/speechcoco_val.json"), "r")
  phone_to_index = dict()
  gold_dicts = []
  triplets = ['#file_ID onset offset #phone prev-phone next-phone speaker']

  sent_idx = 0
  for line in sent_f:
    sent = json.loads(line.rstrip("\n"))
    audio_id = sent["audio_id"]
    fn = audio_id+".wav"
    nframes = int((sent["audio"][-1]["end"] - sent["audio"][0]["begin"]) // 10)+1

    gold_dict = {
                 "sentence_id": fn,
                 "units": [-1]*nframes,
                 "word_full_text": [NULL]*nframes,
                 "phoneme_text": [NULL]*nframes
                 }

    begin_phone = 0
    begin_word = 0
    example_id = f"{fn}_{sent_idx}"
    sent_idx += 1
    for word in sent["audio"]:
      dur_word = int((word["end"] - word["begin"]) // 10)
      
      end_word = begin_word + dur_word
      for x in range(begin_word, end_word):
        gold_dict["word_full_text"][x] = word["text"]
      begin_word += dur_word

      for phn_idx, phone in enumerate(word["phonemes"]):
        if "__" in phone["text"]:
          continue
        if not phone["text"] in phone_to_index:
          phone_to_index[phone["text"]] = len(phone_to_index)
        
        if not phone["text"] in phone_to_index:
          phone_to_index[phone["text"]] = len(phone_to_index)
        dur_phone = int((phone["end"] - phone["begin"]) // 10)
        end_phone = begin_phone + dur_phone
        for t in range(begin_phone, end_phone):
            gold_dict["phoneme_text"][t] = phone["text"]
            gold_dict["units"][t] = phone_to_index[phone["text"]]
        
        if phn_idx == 0:
          prev_token = NULL
        else:
          prev_token = word["phonemes"][phn_idx-1]["text"]

        if phn_idx == len(word["phonemes"]) - 1:
          next_token = NULL
        else:
          next_token = word["phonemes"][phn_idx+1]["text"]

        triplets.append(f"{example_id} {begin_phone} {begin_phone + dur_phone} {phone['text']} {prev_token} {next_token} 0")
        
        begin_phone += dur_phone
      gold_dicts.append(gold_dict)
  
  with open(os.path.join(data_path, "speechcoco_gold_units.json"), "w") as gold_f:
    json.dump(gold_dicts, gold_f, indent=2)

  with open(os.path.join(data_path, "speechcoco_abx_triplets.item"), "w") as triplet_f:
    triplet_f.write('\n'.join(triplets))
  triplet_f.close()

if __name__ == "__main__":
  data_path = "/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/"
  preprocessor = SpeechCOCOPreprocessor(data_path, num_features=80) 
  dataset = SpeechCOCODataset(data_path, preprocessor, "train")
