import torch
import torchaudio
import torchvision
from torchvision import transforms
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
from PIL import Image
from scipy import signal 
# dep_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
# dep_parser._model = dep_parser._model.cuda()
# lemmatizer = WordNetLemmatizer()
UNK = "###UNK###"
NULL = "###NULL###"
BLANK = "###BLANK###"
IGNORED_TOKENS = ["SIL", "GARBAGE", "+BREATH+", "+LAUGH+", "+NOISE+"]  

def log_normalize(x):
    x.add_(1e-6).log_()
    mean = x.mean()
    std = x.std()
    return x.sub_(mean).div_(std + 1e-6)

def fix_embedding_length(emb, L, padding=0):
  size = emb.size()[1:]
  if emb.size(0) < L:
    if padding == 0:
      pad = torch.zeros((L-emb.size(0),)+size, dtype=emb.dtype)
    else:
      pad = padding*torch.ones((L-emb.size(0),)+size, dtype=emb.dtype) 
    emb = torch.cat([emb, pad], dim=0)
  else:
    emb = emb[:L]
  return emb

def embed(feat, method='average'):
  if method == 'average':
    return feat.mean(0)
  elif method == 'resample':
    new_feat = signal.resample(feat.detach().numpy(), 4)
    return torch.FloatTensor(new_feat.flatten())

class FlickrWordImageDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, 
      preprocessor, split,
      splits = {
        "train": ["train", "val"],
        "validation": ["val"],
        "test": ["test"],           
      },
      augment=False,
      use_segment=False,
      audio_feature="mfcc",
      image_feature="image",
      phone_label="multilingual",
      ds_method="average",
      sample_rate=16000,
      min_class_size=50,
      debug=False  
  ):
    self.preprocessor = preprocessor
    self.splits = splits[split]
    self.data_path = data_path
    self.use_segment = use_segment
    self.ds_method = ds_method
    self.sample_rate = sample_rate
    self.max_feat_len = 100
    self.max_word_len = 100
    self.max_phone_num = 50 
    self.max_segment_num = 5 # XXX
    self.max_segment_len = 10
    self.debug = debug
    
    data = []
    for sp in self.splits:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, sp,
                                 min_class_size=min_class_size,
                                 audio_feature=audio_feature,
                                 image_feature=image_feature,
                                 phone_label=phone_label,
                                 debug=debug)
      
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
        self.audio_transforms.extend(augmentation)
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
    boxes = [example["box"] for example in data]
    text = [example["text"] for example in data]
    phonemes = [example["phonemes"] for example in data]
    feat_idxs = [example["box_idx"] for example in data]
    self.dataset = list(zip(audio, image, text, phonemes, boxes, feat_idxs))

    self.image_feature_type = image_feature 
    self.image_feats = np.load(os.path.join(data_path,
                                            f"../flickr8k_{image_feature}.npz")) # XXX np.load(os.path.join(data_path, "flickr8k_res34.npz"))
    self.image_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in self.image_feats}
    self.audio_feature_type = audio_feature

  def load_audio(self, audio_file):
    if self.audio_feature_type == "mfcc":
      audio, _ = torchaudio.load(audio_file)
      try:
        inputs = self.audio_transforms(audio[:, begin:end]).squeeze(0)
      except:
        inputs = self.audio_transforms(audio)
        inputs = inputs.squeeze(0)
    elif self.audio_feature_type == "cpc":
      audio = np.loadtxt(audio_file)
      inputs = torch.FloatTensor(audio).t()
    else: Exception(f"Audio feature type {self.audio_feature_type} not supported")

    nframes = inputs.size(-1)
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.
    inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t()
    return inputs, input_mask

  def load_image(self, image_file, box, box_idx):
    if self.image_feature_type == "image":
      image = Image.open(image_file).convert("RGB")
      if len(np.asarray(image).shape) == 2:
        print(f"Gray scale image {image_file}, convert to RGB".format(image_file))
        image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
      region = image.crop(box=box)
      region = self.image_transforms(region)
      return region
    else: 
      image_id = os.path.splitext(os.path.split(image_file)[1])[0]
      image_feat = self.image_feats[self.image_to_feat[image_id]]
      region_feat = image_feat[box_idx]
      return region_feat

  def segment(self, feat, segments, 
              method="average"):
    """ 
      Args:
        feat : (num. of frames, feature dim.)
        segments : a list of dicts of phoneme boundaries
      
      Returns:
        sfeat : (max num. of segments, feature dim.)
        mask : (max num. of segments,)
    """
    feat = feat
    sfeats = []
    if method == "no-op":
      mask = torch.zeros((self.max_segment_num, self.max_segment_len))
    else:
      mask = torch.zeros(self.max_segment_num) 
      mask[:len(segments)] = 1.

    word_begin = segments[0]["begin"]
    for i, segment in enumerate(segments):
      begin = int((segment["begin"]-word_begin)*100)
      end = int((segment["end"]-word_begin)*100)
      dur = end - begin + 1
      if (begin >= self.max_feat_len) or (i >= self.max_segment_num):
        break
      if method == "no-op":
        segment_feat = fix_embedding_length(feat[begin:end+1], self.max_segment_len)
        mask[i, :dur] = 1. 
      else:      
        segment_feat = embed(feat[begin:end+1], method=method)
      sfeats.append(segment_feat)
      
    sfeat = torch.stack(sfeats)
    sfeat = fix_embedding_length(sfeat, self.max_segment_num)
    return sfeat, mask
    
  def unsegment(self, sfeat, segments):
    """
      Args:
        sfeat : (num. of segments, feature dim.)
        segments : a list of dicts of phoneme boundaries
      
      Returns:
        feat : (num. of frames, feature dim.) 
    """
    if sfeat.ndim == 1:
      sfeat = sfeat.unsqueeze(-1)
    word_begin = segments[0]['begin']
    dur = segments[-1]["end"] - segments[0]["begin"]
    nframes = int(dur * 100)
    feat = torch.zeros((nframes, *sfeat.size()[1:]))
    for i, segment in enumerate(segments):
      begin = int((segment["begin"]-word_begin)*100)
      end = int((segment["end"]-word_begin)*100)
      if i >= sfeat.size(0):
        break
      feat[begin:end+1] = sfeat[i]
    return feat.squeeze(-1)

  def __getitem__(self, idx):
    audio_file, image_file, label, phoneme_dicts, box, box_idx = self.dataset[idx]
    audio_inputs, input_mask = self.load_audio(audio_file)
    audio_inputs = audio_inputs.t()
    if self.use_segment:
      audio_inputs, input_mask = self.segment(audio_inputs, 
                                              phoneme_dicts,
                                              method=self.ds_method)
        
    phonemes = [phn_dict["text"] for phn_dict in phoneme_dicts]
    image_inputs = self.load_image(image_file, box, box_idx)
    word_labels = self.preprocessor.to_word_index([label])

    phone_labels = self.preprocessor.to_index(phonemes)
    phone_labels = fix_embedding_length(phone_labels,
                                        self.max_phone_num,
                                        padding=self.preprocessor.ignore_index) 

    if self.use_segment:
      word_mask = torch.zeros((1, self.max_segment_num, self.max_segment_num))
    else:
      word_mask = torch.zeros((1, self.max_feat_len, self.max_feat_len))

    for t in range(len(phoneme_dicts)):
      if t >= self.max_segment_num:
        break
      word_mask[0, t, t] = 1. 
    phone_mask = torch.zeros(self.max_phone_num,)
    phone_mask[:len(phonemes)] = 1.

    return audio_inputs,\
           phone_labels,\
           word_labels,\
           input_mask,\
           phone_mask,\
           word_mask,\
           image_inputs

  def __len__(self):
    return len(self.dataset)


class FlickrWordImagePreprocessor:
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
        "validation": ["val"],
        "test": ["test"]
    },
    tokens_path=None,
    lexicon_path=None,
    use_words=False,
    prepend_wordsep=False,
    audio_feature="mfcc",
    image_feature="rcnn",
    phone_label="multilingual",
    sample_rate=16000,
    min_class_size=50,
    ignore_index=-100,
    use_blank=True,
    debug=False,      
  ):
    self.num_features = num_features
    self.ignore_index = ignore_index
    self.min_class_size = min_class_size
    self.use_blank = use_blank
    self.wordsep = " "
    self._prepend_wordsep = prepend_wordsep

    metadata_file = os.path.join(data_path, f"flickr8k_word_{min_class_size}.json")
    
    data = []
    for _, spl in splits.items(): 
      for sp in spl:
        data.extend(load_data_split(data_path, sp,
                                    audio_feature=audio_feature,
                                    image_feature=image_feature,
                                    phone_label=phone_label,
                                    min_class_size=self.min_class_size,
                                    debug=debug)) 
    visual_words = set()
    tokens = set()
    for ex in data:
      visual_words.add(ex["text"])
      for phn in ex["phonemes"]:
        tokens.add(phn["text"])

    self.tokens = sorted(tokens)
    self.visual_words = sorted(visual_words)
    if self.use_blank:
      self.tokens = [BLANK]+self.tokens
      self.visual_words = [BLANK]+self.visual_words
    self.tokens_to_index = {t:i for i, t in enumerate(self.tokens)}
    self.words_to_index = {t:i for i, t in enumerate(self.visual_words)}

    print(f"Number of phone classes: {self.num_tokens}")
    print(f"Number of visual word classes: {self.num_visual_words}")
    
  @property
  def num_tokens(self):
    return len(self.tokens)

  @property
  def num_visual_words(self):
    return len(self.visual_words)

  def to_index(self, sent):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in sent])

  def to_word_index(self, sent):
    tok_to_idx = self.words_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in sent])

  def to_text(self, indices):
    text = []
    for t, i in enumerate(indices):
      if (i == 0) and (t != 0):
        prev_token = text[t-1]
        text.append(prev_token)
      else:
        text.append(self.tokens[i])
    return text

  def to_word_text(self, indices):
    return [self.visual_words[i] for i in indices]

  def tokens_to_word_text(self, indices):
    T = len(indices)
    path = [self.visual_words[i] for i in indices]
    sent = []
    for i in range(T):
      if path[i] == BLANK:
        continue
      elif (i != 0) and (path[i] == path[i-1]):
        continue
      else:
        sent.append(path[i])
    return sent

  def tokens_to_text(self, indices): 
    T = len(indices)
    path = self.to_text(indices)
    sent = []
    for i in range(T):
      if path[i] == BLANK:
        continue
      elif (i != 0) and (path[i] == path[i-1]):
        continue 
      else:
        sent.append(path[i])
    return sent


def load_data_split(data_path, split,
                    audio_feature="mfcc",
                    image_feature="rcnn",
                    phone_label="multilingual",
                    min_class_size=50,
                    max_keep_size=400,
                    debug=False):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "image" : filename of image,
            "text" : a list of tokenized words for the class name,
            "box" : 4-tuple,
            "feat_idx" : int, image feature idx
          }
  """
 
  if image_feature.split('_')[-1] == 'label':
    image_feature = 'rcnn'

  image_feats = np.load(os.path.join(data_path, f"../flickr8k_{image_feature}.npz"))
  utt_to_feat = {'_'.join(k.split('_')[:-1]):k for k in image_feats}
  
  examples = []
  word_f = open(os.path.join(data_path,
                             f"flickr8k_word_{min_class_size}.json"), "r")
  label_counts = dict()
  for line in word_f:
    # if debug and len(examples) >= 20: # XXX
    #   break
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
    word_id = int(word['word_id'])
    audio_path = None
    if audio_feature == "mfcc":
      audio_file = f"{audio_id}_{word_id:04d}.wav"
      audio_path = os.path.join(data_path, split, audio_file)
    elif audio_feature == "cpc":
      audio_file = f"{audio_id}_{word_id:04d}.txt"
      audio_path = os.path.join(data_path, f"../flickr8k_word_{min_class_size}_cpc", audio_file)
    else: Exception(f"Audio feature type {audio_feature} not supported")

    image_id = "_".join(audio_id.split("_")[:-1])
    image_path = os.path.join(data_path, "Flicker8k_Dataset", image_id+".jpg") 
    box = word["box"]
    box_idx = word["box_id"]
    phonemes = []
    if phone_label == "groundtruth":
        phonemes = word["phonemes"]["children"]
        noisy = False
        for phn in phonemes:
          if phn["text"] in IGNORED_TOKENS or (phn["text"][0] == "+"):
            noisy = True
            break
        if noisy:
          continue
    elif phone_label == "multilingual":
        phonemes = [{'text': BLANK,
                     'begin': phone_info["begin"],
                     'end': phone_info["end"]} for phone_info in word["phonemes"]["children"]]
        for phn_idx, phn in enumerate(word["multilingual_phones"]):
          phonemes[phn_idx]['text'] = phn 
    else:
        raise ValueError(f"Invalid phone label type: {phone_label}")
        
    example = {"audio": audio_path,
               "image": image_path,
               "text": label,
               "phonemes": phonemes,
               "box": box,
               "box_idx": box_idx} 
    examples.append(example)
  word_f.close()
  return examples  

   
if __name__ == "__main__":
    preproc = FlickrWordImagePreprocessor(num_features=80, data_path="/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/")
