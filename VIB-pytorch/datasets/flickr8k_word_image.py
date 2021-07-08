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
      ds_method="average",
      sample_rate=16000,
      min_class_size=50
  ):
    self.preprocessor = preprocessor
    self.splits = splits[split]
    self.data_path = data_path
    self.use_segment = use_segment
    self.ds_method = ds_method
    self.sample_rate = sample_rate
    self.max_feat_len = 100
    self.max_word_len = 100
    self.max_segment_num = 10
    
    data = []
    for sp in self.splits:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, sp,
                                 min_class_size=min_class_size,
                                 audio_feature=audio_feature,
                                 image_feature=image_feature)
      
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
              method='average'):
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
    word_begin = segments['begin']
    for segment in segments["children"]:
      begin = int((segment["begin"]-word_begin)*100)
      end = int((segment["end"]-word_begin)*100)
      if end >= self.max_feat_len:
        break
      segment_feat = embed(feat[begin:end+1], method=method)
      sfeats.append(segment_feat)
    sfeat = torch.stack(sfeats)
    sfeat = fix_embedding_length(sfeat, self.max_segment_num)
    mask = torch.zeros(self.max_segment_num)
    mask[:len(segments["children"])] = 1.
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
    word_begin = segments['begin']
    dur = segments["end"] - segments["begin"]
    nframes = int(dur * 100)
    feat = torch.zeros((nframes, *sfeat.size()[1:]))
    for i, segment in enumerate(segments["children"]):
      begin = int((segment["begin"]-word_begin)*100)
      end = int((segment["end"]-word_begin)*100)
      if i >= sfeat.size(0):
        break
      feat[begin:end+1] = sfeat[i]
    return feat.squeeze(-1)

  def __getitem__(self, idx):
    audio_file, image_file, label, phoneme_dicts, box, box_idx = self.dataset[idx]
    audio_inputs, input_mask = self.load_audio(audio_file)
    if self.use_segment:
      audio_inputs, input_mask = self.segment(audio_inputs.t(), 
                                              phoneme_dicts,
                                              method=self.ds_method)
      audio_inputs = audio_inputs.t()
    phonemes = [phn_dict["text"] for phn_dict in phoneme_dicts]
    image_inputs = self.load_image(image_file, box, box_idx)
    word_labels = self.preprocessor.to_index(label).squeeze(0)
    phone_labels = self.preprocessor.to_phone_index(phonemes)

    word_mask = torch.zeros((1, self.max_feat_len, self.max_feat_len))
    for t in range(input_mask.sum()):
      word_mask[0, t, t] = 1. 
    phone_mask = torch.zeros(self.max_phone_num,)
    phone_mask[:len(phoneme)] = 1.

    return audio_inputs, image_inputs, word_labels, phone_labels, input_mask, word_mask, phone_mask 

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
    sample_rate=16000,
    min_class_size=50
  ):
    self.num_features = num_features
    self.min_class_size= min_class_size
    self.wordsep = " "
    self._prepend_wordsep = prepend_wordsep

    metadata_file = os.path.join(data_path, f"flickr8k_word_{min_class_size}.json")
    
    data = []
    for _, spl in splits.items(): 
      for sp in spl:
        data.extend(load_data_split(data_path, sp,
                                    audio_feature=audio_feature,
                                    image_feature=image_feature,
                                    min_class_size=self.min_class_size)) 
    tokens = set()
    phone_tokens = set()
    for ex in data:
      tokens.add(ex["text"])
      for phn in ex["phonemes"]:
        phone_tokens.add(phn["text"])
    self.tokens = sorted(tokens)
    self.phone_tokens = sorted(phone_tokens)
    self.tokens_to_index = {t:i for i, t in enumerate(self.tokens)}
    self.phone_tokens_to_index = {t:i for i, t in enumerate(self.phone_tokens)}
    print(f"Number of classes: {self.num_tokens}")
    print(f"Number of phone classes: {self.num_phone_tokens}")
    
  @property
  def num_tokens(self):
    return len(self.tokens)

  @property
  def num_phone_tokens(self):
    return len(self.phone_tokens)

  def to_index(self, line):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in line.split(self.wordsep)])

  def to_phone_index(self, sent):
    tok_to_idx = self.phone_tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in sent])

def load_data_split(data_path, split,
                    audio_feature="mfcc",
                    image_feature="rcnn",
                    min_class_size=50,
                    max_keep_size=400):
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

    example = {"audio": audio_path,
               "image": image_path,
               "text": label,
               "phonemes": word["phonemes"],
               "box": box,
               "box_idx": box_idx} 
    examples.append(example)
  word_f.close()
  return examples  
   
if __name__ == "__main__":
    preproc = FlickrWordImagePreprocessor(num_features=80, data_path="/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/")
