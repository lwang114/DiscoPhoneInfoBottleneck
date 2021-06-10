import torch
import torchaudio
import torchvision
from torchvision import transforms
import numpy as np
import re

UNK = "###UNK###"
NULL = "###NULL###"
BLANK = "###BLANK###"

def log_normalize(x):
  x.add_(1e-6).log_()
  mean = x.mean()
  std = x.std()
  return x.sub_(mean).div_(std + 1e-6)

def fix_embedding_length(emb, L, padding=0):
  size = emb.size()[1:]
  if emb.size(0) < L:
    if padding == 0:
      pad = [torch.zeros(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]
    else:
      pad = [padding*torch.ones(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]  
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb


class LibrispeechDataset(torch.utils.data.Dataset):
  
  
  def __init__(
      self, data_path,
      preprocessor, split,
      splits = {
        "train": ["train-clean"],
        "test": ["dev-clean"]    
      },
      augment=False,
      audio_feature="mfcc",
      image_feature="image",
      sample_rate=16000,
  ):
    self.preprocessor = preprocessor
    self.splits = splits[split]
    self.data_path = data_path
    
    for sp in self.splits:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, sp,
                                 min_class_size=min_class_size,
                                 audio_feature=audio_feature,
                                 image_feature=image_feature)
      data.extend(examples)
      print(f"Number of {split} audio files = {len(examples)}")

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
    ''' TODO
    self.image_transforms = transforms.Compose(
                [transforms.Scale(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), 
                                      (0.229, 0.224, 0.225))]
                )
    '''
    # Load each image-caption pairs
    audio = [example["audio"] for example in data]
    visual_words = [example["visual_words"] for example in data]
    phonemes = [example["phonemes"] for example in data]
    self.dataset = list(zip(audio, visual_words, phonemes))
    self.audio_feature_type = audio_feature
    self.max_feat_len = 1024
    self.max_phone_num = 200
    self.max_word_num = 10

  def load_audio(self, audio_file):
    inputs = self.audio_transforms(audio_file)
    inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t() 
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.
    return inputs, input_mask
  
  def __getitem__(self, idx): 
    audio_file, visual_words, phonemes = self.dataset[idx]
    audio_inputs, input_mask = self.load_audio(audio_file)
    sent = [phn["text"] for phn in phonemes]
    visual_sent = [w["text"] for w in visual_words]
    phoneme_labels = self.preprocessor.to_index(sent)
    word_labels = self.preprocessor.to_word_index(visual_sent)
    phoneme_labels = fix_embedding_length(phoneme_labels,
                                          self.max_phone_num,
                                          padding=-100)
    word_labels = fix_embedding_length(word_labels, 
                                       self.max_word_num,
                                       padding=-100)

    phone_mask = torch.zeros(self.max_phone_num)
    n_phones = len(sent)
    phone_mask[:n_phones] = 1.

    word_mask = torch.zeros(self.max_word_num, self.max_feat_len)
    for i, w in enumerate(visual_words):
      begin_frame = int(w['begin']*100)
      end_frame = int(w['end']*100)
      word_mask[i, begin_frame:end_frame] = 1.

    return audio_inputs, 
           phoneme_labels, 
           word_labels, 
           input_mask,
           phone_mask,
           word_mask 
  
  def __len__(self):
    return len(self.dataset)
 

class LibrispeechPreprocessor:
  
  
  def __init__(
    self,
    data_path,
    num_features,
    splits = {
        "train": ["train-clean"],
        "test": ["dev-clean"]
    },
    tokens_path=None,
    lexicon_path=None,
    use_words=False,
    prepend_wordsep=False,
    audio_feature="mfcc",
    image_feature="rcnn",
    sample_rate=16000,
    ignore_index=-100
  ):
    self.num_features = num_features
    

    data = []
    for spl in splits:
      for sp in splits[spl]:
        data.extend(load_data_split(data_path, sp,
                                    audio_feature=audio_feature,
                                    image_feature=image_feature))
    tokens = set()
    visual_words = set()
    for ex in data:
      sent = [phn["text"] for phn in ex["phonemes"]]
      visual_sent = [w["text"] for w in ex["visual_words"]]
      tokens.add(sent)
      visual_words.add(visual_sent)
    self.tokens = [BLANK]+sorted(tokens)
    self.visual_words = sorted(visual_words)
    self.tokens_to_index = {t:i for i, t in enumerate(self.tokens)}
    self.word_to_index = {w:i for i, w in enumerate(self.visual_words)}

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
    return torch.LongTensor([self.word_to_idx[label] for t in sent])
  
  def to_text(self, indices):
    return [self.tokens[i] for i in indices] 

  def to_word_text(self, indices):
    return [self.visual_words[i] for i in indices]

  def tokens_to_text(self, indices): 
    T = len(indices)
    path = self.to_text(indices)
    sent = []
    for i in enumerate(T):
      if path[i] == BLANK:
        continue
      elif (i != 0) and (path[i] == path[i-1]):
        continue 
      else:
        sent.append(path[i])
    return sent
                                    
def load_data_split(data_path, sp,
                    audio_feature="mfcc",
                    image_feature="rcnn"):
  """
  Returns: 
      examples : a list of mappings of
          { "audio" : filename of audio,
            "visual_words" : a list of dicts for visual words in each utterance as
                { "text" : str,
                  "begin" : float,
                  "end" : float}
            "phonemes" : a list of dicts for phonemes in each utterance as
                { "text" : str,
                  "begin" : float,
                  "end" : float}
          }
  """ 
  label_f = open(os.path.join(data_path, split, f"{split}.json"), "r") 
  for line in label_f:
    label_dict = json.loads(line.rstrip("\n"))
    utt_id = label_dict["utterance_id"]
    visual_words = label_dict["visual_words"]
    phonemes = [phn for w in label_dict["words"] for phn in w["phonemes"]]
    audio_path = os.path.join(data_path, f"{utt_id}.wav")
    
    example = {"audio": audio_path,
               "visual_words": visual_words,
               "phonemes": phonemes}
    examples.append(example)
  label_f.close()
  return examples
