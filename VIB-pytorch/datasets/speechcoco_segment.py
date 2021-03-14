import json
import os
import torch
import torchaudio
import torchvision
import collections
import numpy as np

UNK = '###UNK###'
NULL = 'NONE'
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

class SpeechCOCOSegmentDataset(torch.utils.data.Dataset):
  def __init__(
    self, data_path,
    preprocessor, split,
    splits = {
      "train": ["train"],
      "validation": ["val"],
      "test": ["val"]
      },
    augment=True,
    sample_rate = 16000 
  ):
    self.splits = splits
    self.data_path = data_path
    self.sample_rate = sample_rate
    self.max_feat_len = 100

    data = []
    for sp in self.splits[split]:
      data.extend(load_data_split(data_path, sp, preprocessor.wordsep, self.sample_rate))

    self.preprocessor = preprocessor

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
        # self.transforms.extend(augmentation)

    self.transforms = torchvision.transforms.Compose(self.transforms)

   # Load each audio file 
    audio = [example['audio'] for example in data]
    text = [example['text'] for example in data]
    duration = [(example['interval'][1] - example['interval'][0]) // 10 for example in data]
    interval = [example['interval'] for example in data]
    self.dataset = list(zip(audio, text, duration, interval))    
    # Create gold unit file
    if not os.path.exists(os.path.join(data_path, "gold_units.json")) or not os.path.exists(os.path.join(data_path, "abx_triplets.item")):
      create_gold_file(data_path, sample_rate) 
    self.gold_dicts = json.load(open(os.path.join(data_path, "gold_units.json")))
  
  def sample_sizes(self):
    """
    Returns a list of tuples containing the input size
    (time, 1) and the output length for each sample.
    """
    return [((duration, 1), len(text)) for _, text, duration in self.dataset]

  def __getitem__(self, index):
    audio_file, text, dur, interval = self.dataset[index]
    begin = int(interval[0] * (self.sample_rate // 1000))
    end = int(interval[1] * (self.sample_rate // 1000))
    audio, _ = torchaudio.load(audio_file)
    try:
        inputs = self.transforms(audio[:, begin:end]).squeeze(0)
    except:
        inputs = self.transforms(audio)[:, :, int(begin // 10):int(end // 10)].squeeze(0)
    nframes = inputs.size(-1)
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.
    inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t()
    outputs = self.preprocessor.to_index(text).squeeze(0)
    
    return inputs, outputs, input_mask

  def __len__(self):
    return len(self.dataset)
 
class SpeechCOCOSegmentPreprocessor:
  """
  A preprocessor for the SpeechCOCO segment dataset.
  Args:
      data_path (str) : Path to the top level data directory.
      num_features (int) : Number of audio features in transform.
      tokens_path (str) (optional) : The path to the .txt file of list of model output
          tokens. If not provided the token set is built dynamically from the vocab of the tokenized text.
      lexicon_path (str) (optional) : The path to the .txt file of a mapping of words to tokens. If 
          provided the preprocessor will split the text into words and 
          map them to the corresponding token. 
  """
  def __init__(
    self,
    data_path,
    num_features,
    splits = {
        "train": ["train"], # Prefix of the training metainfo
        "validation": ["val"], # Prefix of the test metainfo
        "test": ["val"], # Prefix of the test metainfo 
      },
    tokens_path=None,
    lexicon_path=None,
    use_words=False,
    prepend_wordsep=False,
    sample_rate = 16000,
    supervised = False,
    level = 'word'
  ):
    self.wordsep = ' '
    self._prepend_wordsep = prepend_wordsep
    self.num_features = num_features
    self.supervised = supervised
    
    data = []
    for _, spl in splits.items():
      for sp in spl:
        data.extend(load_data_split(data_path, sp, self.wordsep, sample_rate))
        
    tokens = set()
    for ex in data:
      for w in ex['text'].split(self.wordsep):
        tokens.update(w)
          
    self.tokens = sorted(tokens)
    self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}
    self.graphemes_to_index = self.tokens_to_index
    
  @property
  def num_tokens(self):
    return len(self.tokens)
    
  def to_index(self, line):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in line]) 

  def to_onehot(self, line):
    i = self.tokens_to_index.get(line, 0)
    v = np.eye(self.num_tokens)[i]
    return torch.FloatTensor(v)

  def _post_process(self, indices):
    return "".join(indices).strip(self.wordsep)

def load_data_split(data_path, split, wordsep, sample_rate):
  """
  Returns:
      examples : a list of mappings of
          { "audio": filename of audio,
            "text": a list of tokenized words,
            "duration": float, duration in milli-seconds,
            "interval": [float, float], start and end time of the segment in the audio in ms}
  """ 
  # json_file format: a list training file name
  segment_file = os.path.join(data_path, f"{split}2014/mscoco_{split}_word_segments.txt") # TODO Check name
  examples = []
  with open(segment_file, 'r') as segment_f:
    for line in segment_f:
      parts = line.strip().split()
      audio_id = parts[0]
      word_token = parts[1]
      begin = float(parts[2]) 
      end = float(parts[2])
      dur = end - begin
      examples.append({"audio": os.path.join(data_path, f"{split}2014/wav/{audio_id}.wav"),
                       "text": word_token,
                       "duration": dur,
                       "interval": [begin, end]})
  return examples

def create_gold_file(data_path, sample_rate): # TODO
  """
  Create the following files:
      gold_units.json : contains gold_dicts, a list of mappings of
          {"sentence_id" : str,
           "units" : a list of ints representing phoneme id for each feature frame,
           "text" : a list of strs representing phoneme tokens for each feature frame}
      abx_triplets.item : contains ABX triplets in the format
                          line 0 : whatever (not read)
                          line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
                          onset : begining of the triplet (in s)
                          offset : end of the triplet (in s)
  """ 
