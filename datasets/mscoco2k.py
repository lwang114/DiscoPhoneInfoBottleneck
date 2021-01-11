import json
import os
import re
import torch
import torchaudio
import torchvision 

UNK = '###UNK###'
def log_normalize(x):
    x.add_(1e-6).log_()
    mean = x.mean()
    std = x.std()
    return x.sub_(mean).div_(std + 1e-6)


class Dataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path,
      preprocessor, split,
      splits = {
        "train": ["train"], # Prefix of the training metainfo
        "validation": ["val"], # Prefix of the test metainfo
        "test": ["val"], # Prefix of the test metainfo 
      },
      augment=True,
      sample_rate = 16000
  ):
    self.splits = splits
    self.sample_rate = sample_rate

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
    duration = [example['duration'] for example in data]
    self.dataset = list(zip(audio, text, duration))

  def sample_sizes(self):
    """
    Returns a list of tuples containing the input size
    (time, 1) and the output length for each sample.
    """
    return [((duration, 1), len(text)) for _, text, duration in self.dataset]

  def __getitem__(self, index):
      audio_file, text, _ = self.dataset[index]
      audio = torchaudio.load(audio_file)
      inputs = self.transforms(audio[0])
      outputs = self.preprocessor.to_index(text)
      return inputs, outputs

  def __len__(self):
      return len(self.dataset)


 
class Preprocessor:
  """
  A preprocessor for the MSCOCO 2k dataset.
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
    sample_rate = 16000
  ):
    self.wordsep = ' '
    self._prepend_wordsep = prepend_wordsep
    self.num_features = num_features

    data = []
    for sp in splits['train']:
      data.extend(load_data_split(data_path, sp, self.wordsep, sample_rate))

    # Load the set of graphemes:
    tokens = set()
    for ex in data:
      tokens.update(ex['text'])
    self.tokens = sorted(tokens)
    self.tokens = [UNK] + self.tokens 

    # Build the token-to-index and index-to-token maps:
    if tokens_path is not None:
      with open(tokens_path, 'r') as fid:
        self.tokens = [l.strip() for l in fid]
    
    if lexicon_path is not None:
      with open(lexicon_path, "r") as fid:
        lexicon = (l.strip().split() for l in fid)
        lexicon = {l[0]: l[1:] for l in lexicon}
        self.lexicon = lexicon
    else:
      self.lexicon = None

    self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}

  @property
  def num_tokens(self):
      return len(self.tokens)

  def to_index(self, line):
      tok_to_idx = self.tokens_to_index
      if self.lexicon is not None:
          if len(line) > 0:
              # If the word is not found in the lexicon, fall back to letters.
              line = [
                  t
                  for w in line.split(self.wordsep)
                  for t in self.lexicon.get(w, self.wordsep + w)
              ]
          tok_to_idx = self.tokens_to_index
      
      return torch.LongTensor([tok_to_idx.get(t, 0) for t in line])

  def to_text(self, indices):
      # Roughly the inverse of `to_index`
      encoding = self.tokens
      return self._post_process(encoding[i] for i in indices)

  def tokens_to_text(self, indices):
      return self._post_process(self.tokens[i] for i in indices)

  def _post_process(self, indices):
      # ignore preceding and trailling spaces
      return "".join(indices).strip(self.wordsep)


def load_data_split(data_path, split, wordsep, sample_rate):
  """
  Returns:
      examples : a list of mappings of
          { "audio": filename of audio,
            "text": a list of tokenized words,
            "duration": float}
  """
  # json_file format: a list training file name
  wav_scp_file = os.path.join(data_path, 'mscoco2k_wav.scp')
  text_file = os.path.join(data_path, 'text')
  split_file = os.path.join(data_path, 'mscoco2k_retrieval_split.txt')

  if split == 'train':
    select_idxs = [idx for idx, is_test in enumerate(open(split_file, 'r')) if not int(is_test)] # XXX
    print('Number of training examples={}'.format(len(select_idxs)))  
  else:
    select_idxs = [idx for idx, is_test in enumerate(open(split_file, 'r')) if int(is_test)] # XXX
    print('Number of test examples={}'.format(len(select_idxs)))  

  with open(wav_scp_file, 'r') as wav_scp_f,\
       open(text_file, 'r') as text_f:
    filenames = [l.split()[-1] for idx, l in enumerate(wav_scp_f) if idx in select_idxs]
    texts = [l.split()[1:] for idx, l in enumerate(text_f) if idx in select_idxs]
    durations = [torchaudio.load(fn)[0].size()[0] / float(sample_rate) for fn in filenames]
    examples = [{'audio': fn, 'text':text, 'duration':dur} for text, fn, dur in zip(texts, filenames, durations)]  
  return examples
