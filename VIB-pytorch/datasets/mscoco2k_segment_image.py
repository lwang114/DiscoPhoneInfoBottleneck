
import json
import os
import re
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

class MSCOCO2kSegmentImageDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path,
      preprocessor, split,
      splits = {
        "train": ["train"], # Prefix of the training metainfo
        "validation": ["val"], # Prefix of the test metainfo
        "test": ["val"], # Prefix of the test metainfo 
      },
      augment=True,
      sample_rate = 16000,
  ):
    self.splits = splits
    self.data_path = data_path
    self.sample_rate = sample_rate
    self.max_feat_len = 64
    
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
    image_ids = [example['image_id'] for example in data]
    feat_idxs = [example['feat_idx'] for example in data]
    self.dataset = list(zip(audio, text, duration, interval, image_ids, feat_idxs))    
    # Create gold unit file
    if not os.path.exists(os.path.join(data_path, "gold_units.json")) or not os.path.exists(os.path.join(data_path, "abx_triplets.item")):
      create_gold_file(data_path, sample_rate) 
    self.gold_dicts = json.load(open(os.path.join(data_path, "gold_units.json")))
    self.image_feats = np.load(os.path.join(data_path, f"feats/mscoco2k_res34_embed512dim.npz"))

  def sample_sizes(self):
    """
    Returns a list of tuples containing the input size
    (time, 1) and the output length for each sample.
    """
    return [((duration, 1), len(text)) for _, text, duration in self.dataset]

  def __getitem__(self, index):
      audio_file, text, dur, interval, image_id, feat_idx = self.dataset[index]
      begin = int(interval[0] * (self.sample_rate // 1000))
      end = int(interval[1] * (self.sample_rate // 1000))
      audio, _ = torchaudio.load(audio_file)
      try:
          inputs = self.transforms(audio[:, begin:end]).squeeze(0)
      except:
          inputs = self.transforms(audio)[:, :, int(begin // 10):int(end // 10)].squeeze(0)
      
      image_feat = self.image_feats[image_id][feat_idx] 
      image_inputs = torch.FloatTensor(image_feat)

      nframes = inputs.size(-1)
      input_mask = torch.zeros(self.max_feat_len)
      input_mask[:nframes] = 1.
      inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t()
      outputs = self.preprocessor.to_index(text).squeeze(0)
      
      return inputs, image_inputs, outputs, input_mask

  def __len__(self):
      return len(self.dataset)

class MSCOCO2kSegmentImagePreprocessor:
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
    sample_rate = 16000,
    supervised = True,
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

    # lexicon is a mapping from word to its corresponding token ids 
    tokens = set()
    lexicon = {}
    for ex in data:
        for w in ex['text'].split(self.wordsep):
            if not w in lexicon:
                if level == 'phone':
                    lexicon[w] = ['{:03d}'.format(5 * len(lexicon) + i) for i in range(5)]
                else:
                    lexicon[w] = [len(lexicon)]    
                tokens.update(lexicon[w])
    self.lexicon = lexicon
    self.tokens = sorted(tokens)
    self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}
    self.graphemes_to_index = self.tokens_to_index

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
      else:
          return torch.LongTensor(tok_to_idx.get(line, 0))

  def to_onehot(self, line):
      i = self.tokens_to_index.get(line, 0)
      v = np.eye(self.num_tokens)[i]
      return torch.FloatTensor(v)
  
  def to_text(self, indices):
      # Roughly the inverse of `to_index`
      encoding = self.tokens
      text = [self.tokens_to_lexicon[encoding[i]] for i in indices]
      text = []
      prev_word = None
      for i in indices:
        word = self.tokens_to_lexicon[encoding[i]]
        if not prev_word or prev_word != word:
          prev_word = word
          text.append(word)
      return self._post_process(text)

  def tokens_to_text(self, indices):
      text = []
      prev_word = None
      for i in indices:
        word = self.tokens_to_lexicon[self.tokens[i]]
        if not prev_word or prev_word != word:
          prev_word = word
          text.append(word)
      return self._post_process(text)
      # return self._post_process(self.tokens[i] for i in indices)

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
    select_idxs = [idx for idx, is_test in enumerate(open(split_file, 'r')) if not int(is_test)][:20] # XXX
    print('Number of training examples={}'.format(len(select_idxs)))  
  else:
    select_idxs = [idx for idx, is_test in enumerate(open(split_file, 'r')) if int(is_test)][:20] # XXX
    print('Number of test examples={}'.format(len(select_idxs)))  

  examples = []
  phone_info_dict = json.load(open(os.path.join(data_path, 'mscoco2k_phone_info.json'), 'r'))

  with open(wav_scp_file, 'r') as wav_scp_f,\
       open(text_file, 'r') as text_f:
      filenames = [l.split()[-1] for idx, l in enumerate(wav_scp_f)]
      texts = [' '.join(l.split()[1:]) for idx, l in enumerate(text_f)]
      for idx, (_, phone_info) in enumerate(sorted(phone_info_dict.items(), key=lambda x:int(x[0].split("_")[-1]))):
          if not idx in select_idxs:
              continue

          begin_word = 0
          for w_idx, (word_info, word_token) in enumerate(zip(phone_info["data_ids"], phone_info["concepts"])):
              dur_word = word_info[2][-1][2] - word_info[2][0][1]
              end_word = begin_word + dur_word
              example = {'audio': filenames[idx],
                         'text': word_token,
                         'duration': dur_word,
                         'interval': [begin_word, end_word],
                         'image_id': f'arr_{idx}',
                         'feat_idx': w_idx}
              examples.append(example)
              begin_word += dur_word
  return examples

def create_gold_file(data_path, sample_rate):
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
  wav_scp_file = os.path.join(data_path, "mscoco2k_wav.scp")
  split_file = os.path.join(data_path, "mscoco2k_retrieval_split.txt")
  select_idxs = [idx for idx, is_test in enumerate(open(split_file, 'r')) if int(is_test)]

  phone_info_dict = json.load(open(os.path.join(data_path, "mscoco2k_phone_info.json"), "r"))
  phone_to_index = {}
  gold_dicts = []
  triplets = ['#file_ID onset offset #phone prev-phone next-phone speaker']
  # Extract audio file names as sentence ids
  with open(wav_scp_file, 'r') as wav_scp_f:
    filenames = [l.split()[-1] for idx, l in enumerate(wav_scp_f)]

  # Extract utterance duration
  durations = [int(torchaudio.load(fn)[0].size(-1) * 1000 // (10 * sample_rate)) for fn in filenames]

  # Extract phone mapping
  phone_path = os.path.join(data_path, "phone2id.json")
  if os.path.exists(phone_path):
    phone_to_index = json.load(open(phone_path, "r"))
  else:
    phones = set()
    for idx, (_, phone_info) in enumerate(sorted(phone_info_dict.items(), key=lambda x:int(x[0].split("_")[-1]))):
      for word_token in phone_info["data_ids"]:
        for phone_token in word_token[2]:
          token = phone_token[0]
          phones.update([token])
    phone_to_index = {x:i for i, x in enumerate(sorted(phones))}
    phone_to_index[UNK] = len(phone_to_index) 
    json.dump(phone_to_index, open(phone_path, "w"), indent=2)  

  # Extract phone units
  phone_to_word_counts = collections.defaultdict(dict)
  global_idx = 0
  for idx, (_, phone_info) in enumerate(sorted(phone_info_dict.items(), key=lambda x:int(x[0].split("_")[-1]))):
    if not idx in select_idxs:
      continue

    begin_word = 0
    for word_info, word_token in zip(phone_info["data_ids"], phone_info["concepts"]):
      dur_word = word_info[2][-1][2] - word_info[2][0][1]
      end_word = begin_word + dur_word
      nframes = int(dur_word // 10)
      gold_dict = {"sentence_id": filenames[idx],
                   "units": [-1]*nframes,
                   "phoneme_text": [UNK]*nframes,
                   "word_text": [word_token]*nframes,
                   "interval": [begin_word, end_word]
      }
      begin_phone = 0
      prefix = filenames.split('/')[-1] 
      example_id = f"{prefix}_{global_idx}"
      global_idx += 1
      for phone_token in word_info[2]:
        if not word_token in phone_to_word_counts[phone_token[0]]:
            phone_to_word_counts[phone_token[0]][word_token] = 1
        else:
            phone_to_word_counts[phone_token[0]][word_token] += 1
        
        token, begin, end = phone_token[0], phone_token[1], phone_token[2]
        
        dur_phone = end - begin
        begin_frame = int(begin_phone // 10)
        end_frame = int((begin_phone + dur_phone) // 10)
        if (begin_word + begin_phone + dur_phone) // 10 > durations[idx]:
          print('In {}: end frame exceeds duration of audio, {} > {}'.format(filenames[idx], (begin_word + begin_phone + dur_phone) // 10, durations[idx]))
          break
        triplets.append(f'{example_id} {begin_phone / 1000.0:.4f} {(begin_phone + dur_phone)/ 1000.0:.4f} {token} {NULL} {NULL} 0')

        for t in range(begin_frame, end_frame):
          gold_dict["units"][t] = phone_to_index[token]
          gold_dict["phoneme_text"][t] = token
        begin_phone += dur_phone
      if end_frame != nframes:
          gold_dict['phoneme_text'] = gold_dict['phoneme_text'][:end_frame]
          gold_dict['word_text'] = gold_dict['word_text'][:end_frame]
          print('sentence_id, end_frame, nframes: ', filenames[idx], end_frame, nframes)
      gold_dicts.append(gold_dict)
      begin_word += dur_word
      
  with open(os.path.join(data_path, 'phone_token_top_10_words.txt'), 'w') as f:
    f.write('Phone\tWord\tCounts\n')
    for p in phone_to_word_counts:
      for w in sorted(phone_to_word_counts[p], key=lambda x:phone_to_word_counts[p][x], reverse=True):
        f.write('{}\t{}\t{}\n'.format(p, w, phone_to_word_counts[p][w]))

  with open(os.path.join(data_path, "gold_units.json"), "w") as gold_f:
    json.dump(gold_dicts, gold_f, indent=2) 
  
  with open(os.path.join(data_path, "abx_triplets.item"), "w") as triplet_f:
    f.write('\n'.join(triplets))

if __name__ == "__main__":
  data_path = "/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k"
  preproc = Preprocessor(data_path, 80) 
  dataset = Dataset(data_path, preproc, "test")
  for k in range(5):
      inputs, outputs, input_masks = dataset[k]
