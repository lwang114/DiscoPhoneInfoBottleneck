import torch
import torchaudio
import torchvision
from torchvision import transforms
# import fairseq
import librosa
import numpy as np
import scipy
import re
import os
import json
from copy import deepcopy
from kaldiio import ReadHelper

UNK = "###UNK###"
PHN = "###PHN###"
BLANK = "###BLANK###"
SIL = "SIL"
IGNORED_TOKENS = ["SIL", "GARBAGE"]  


def log_normalize(x):
  x.add_(1e-6).log_()
  mean = x.mean()
  std = x.std()
  return x.sub_(mean).div_(std + 1e-6)


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def get_segment_id(begin, end, min_len, max_len):
  n_cdds = max_len - min_len + 1
  min_begin = end - max_len
  if begin < min_begin:
    return -1
  return n_cdds * (end - 1) + begin - min_begin 

def process_wav(wav_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    return logmel


def collate_fn_librispeech(batch):
  audios = [t[0] for t in batch]
  span_ids = [t[1] for t in batch]
  word_labels = [t[2] for t in batch]
  input_masks = [t[3] for t in batch]
  span_masks = [t[4] for t in batch]
  phone_nums = [t[5] for t in batch]
  segment_nums = [t[6] for t in batch]
  indices = [t[7] for t in batch]
  
  audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True) 
  span_ids = torch.nn.utils.rnn.pad_sequence(span_ids, batch_first=True)
  word_labels = torch.nn.utils.rnn.pad_sequence(word_labels, batch_first=True)
  input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True)
  span_masks = torch.nn.utils.rnn.pad_sequence(span_masks, batch_first=True)
  return audios, span_ids, word_labels, input_masks, span_masks, phone_nums, segment_nums, indices 
  

class UnsegmentedLibriSpeechDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path,
      preprocessor, split,
      splits = {
        "train": ["train-clean-100"],
        "test": ["dev-clean"]
      },
      augment=False,
      audio_feature="mfcc",
      image_feature="image",
      phone_label="predicted",
      sample_rate=16000,
      min_length=-1,
      max_length=40,
      wav2vec_path=None,
      debug=False
  ):
    self.preprocessor = preprocessor
    self.splits = splits[split]
    self.data_path = data_path
    self.phone_label = phone_label
    self.debug = debug 
    self.min_length = min_length
    self.max_length = max_length 
    data = [] 
    for sp in self.splits:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, sp,
                                 audio_feature=audio_feature,
                                 image_feature=image_feature,
                                 phone_label=self.phone_label,
                                 debug=debug)
      data.extend(examples)
    print(f"Number of {split} audio files = {len(data)}")

    # Set up transforms
    self.audio_feature = audio_feature
    if audio_feature == "mfcc":
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
      self.hop_length = 10
    elif audio_feature == "fbank":
      self.audio_transforms = None
      self.hop_length = 10
    elif audio_feature in ["cpc", "cpc_big"]:
      self.audio_transforms = None
      self.hop_length = 10
    elif audio_feature in ["wav2vec", "wav2vec2", "vq-wav2vec"]:
      self.audio_transforms = None
      self.hop_length = 20
    elif audio_feature in ["bnf", "bnf+cpc"]:
      self.audio_transforms = None
      self.hop_length = 10
    else:
      raise ValueError(f"Feature type {audio_feature} not supported")

    # Load each image-caption pairs
    audio = [example["audio"] for example in data]
    visual_words = [example["visual_words"] for example in data]
    spans = [example["spans"] for example in data]
    segments = [example["segments"] for example in data]
    phonemes = [example["phonemes"] for example in data]
    self.dataset = [list(item) for item in zip(audio, visual_words, spans, segments, phonemes)]
    self.audio_feature_type = audio_feature

  def segment(self, feat, segments, method="average"):
    """
      Args:
          feat : (num. of frames, feature dim.)
          segments : a list of dicts of phoneme boundaries

      Returns:
          sfeat : (num. of segments, feature dim.)
          segment_mask : (num. of segments) LongTensor
    """
    sfeats = []
    segment_mask = []
    n = len(segments)
    for end_idx in range(n):
      for l in range(end_idx+1):
        begin_idx = end_idx - l
        begin = sec_to_frame(segments[begin_idx]["begin"])
        end = sec_to_frame(segments[end_idx]["end"])
        if l < self.min_length or l > self.max_length:
          sfeats.append(torch.zeros(feat.size(-1)).log())
          segment_mask.append(0)
        else:
          while begin >= feat.size(0):
            print(f'Warning: begin time {begin} >= feat size {feat.size(0)}')
            begin -= 1
            end -= 1
          if begin == end:
            end += 1
          segment_feat = embed(feat[begin:end], method=method)
          sfeats.append(segment_feat)
          segment_mask.append(1)
    sfeats = torch.stack(sfeats)
    segment_mask = torch.tensor(segment_mask)
    return sfeats, segment_mask

  def unsegment(self, sfeat, segments):
    if sfeat.ndim == 1:
      sfeat = sfeat.unsqueeze(-1)
    dur = segments[-1]["end"] - segments[0]["begin"] 
    nframes = sec_to_frame(dur) + 1
    feat = torch.zeros((nframes, *sfeat.size()[1:]))
    for i, segment in enumerate(segments):
      if segment["text"] == SIL:
        continue
      begin = sec_to_frame(segment["begin"])
      end = sec_to_frame(segment["end"])
      if begin >= feat.size(0):
        continue
      if end == begin:
        end += 1  
      feat[begin:end] = sfeat[i]
    return feat.squeeze(-1)

  def update_segment(self, idx, new_segment_ids):
    self.dataset[idx][2] = None
    self.dataset[idx][2] = [self.dataset[idx][3][i] for i in new_segment_ids]

  def load_audio(self, audio_file):
    if self.audio_feature == "mfcc":
      audio, _ = torchaudio.load(audio_file)
      inputs = self.audio_transforms(audio).squeeze(0).t() 
    elif self.audio_feature == "fbank":
      inputs = process_wav(audio_file)
      inputs = torch.FloatTensor(inputs).t()
    elif self.audio_feature in ["cpc", "cpc_big"]:
      if audio_file.split(".")[-1] == "txt":
        inputs = np.loadtxt(audio_file)
      else:
        with ReadHelper(f"ark: gunzip -c {audio_file} |") as reader:
          for _, inputs in reader:
            continue
      inputs = torch.FloatTensor(inputs)
    elif self.audio_feature in ["bnf", "bnf+cpc"]:
      if audio_file.split('.')[-1] == "txt":
        inputs = np.loadtxt(audio_file)
      else:
        with ReadHelper(f"ark: gunzip -c {audio_file} |") as ark_f:
          for k, inputs in ark_f:
            continue
      if self.audio_feature_type == "bnf+cpc":
        cpc_feat = np.loadtxt(audio_file.replace("bnf", "cpc"))
        feat_len = min(inputs.shape[0], cpc_feat.shape[0])
        inputs = np.concatenate([inputs[:feat_len], cpc_feat[:feat_len]], axis=-1)
      inputs = torch.FloatTensor(inputs)
    else:
      raise NotImplementedError(f'Audio feature {self.audio_feature} not implemented')
    return inputs

  def __getitem__(self, idx): 
    audio_file, visual_words,\
    spans, segments,\
    phonemes = self.dataset[idx]
    
    audio_inputs = self.load_audio(audio_file)
    audio_inputs, input_mask = self.segment(audio_inputs, segments)
    span_ids = torch.LongTensor(
        [self.get_span_id(*span) for span in spans]
    )
    span_mask = torch.ones(len(span_ids))
    segment_num = len(segments)
    phoneme_num = len(phonemes)
    visual_sent = [w["text"] for w in visual_words]
    word_labels = self.preprocessor.to_word_index(visual_sent)
    return audio_inputs, span_ids, word_labels, input_mask, span_mask, phoneme_num, segment_num, idx   
  def span_to_segment(self, idx):
    spans = self.dataset[idx][2]
    segments = self.dataset[idx][3]
    return [
        {
            'begin': segments[span[0]]['begin'], 
            'end': segments[span[1]]['end'], 
            'text': PHN
        } for span in spans
    ]

  def get_span_id(self, begin, end):
    return int(end * (end + 1) / 2 + end - begin)

  def update_spans(self, idx, spans):
    self.dataset[idx][2] = None
    self.dataset[idx][2] = deepcopy(spans)

  def __len__(self):
    return len(self.dataset)
 

class UnsegmentedLibriSpeechPreprocessor: 
  def __init__(
    self, data_path,
    num_features,
    splits = {
        "train": ["train-clean-100"],
        "test": ["dev-clean"]
    },
    audio_feature="mfcc",
    image_feature="rcnn",
    phone_label="predicted",
    sample_rate=16000,
    ignore_index=-100,
    debug=False
  ):
    self.num_features = num_features 
    self.ignore_index = ignore_index
    
    data = []
    for spl in splits:
      for sp in splits[spl]:
        data.extend(load_data_split(data_path, sp,
                                    audio_feature=audio_feature,
                                    image_feature=image_feature,
                                    phone_label=phone_label,
                                    debug=debug))
    tokens = set()
    visual_words = set()
    for ex in data:
      sent = [phn["text"] for phn in ex["phonemes"]]
      visual_sent = [w["text"] for w in ex["visual_words"]]
      tokens.update(sent)
      visual_words.update(visual_sent)
    self.tokens = [BLANK]+sorted(tokens)
    self.visual_words = [BLANK]+sorted(visual_words)
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
    return torch.LongTensor([self.word_to_index[t] for t in sent])
  
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
                                    

def load_data_split(data_path, sp,
                    audio_feature="cpc",
                    image_feature="rcnn",
                    phone_label="predicted",
                    debug=False):
  """
  Returns : 
      examples : a list of mappings of
          { ``audio`` : filename of audio,
            ``visual_words`` : a list of dicts for visual words in each utterance as
                { ``text`` : str,
                  ``begin`` : float,
                  ``end`` : float,
                  ``segment_id``: int}
            ``phonemes`` : a list of dicts for phonemes in each utterance as
                { ``text`` : str,
                  ``begin`` : float,
                  ``end`` : float,
                  ``segment_id`` : int
                }
            ``phoneme_num`` : int,
            ``frame_num`` : int
          }
  """
  # TODO Add option to exclude silences
  label_f = open(os.path.join(data_path, sp, f"{sp}.json"), "r") 
  examples = []
  absent_utt_ids = []
  for idx, line in enumerate(label_f):
    if debug and idx > 20:
      break
    label_dict = json.loads(line.rstrip("\n"))
    if "utterance_id" in label_dict:
      utt_id = label_dict["utterance_id"] 
    else:
      utt_id = label_dict["audio_id"]
    visual_words = [label_dict["words"][i] for i in label_dict.get("visual_words", [])]
    phonemes_with_stress = [phn for w in label_dict["words"] for phn in w["phonemes"]]   
    phonemes = []
    for phn in phonemes_with_stress: # Remove stress label
      if (phn["text"][0] == "+") or (phn["text"] in IGNORED_TOKENS):
        continue 
      if not "phoneme" in phn["text"]:
        phn["text"] = re.sub(r"[0-9]", "", phn["text"])
      phonemes.append(phn)

    segments = []
    if phone_label == "groundtruth":
      segments = phonemes.copy()
    elif phone_label == "multilingual": 
      segments = deepcopy(label_dict["predicted_segments_multilingual"])
    elif phone_label == "multilingual_phones":
      segments = deepcopy(label_dict["multilingual_phones"])
    elif phone_label == "quantized_units":
      segments = deepcopy(label_dict["quantized_units"])
    elif phone_label == "predicted":
      segments = deepcopy(label_dict["predicted_segments"])
    else:
      raise ValueError(f"Invalid phone label type: {phone_label}")
    
    if audio_feature in ["mfcc", "fbank", "wav2vec", "wav2vec2", "vq-wav2vec"]:
      if len(utt_id.split("/")) > 1:
        audio_path = f"{utt_id}.wav"
      else:
        audio_path = os.path.join(data_path, sp, f"{utt_id}.wav")
    elif audio_feature in ["cpc", "cpc_big"]:
      utt_id = os.path.basename(utt_id)
      audio_file = f"{utt_id}.ark.gz"
      audio_path = os.path.join(data_path, f"{sp}_{audio_feature}", audio_file)
      if not os.path.exists(audio_path):
        audio_file = f"{utt_id}.txt"
        audio_path = os.path.join(data_path, f"{sp}_{audio_feature}_txt", audio_file)
    elif audio_feature in ["bnf", "bnf+cpc"]:
      utt_id = os.path.basename(utt_id)
      audio_file = f"{utt_id}.txt"
      audio_path = os.path.join(data_path, f"{sp}_bnf_txt", audio_file)
    else:
      raise ValueError(f"Audio feature type {audio_feature} not supported")
    
    if len(phonemes) == 0:
      print(f'{utt_id} has no phoneme annotations')
      continue

    # Extract number of phonemes
    phoneme_num = len(phonemes) 

    # Initialize spans to be all seed segments
    spans = []
    for span_idx, segment in enumerate(segments):
      spans.append([span_idx, span_idx])

    if os.path.exists(audio_path):
      example = {"audio": audio_path,
                 "visual_words": visual_words,
                 "spans": spans,
                 "segments": segments,
                 "phonemes": phonemes,
                 "phoneme_num": phoneme_num}
      examples.append(example)
    else:
      absent_utt_ids.append(utt_id)
  
  if len(absent_utt_ids) > 0:
    print(f'Ignore the following utterance that does not exist: {absent_utt_ids}')
  label_f.close()
  return examples

def embed(feat, method='average'):
    if method == 'average':
        return feat.mean(0)
    elif method == 'resample':
        new_feat = signal.resample(feat.detach().numpy(), 4)
        return torch.FloatTensor(new_feat.flatten())
    else:
        raise ValueError(f'Unknown embedding method {method}')

def sec_to_frame(t):
    return int(round(t*100, 3))

def frame_to_sec(n):
    return float(n) / 100
