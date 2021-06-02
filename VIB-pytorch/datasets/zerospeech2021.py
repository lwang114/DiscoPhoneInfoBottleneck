import torch
import os
import json
import numpy as np

def fix_embedding_length(emb, L):
  size = emb.size()[1:]
  if emb.size(0) < L:
    pad = [torch.zeros(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb


class ZeroSpeech2021_Dataset(torch.utils.data.Dataset):


  def __init__(
    self, data_path,
    split,
    preprocessor=None,
    splits={
      "dev": ["dev-clean"],
      "test": ["test-clean"]  
    }
  ):
    self.preprocessor = preprocessor
    self.splits = splits[split]
    self.data_path = data_path
    self.max_feat_len = 1024

    data = []
    for sp in self.splits:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, sp)
      data.extend(examples)
    
    audio = [example["audio"] for example in data]
    text = [example["text"] for example in data]
    self.dataset = list(zip(audio, text))

  def __getitem__(self, idx):
    audio_file, text = self.dataset[idx]
    audio_input = np.loadtxt(audio_file)
    audio_input = torch.FloatTensor(audio_input)
    audio_input = fix_embedding_length(audio_input, self.max_feat_len)
    nframes = audio_input.size(0)
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.
    
    return audio_input.t(), 0, 0, input_mask, 1.
     
  def __len__(self):
    return len(self.dataset)


def load_data_split(data_path, sp):
  examples = []
  for fn in os.listdir(os.path.join(data_path, sp)):
    if fn.endswith(".wav"):
      audio_id = os.path.splitext(fn)[0]
      path = os.path.join(data_path, f"cpc/{audio_id}.txt")
      examples.append({"audio": path,
                       "text": ''})
  return examples
        
