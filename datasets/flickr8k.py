import torchaudio

def log_normalize(x):
    x.add_(1e-6).log_()
    mean = x.mean()
    std = x.std()
    return x.sub_(mean).div_(std + 1e-6)


class Dataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, 
      split,
      splits = {
        "train": ["fbanks"],
        "validation": ["fbank"],
        "test": ["fbank"],           
      },
      augment = True,
      sample_rate = 16000
  ):
    self.splits = splits
    self.visual_feats = np.load(os.path.join(data_path, 'flickr30k_rcnn.npz'))
 
    data = []
    for sp in self.splits[split]:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, split)
      data.extend(examples)    
  
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
    
    # Load each caption-image pairs
    self.dataset = list(zip(audio, image)) 

  def __getitem__(self, idx):
    audio_file, feat_key = self.dataset[idx]
    audio = torchaudio.load(audio_file)
    audio_feat = self.transforms(audio[0])
    image_feat = self.image_feats[feat_key]
    return audio_feat, image_feat 

def load_data_split(data_path, split): 
  visual_feats = np.load(os.path.join(data_path, 'flickr30k_rcnn.npz'))
  with open(os.path.join(sp, 'flickr40k_{}.txt'.format(split)), 'r') as f:
    filenames = [line.split('/')[-1] for line in f]  
    
  feat_keys = [k for k in sorted(visual_feats, key=lambda x:int(x.split('_')[-1])) if k.split('_')[0] in filenames]

  print('Number of {} audio files = {}, number of image files = {}'.format(split, len(filenames), len(feat_keys)))
  examples = ({'audio': fn, 'image': k}
              for fn, k in zip(filenames, feat_keys))
  return examples
