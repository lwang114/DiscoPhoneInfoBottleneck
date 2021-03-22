import torchaudio
import nltk
from nltk.stem import WordNetLemmatizer
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
import numpy as np
import re

# TODO Install torchaudio
dep_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
dep_parser._model = dep_parser._model.cuda()
lemmatizer = WordNetLemmatizer()
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

class FlickrSegmentDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, 
      preprocessor, split,
      splits = {
        "train": ["fbanks"],
        "validation": ["fbank"],
        "test": ["fbank"],           
      },
      augment = True,
      sample_rate = 16000
  ):
    self.splits = splits
    self.data_path = data_path
    self.sample_rate = sample_rate
    self.max_feat_len = 200 # TODO Check the best length
 
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
    self.transforms = torchvision.transforms.Compose(self.transforms)

    # Load each image-caption pairs
    audio = [example["audio"] for example in data]
    text = [example["text"] for example in data]
    duration = [example["duration"] for example in data]
    interval = [example["interval"] for example in data]
    image_ids = [example["image_id"] for example in data]
    feat_idxs = [example["feat_idx"] for example in data]
    self.dataset = list(zip(audio, text, duration, interval, image_ids, feat_idxs))

    # Create gold unit file
    if not os.path.exists(os.path.join(data_path, "gold_units.json")) or not os.path.exists(os.path.join(data_path, "abx_triplets.item")):
      create_gold_file(data_path, sample_rate) # TODO
    self.gold_dicts = json.load(open(os.path.join(data_path, "gold_units.json")))
    self.image_feats = np.load(os.path.join(data_path, "flickr30k_res34_embed512dim.npz")) # TODO

  def sample_sizes(self):
    """
    Returns a list of tuples containing the input size
    (time, 1) and the output length for each sample.
    """
    return [((duration, 1), len(text)) for _, text, duration in self.dataset]

  def __getitem__(self, idx):
    audio_file, label, dur, interval, image_id, feat_idx = self.dataset[idx]
    begin = int(interval[0] * self.sample_rate)
    end = int(interval[1] * self.sample_rate)
    audio = torchaudio.load(audio_file)
    try:
      inputs = self.transforms(audio[0]).squeeze(0)
    except:
      inputs = self.transforms(audio)[:, :, int(begin // 10):int(end // 10)].squeeze(0)

    image_feat = self.image_feats[image_id][feat_idx]
    image_inputs = torch.FloatTensor(image_feat)

    nframes = inputs.size(-1)
    input_mask = torch.zeros(self.max_feat_len)
    input_mask[:nframes] = 1.
    inputs = fix_embedding_length(inputs.t(), self.max_feat_len).t()
    outputs = self.preprocessor.to_index(label).squeeze(0) # TODO Check this

    return inputs, image_inputs, outputs, input_mask 

  def __len__(self):
    return len(self.dataset)


class FlickrSegmentPreprcessor:
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
        "validation": ["test"],
        "test": ["test"]
    },
    tokens_path=None,
    lexicon_path=None,
    use_words=False,
    prepend_wordsep=False,
    sample_rate=16000
  ):
    self.wordsep = " "
    self._prepend_wordsep = prepend_wordsep
    self.num_features = num_features

    metadata_file = os.path.join(data_path, "flickr8k_phrases.json")
    if not metadata_file:
      self.extract_phrase_info(data_path)
    
    data = []
    for _, spl in splits.items(): 
      for sp in spl:
        data.extend(load_data_split(data_path, sp, self.wordsep, sample_rate))
   
    tokens = set()
    lexicon = {}
    for ex in data:
      tokens.add(ex["label"])
    self.tokens = sorted(tokens)
    self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}
    print(f"Number of types: {self.num_tokens:d}")
  
  @property
  def num_tokens(self):
    return len(self.tokens)

  def to_index(self, line):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in line])

  def extract_phrase_info(self, data_path):
    word_dir = os.path.join(os.path.join(data_path, "word_segmentation"))
    phone_f = open(os.path.join(data_path, "flickr_labels.json"), "r")
    phrase_f = open(os.path.join(data_path, "flickr8k_phrases.txt"), "r")
    bbox_f = open(os.path.join(data_path, "flickr8k_bboxes.txt"), "r")
    sent_f = open(os.path.join(data_path, "flickr_sentences.txt"), "r")
    out_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")

    # Build a mapping form file id to caption text
    id_to_text = {} 
    for line in sent_f: 
      parts = line.strip("\n").split()
      image_caption_id = parts[0] 
      capt_id = image_caption_id.split('.')[-1].split("_")[-1]
      utterance_id = "_".join(image_caption_id.split(".")[0], capt_id)
      text = " ".join(parts[1:])
      id_to_text[utterance_id] = text

    # Build a mapping from file id to phrase info
    id_to_phrase = {}
    for line_phrase, line_bbox in zip(phrase_f, bbox_id):
      parts = line_phrase.strip("\n").split()
      image_id = parts[0].split(".")[0]
      capt_id = parts[1] 
      phrase = " ".join(parts[3:-1])
      begin = int(parts[-1])
      end = begin + len(parts[3:-1]) - 1
      if capt_id != '1':
        continue
      utterance_id = "_".join([image_id, capt_id])
      if not utterance_id in id_to_phrase:
        id_to_phrase[utterance_id] = []

      bbox = line_bbox.strip("\n").split()[-4:]
      phrase_info = {"utterance_id": utterance_id,
                     "text": phrase, 
                     "begin": begin, 
                     "end": end,
                     "bbox": [int(x) for x in bbox],
                     "feat_idx": len(id_to_phrase[utterance_id])}
      id_to_phrase[utterance_id].append(phrase_info)

    # Iterate over each speech file to extract utterance info
    cur_utterance_id = None
    cur_phones = None
    cur_words = None
    idx = 0
    for line in phone_f:
      if "align" in line:
        if cur_utterance_id:
          idx += 1 # XXX
          if idx > 10:
            break
          print(cur_utterance_id) # XXX
          utt = Utterance(cur_phones, cur_words, 
                          id_to_phrase[cur_utterance_id],
                          id_to_text[cur_utterance_id])
          f.write(json.dumps(utt.phrases)+'\n')
        cur_utterance_id = "_".join(line.strip("\n").split(".")[0].split("_")[1:])
        cur_phones = []
        cur_words = []
        with open(os.path.join(word_dir, cur_utterance_id+".word", "r") as word_f:
          for line in word_f:
            w, begin, end = line.split()
            if "$(" in w:
              continue
            w = re.sub(r"[^\w]", "", w)
            cur_words.append({"text": w, 
                              "begin": float(begin), 
                              "end": float(end)})
      phn, begin, end = line.split()
      cur_phones.append({"text": phn, 
                         "begin": float(begin), 
                         "end": float(end)})
    utt = Utterance(cur_phones, cur_words,
                    id_to_phrase[cur_utterance_id],
                    id_to_text[cur_utterance_id])
    f.write(json.dumps(utt.phrases)+'\n')
    phone_f.close()
    phrase_f.close()
    sent_f.close()

class Utterance:
  def __init__(self, phones, words, phrases, rawtext):
    """
    Args:
        phones : a list of dicts of {"text": str, "begin": begin time in sec, "end": end time in sec}
        words : a list of dicts of {"text": str, "begin": begin time in sec, "end": end time in sec}
        phrases : a list of dicts of {"text": str, "begin": begin time in word tokens, "end": end time in word tokens}
        rawtext : a list of strs
    """
    phones = sorted(phones, key=lambda x:x['begin'])
    words = sorted(words, key=lambda x:x['begin'])
    phrases = sorted(phrases, key=lambda x:x['begin'])

    words, phrases = self.extract_char_offsets(words, phrases, rawtext)
    words = self.align(phones, words)
    phrases = self.align(words, phrases)
    self.phrases = self.extract_phrase_labels(phrases)

  def extract_char_offsets(self, words, phrases, rawtext):
    # Extract char offsets for text words
    text_offsets = []
    begin = 0
    for token in rawtext:
      text_offsets.append([begin, begin+len(token)-1])
      begin += len(token)

    # Convert token offsets to char offsets
    for idx in range(len(phrases)):
      begin_char = text_offsets[phrases[idx]["begin"]][0]
      end_char = text_offsets[phrases[idx]["end"]][1]
      phrases[idx]["begin"] = begin_char
      phrases[idx]["end"] = end_char 

    # Extract char offsets for acoustic words
    begin = 0
    for word in words:
      word["begin_char"] = begin
      word["end_char"] = begin+len(word["text"])-1
      begin += len(word["text"])
    
    return words, phrases

  def align(self, children, parents):
    parent_idx = 0
    n_parents = len(parents)

    for child in children:
      parent = parents[parent_idx]
      parent["children"] = []
      if "begin_char" in child:
        begin, end = child["begin_char"], child["end_char"]
      else:
        begin, end = child["begin"], child["end"]

      if begin > parent["end"]:
        parent_idx += 1
        if parent_idx >= n_parents:
          break
        parent = parents[parent_idx] 
      
      if end < parent["begin"]:
        continue
      parent["children"].append(child)
    return parents  

  def extract_phrase_labels(self, phrases):
    for phrase in phrases:
      text = phrase["text"].split()
      pos_tags = nltk.pos_tag(text, tagset="universal")
      instance = dep_parser._dataset_reader.text_to_instance(text, pos_tags)
      parsed_text = dep_parser.predict_batch_instance([instance])[0]
      head_idx = np.nonzero(parsed_text["predicted_heads"] == 0)[0]
      phrase["label"] = lemmatizer.lemmatize(text[head_idx])
    return phrases

def load_data_split(data_path, split):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "text" : a list of tokenized words for the class name,
            "full_text" : a list of tokenized words for the whole phrase, 
            "duration" : float,
            "interval": [begin of the word in ms, end of the word in ms],
            "image_id": str,
            "feat_idx": int, image feature idx
          }
  """
  with open(os.path.join(sp, "flickr40k_{}.txt".format(split)), "r") as f:
    filenames = [line.split("/")[-1] for line in f]  
  
  image_feats = np.load(os.path.join(data_path, "flickr30k_res34_embed512dim.npz")) # TODO Check filename
  utt_to_feat = {'_'.join(k.split('_')[:-1]):k for k in image_feats} # TODO Check key name

  examples = []
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"))
  for line in phrase_f:
    phrase = json.loads(line.rstrip("\n"))
    utterance_id = phrase["utterance_id"]
    fn = os.path.join(data_path, "wav", utterance_id + ".wav") # TODO Check this
    if fn in filenames:
      example = {"audio": fn,
                 "text": phrase["label"],
                 "full_text": phrase["text"],
                 "duration": phrase["end"] - phrase["begin"],
                 "interval": [phrase["begin"], phrase["end"]],
                 "feat_idx": phrase["feat_idx"],
                 "image_id": utt_to_feat["_".join(phrase["feat_idx"].split("_")[:-1])]}
      examples.append(example)

  print("Number of {} audio files = {}".format(split, len(examples))
  phrase_f.close()
  return examples

def create_gold_file(data_path, sample_rate): # TODO
  """
  Create the following files:
      gold_units.json : contains gold_dicts, a list of mappings 
          {"sentence_id" : str,
           "units" : a list of ints representing phoneme id for each feature frame,
           "text" : a list of strs representing phoneme tokens for each feature frame}
     abx_triplets.item : contains ABX triplets in the format
                         line 0 : whatever (not read)
                         line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
                         onset : begining of the triplet (in s)
                         offset : end of the triplet (in s)
  """
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  filenames = [line.split("/")[-1] for line in open(os.path.join(data_path, "flickr40k_test.txt"), "r")] 
  phone_to_index = {}
  gold_dicts = []
  triplets = ['#file_ID onset offset #phone prev-phone next-phone speaker']

  phrase_idx = 0
  for line in phrase_f:
    phrase = json.loads(line.rstrip("\n"))
    utterance_id = phrase["utterance_id"]
    fn = os.path.join(data_path, "wav", utterance_id + ".wav") # TODO Check this
    if fn in filenames:
      label = phrase["text"]
      dur_word = phrase["end"] - phrase["start"]
      nframes = int(dur_word * 100)

      gold_dict = {"sentence_id": fn,
                   "units": [-1]*nframes,
                   "phoneme_text": [NULL]*nframes,
                   "word_text": [label]*nframes,
                   "word_full_text": [NULL]*nframes
      }
      
      begin_phone = 0
      begin_word = 0
      example_id = f"{utterance_id}_{phrase_idx}"
      phrase_idx += 1
      for word in phrase["children"]:
        dur_word = int((word["end"] - word["begin"])*100) 
        end_word = begin_word + dur_word
        gold_dict["word_full_text"][begin_word:end_word+1] = word["text"]
        begin_word += dur_word 
        
        for phn_idx, phone in enumerate(word["children"]):
          if not phone["text"] in phone_to_index:
            phone_to_index[phone["text"]] = len(phone_to_index)
          dur_phone = int((phone["end"] - phone["begin"])*100)
          end_phone = begin_phone + dur_phone 
          gold_dict["phoneme_text"][begin_phone:end_phone] = phone["text"]
          gold_dict["units"][begin_phone:end_phone] = phone_to_index[token]
          
          if phn_idx == 0:
            prev_token = NULL
          else:
            prev_token = word["children"][phn_idx-1]["text"]

          if phn_idx == len(word["children"]) - 1:
            next_token = NULL
          else:
            next_token = word["children"][phn_idx+1]["text"]
          triplets.append(f'{example_id} {begin_phone} {begin_phone + dur_phone} {token} {prev_token} {next_token} 0')
          begin_phone += dur_phone

      gold_dicts.append(gold_dict)
  
  with open(os.path.join(data_path, "gold_units.json"), "w") as gold_f:
    json.dump(gold_dicts, gold_f, indent=2)

  with open(os.path.join(data_path, "abx_triplets.item"), "w") as triplet_f:
    triplet_f.write('\n'.join(triplets))
