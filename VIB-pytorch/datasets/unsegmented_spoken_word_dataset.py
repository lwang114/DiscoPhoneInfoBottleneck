import torch
import torchaudio
import torchvision
from torchvision import transforms
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import os
import json
from kaldiio import ReadHelper
from copy import deepcopy
from scipy import signal

PHN = '###PHN###'
BLANK = '###BLANK###'
SIL = 'SIL' 
IGNORED_TOKENS = ['GARBAGE', '+BREATH+', '+LAUGH+', '+NOISE+']  

def collate_fn_unsegment_spoken_word(batch):
    audios = [t[0] for t in batch]
    span_ids = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    input_masks = [t[3] for t in batch]
    span_masks = [t[4] for t in batch]
    phone_nums = [t[5] for t in batch]
    segment_nums = [t[6] for t in batch] 
    indices = [t[7] for t in batch]

    if isinstance(audios[0], list):
        audios = [
            torch.nn.utils.rnn.pad_sequence(audio)\
            for audio in audios
        ]
        span_ids = [
            torch.nn.utils.rnn.pad_sequence(span_id)
            for span_id in span_ids
        ]
        span_masks = [
            torch.nn.utils.rnn.pad_sequence(span_mask)
            for span_mask in span_masks
        ]
        audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        span_ids = torch.nn.utils.rnn.pad_sequence(span_ids, batch_first=True)
        span_masks = torch.nn.utils.rnn.pad_sequence(span_masks, batch_first=True)
        audios = audios.permute(0, 2, 1, 3)
        span_ids = span_ids.permute(0, 2, 1)
        span_masks = span_masks.permute(0, 2, 1)
    else:
        audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)    
        span_ids = torch.nn.utils.rnn.pad_sequence(span_ids, batch_first=True)
    labels = torch.stack(labels)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True)
    span_masks = torch.nn.utils.rnn.pad_sequence(span_masks, batch_first=True)
    return audios, span_ids, labels, input_masks, span_masks, phone_nums, segment_nums, indices

class UnsegmentedSpokenWordDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, 
        preprocessor, split,
        splits = {
            'train': ['train-clean-100', 'train-clean-360'],
            'validation': ['dev-clean'],
            'test': ['dev-clean'],     
        },
        augment=False,
        audio_feature='cpc',
        phone_label='predicted',
        ds_method='average',
        sample_rate=16000,
        min_class_size=50,
        min_length=-1,
        max_length=40,
        n_positives=0, 
        debug=False
    ):
        if debug:
            splits['train'] = [splits['train'][0]]
        self.preprocessor = preprocessor
        self.splits = splits[split]
        self.data_path = data_path
        self.ds_method = ds_method
        self.sample_rate = sample_rate
        self.n_positives = n_positives
        self.min_length = min_length
        self.max_length = max_length
        self.debug = debug
    
        data = []
        for sp in self.splits:
            # Load data paths to audio and visual features
            examples = load_data_split(
                preprocessor.dataset_name,
                data_path, sp,
                min_class_size=min_class_size,
                audio_feature=audio_feature,
                phone_label=phone_label,
                debug=debug) 
            data.extend(examples)
            print('Number of {} audio files = {}'.format(split, len(examples)))

        audio = [example['audio'] for example in data]
        text = [example['text'] for example in data]
        spans = [example['spans'] for example in data] # best segments
        segments = [example['segments'] for example in data]
        phonemes = [example['phonemes'] for example in data]
        self.dataset = [list(item) for item in zip(audio, text, spans, segments, phonemes)]
        self.audio_feature_type = audio_feature

    def __getitem__(self, idx):        
        audio_inputs,\
        span_ids, label,\
        input_mask, span_mask,\
        phoneme_num, segment_num = self.load_audio(idx)
        word_labels = self.preprocessor.to_word_index([label])
        if self.n_positives > 0: # Draw positive examples
            audio_inputs = [audio_inputs]
            span_ids = [span_ids]
            span_mask = [span_mask]
            pos_idxs = self.sample_positives(idx, label, self.n_positives)
            for pos_idx in pos_idxs:
                outputs = self.load_audio(pos_idx)
                audio_inputs.append(outputs[0].clone())
                span_ids.append(outputs[1].clone())
                span_mask.append(outputs[4].clone()) 
        return audio_inputs, span_ids, word_labels, input_mask, span_mask, phoneme_num, segment_num, idx 

    def sample_positives(self, idx, label, n):
        random_idxs = np.random.permutation(len(self.dataset))
        pos_idxs = []
        for i in random_idxs:
            if i == idx:
                continue
            if self.dataset[i][1] == label:
                pos_idxs.append(i)
            if len(pos_idxs) == n:
                break
        if not len(pos_idxs):
            print(f'No positive examples found for class {self.dataset[idx][1]}')
            return [idx]*n

        if len(pos_idxs) < n:
            return pos_idxs+[pos_idxs[-1]]*(n-len(pos_idxs))
        return pos_idxs

    def load_audio(self, idx):
        audio_file, label,\
        spans, segments,\
        phonemes = self.dataset[idx]

        if self.audio_feature_type in ["mfcc", "bnf+mfcc"]:
            audio, _ = torchaudio.load(audio_file)
            inputs = self.audio_transforms(audio)
            inputs = inputs.squeeze(0)
        elif self.audio_feature_type in ["cpc", "cpc_big"]:
            if audio_file.split('.')[-1] == "txt":
                audio = np.loadtxt(audio_file)
            else:
                with ReadHelper(f"ark: gunzip -c {audio_file} |") as ark_f:
                    for k, audio in ark_f:
                        continue
            inputs = torch.FloatTensor(audio)
        elif self.audio_feature_type in ["bnf", "bnf+cpc"]:
            if audio_file.split('.')[-1] == "txt":
                audio = np.loadtxt(audio_file)
            else:
                with ReadHelper(f"ark: gunzip -c {audio_file} |") as ark_f:
                    for k, audio in ark_f:
                        continue
            if self.audio_feature_type == "bnf+cpc":
                cpc_feat = np.loadtxt(audio_file.replace("bnf", "cpc"))
                feat_len = min(audio.shape[0], cpc_feat.shape[0])
                audio = np.concatenate([audio[:feat_len], cpc_feat[:feat_len]], axis=-1)
            inputs = torch.FloatTensor(audio)
        elif self.audio_feature_type in ['vq-wav2vec', 'wav2vec', 'wav2vec2']:
            audio, _ = torchaudio.load(audio_file)
            inputs = audio.squeeze(0)
        else: Exception(f"Audio feature type {self.audio_feature_type} not supported")
        input_mask = torch.ones(inputs.size(0))

        inputs, input_mask = self.segment(inputs, segments, method=self.ds_method)
        span_ids = torch.LongTensor(
            [self.get_span_id(*span) for span in spans]
        )
        span_mask = torch.ones(len(span_ids))
        segment_num = len(segments)
        phoneme_num = len(phonemes)
        return inputs, span_ids, label, input_mask, span_mask, phoneme_num, segment_num

    def segment(self, feat, seed_segments, method='average'):
        sfeats = []
        segment_mask = []
        word_begin = seed_segments[0]['begin']
        n = len(seed_segments)
        for end_idx in range(n):
            for l in range(end_idx+1):
                begin_idx = end_idx - l
                begin = sec_to_frame(seed_segments[begin_idx]['begin'] - word_begin)
                end = sec_to_frame(seed_segments[end_idx]['end'] - word_begin)

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
        word_begin = segments[0]['begin']
        dur = segments[-1]['end'] - word_begin
        nframes = sec_to_frame(dur) + 1
        feat = torch.zeros((nframes, *sfeat.size()[1:]))
        for i, segment in enumerate(segments):
            if segment['text'] == SIL:
                continue
            begin = sec_to_frame(segment['begin']-word_begin)
            end = sec_to_frame(segment['end']-word_begin)
            if begin >= feat.size(0):
                continue 
            if end == begin:
                end += 1
            feat[begin:end] = sfeat[i]
        return feat.squeeze(-1)

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

class UnsegmentedSpokenWordPreprocessor:
    def __init__(
        self,
        dataset_name,
        data_path,
        num_features,
        splits = {
            'train': ['train-clean-100', 'train-clean-360'],
            'validation': ['dev-clean'],
            'test': ['dev-clean']
        },
        tokens_path=None,
        lexicon_path=None,
        use_words=False,
        prepend_wordsep=False,
        audio_feature='mfcc',
        phone_label='predicted',
        sample_rate=16000,
        min_class_size=50,
        ignore_index=-100,
        use_blank=True,
        min_length=1,
        max_length=40,
        debug=False,      
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.num_features = num_features
        self.ignore_index = ignore_index
        self.min_class_size = min_class_size
        self.use_blank = use_blank
        metadata_file = os.path.join(data_path, f'{dataset_name}.json')
        
        if debug:
            splits['train'] = [splits['train'][0]]
        data = []
        for split_type, spl in splits.items():
            if split_type == 'test_oos':
                continue
            for sp in spl:
                data.extend(
                    load_data_split(
                        dataset_name,
                        data_path, sp,
                        audio_feature=audio_feature,
                        phone_label=phone_label,
                        min_class_size=self.min_class_size,
                        debug=debug)
                )
        visual_words = set()
        tokens = set()
        for ex in data:
            visual_words.add(ex['text'])
            for phn in ex['phonemes']:
                if phone_label == 'groundtruth' and not 'phoneme' in phn['text']:
                   phn['text'] = re.sub(r'[0-9]', '', phn['text'])
                tokens.add(phn['text'])
        self.tokens = sorted(tokens)
        self.visual_words = sorted(visual_words)
        if self.use_blank:
            self.tokens = [BLANK]+self.tokens
            self.visual_words = [BLANK]+self.visual_words
        self.tokens_to_index = {t:i for i, t in enumerate(self.tokens)}
        self.words_to_index = {t:i for i, t in enumerate(self.visual_words)}
        print(f'Preprocessor: number of phone classes: {self.num_tokens}')
        print(f'Preprocessor: number of visual word classes: {self.num_visual_words}')

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

def load_data_split(dataset_name,
                    data_path, split,
                    audio_feature='cpc',
                    phone_label='predicted',
                    min_class_size=50,
                    max_keep_size=1000,
                    debug=False):
    '''
    Returns:
        examples: a list of dicts with key-value pairs
            {
                ``audio``: str, filename of audio,
                ``text``: str,
                ``spans``: a list of [int, int] lists
                ``phonemes``: a list of dicts with key-value pairs
                { 
                    ``begin``: int, index of the beginning seed boundary
                    ``end``: int, index of the end seed boundary
                },
                ``segments``: list of dicts with key-value pairs
                { 
                    ``begin``: float,
                    ``end``: float,
                    ``text``: str,
                },
                ``phoneme_num``: int,
                ``frame_num`` : int
            }
    '''
    examples = []
    for word_file in os.listdir(data_path):
        if word_file.split('.')[-1] != 'json':
            continue

        label_counts = dict()
        with open(os.path.join(data_path, word_file), 'r') as f:
            count = 0
            for line in f:
                if debug and len(examples) >= 10:
                    break
                word_dict = json.loads(line.rstrip('\n'))
                audio_id = word_dict['audio_id']
                word_id = word_dict['word_id']

                label = WordNetLemmatizer().lemmatize(word_dict['label'].lower())
                if not label in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1
                if label_counts[label] > max_keep_size:
                    continue

                if word_dict['split'] != split:
                    continue
                
                if audio_feature in ['cpc', 'cpc_big']:
                    for phn_dict in word_dict['phonemes']:
                        audio_path = os.path.join(data_path, f'../{dataset_name}_{audio_feature}_txt/{audio_id}_{word_id}.txt')
                        if not os.path.exists(audio_path):
                            audio_path = os.path.join(data_path, f'../{dataset_name}_{audio_feature}/{audio_id}_{word_id}.ark.gz')
                        if not os.path.exists(audio_path):
                            word_id = int(word_id)
                            audio_file = f'{audio_id}_{word_id:04d}.txt'
                            audio_path = os.path.join(data_path, f'../{dataset_name}_{audio_feature}_txt', audio_file)
                elif audio_feature in ['bnf', 'bnf+cpc']:
                    audio_file = f'{audio_id}_{word_id}.txt'
                    audio_path = os.path.join(data_path, f'../{dataset_name}_bnf_txt', audio_file)
                else:
                    raise NotImplementedError(f'Audio feature type {audio_feature} not implemented')
                
                # Extract seed segments
                phonemes = word_dict['phonemes']
                if 'children' in phonemes:
                    phonemes = [phn for phn in phonemes['children'] if phn['text'] != SIL]
                if len(phonemes) == 0:
                    continue
                 
                noisy = False
                for phn_idx, phn in enumerate(phonemes):
                    if not 'phoneme' in phn['text']:
                        phn['text'] = re.sub(r'[0-9]', '', phn['text']) 
                        if phn['text'] in IGNORED_TOKENS or (phn['text'][0] == '+'):
                            noisy = True
                            break
                if noisy:
                    continue

                dur = round(phonemes[-1]['end'] - phonemes[0]['begin'])
                segments = None 
                if phone_label == 'predicted':
                    segments = [phn for phn in word_dict['predicted_segments'] if phn['text'] != SIL]
                    if not len(segments):
                        continue
                elif phone_label == 'predicted_wav2vec2':
                    segments = [phn for phn in word_dict['predicted_segments_wav2vec2'] if phn['text'] != SIL]
                    if not len(segments):
                        continue
                elif phone_label == 'groundtruth':
                    if debug:
                        print('Warning: ground truth segments is not allowed in this setting!') 
                    segments = word_dict['phonemes'] 
                    if 'children' in phonemes:
                        segments = [phn for phn in phonemes['children'] if phn['text'] != SIL]
                    word_begin = segments[0]['begin']
                    segments = [
                        {'begin': segment['begin'] - word_begin, 'end': segment['end'] - word_begin} 
                        for segment in segments 
                    ]
                if not len(segments):
                    continue
                length = len(segments)

                # Extract number of phonemes 
                phoneme_num = len(word_dict['phonemes']) 

                # Initialize spans to be all seed segments
                spans = []
                for span_idx, segment in enumerate(segments):
                    spans.append([span_idx, span_idx])
                examples.append(
                    {
                        'audio': audio_path,
                        'text': label,
                        'spans': spans,
                        'segments': segments,
                        'phonemes': phonemes
                    }
                ) 
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
