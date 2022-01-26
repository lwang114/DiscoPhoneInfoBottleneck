import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import fairseq
import argparse
import sys
import os
import shutil
import json
import time
import numpy as np
from kaldiio import WriteHelper
from copy import deepcopy
from pyhocon import ConfigFactory
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from utils.utils import cuda, str2bool
from model import InfoQuantizer, masked_kl_div
from datasets.datasets import return_data
from utils.evaluate import compute_accuracy, compute_token_f1, compute_boundary_f1, compute_edit_distance

EPS = 1e-10
BLANK = "###BLANK###"

class Solver(object):

  def __init__(self, config):
    self.config = config

    self.cuda = torch.cuda.is_available()
    self.epoch = config.epoch
    self.batch_size = config.batch_size
    self.lr = config.lr
    self.n_layers = config.get('num_layers', 3)
    self.eps = 1e-9
    self.K = config.K
    self.global_iter = 0
    self.global_epoch = 0
    self.audio_feature = config.audio_feature
    self.image_feature = config.image_feature
    self.debug = config.debug
    self.dataset = config.dataset
    
    self.ckpt_dir = Path(config.ckpt_dir)
    if not self.ckpt_dir.exists(): 
      self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(self.ckpt_dir / f'results_file_{self.config.seed}.txt', 'a') as f:
      f.write(str(config)+'\n')

    self.get_feature_config(config)
    self.get_dataset_config(config)
    self.get_model_config(config)
    self.get_optim_config(config)

    self.load_ckpt = config.load_ckpt
    if not self.debug and (self.load_ckpt or config.mode in ['test', 'test_oos', 'cluster', 'test_zerospeech']): 
      self.load_checkpoint(f'best_acc_{self.config.seed}.tar')
    
    # History
    self.history = dict()
    self.history['token_result']=[0., 0., 0.]
    self.history['oos_token_result']=[0., 0., 0.]
    self.history['word_acc']=0. 
    self.history['loss']=0.
    self.history['epoch']=0
    self.history['iter']=0

  def get_dataset_config(self, config):
    self.data_loader = return_data(config)
    self.oos_dataset_name = config.get('oos_dataset', None) 
    if self.oos_dataset_name:
      oos_config = deepcopy(config)
      oos_config['dataset'] = oos_config['oos_dataset']
      oos_config['dset_dir'] = oos_config['oos_dset_dir']
      oos_config['splits'] = {'train': oos_config['splits']['test_oos'],
                              'test': oos_config['splits']['test_oos']} 
      oos_data_loader = return_data(oos_config)
      self.data_loader['test_oos'] = oos_data_loader['test']
    self.dataset_name = config.dataset
    self.ignore_index = config.get('ignore_index', -100)

    self.n_visual_class = config.get('n_visual_class', None)
    if not self.n_visual_class:
      self.n_visual_class = self.data_loader['train']\
                            .dataset.preprocessor.num_visual_words

    self.n_phone_class = self.data_loader['train'].dataset.preprocessor.num_tokens
    self.visual_words = self.data_loader['train'].dataset.preprocessor.visual_words
    self.phone_set = self.data_loader['train'].dataset.preprocessor.tokens
    self.phoneme_itos = None
    if config.get('phoneme_itos', None):
      self.phoneme_itos = json.load(open(config['phoneme_itos']))
    self.n_clusters = config.get("n_clusters", self.n_phone_class)
    print(f'Number of visual label classes = {self.n_visual_class}')
    print(f'Number of phone classes = {self.n_phone_class}')
    print(f'Number of clusters = {self.n_clusters}')

  def get_feature_config(self, config):
    if config.audio_feature == 'mfcc':
      self.audio_feature_net = None
      self.input_size = 80
      self.hop_len_ms = 10
    elif config.audio_feature == 'wav2vec2':
      self.audio_feature_net = cuda(fairseq.checkpoint_utils.load_model_ensemble_and_task([config.wav2vec_path])[0][0],
                                    self.cuda)
      for p in self.audio_feature_net.parameters():
        p.requires_grad = False
      self.input_size = 512
      self.hop_len_ms = 20 
    elif config.audio_feature == 'cpc':
      self.audio_feature_net = None
      self.input_size = 256
      self.hop_len_ms = 10
    elif config.audio_feature == 'bnf':
      self.audio_feature_net = None
      self.input_size = 40
      self.hop_len_ms = 10
    elif config.audio_feature == 'bnf+cpc':
      self.audio_feature_net = None
      self.input_size = 296
      self.hop_len_ms = 10
    else:
      raise ValueError(f"Feature type {config.audio_feature} not supported")

    if config.downsample_method == 'resample':
      self.input_size *= 5
    self.embed_type = self.config.get('embed_type', 'proba')
    print(f'Embedding type: {self.embed_type}')

  def get_model_config(self, config):
    self.use_conv = config.get('use_conv', False) 
    self.conv_width = config.get('conv_width', 5)

    self.audio_net = cuda(InfoQuantizer(in_channels=self.input_size,
                                        channels=self.K,
                                        n_embeddings=self.n_clusters,
                                        z_dim=self.n_visual_class,
                                        use_conv=self.use_conv,
                                        conv_width=self.conv_width), self.cuda)

  def get_optim_config(self, config):
    trainables = [p for p in self.audio_net.parameters()]
    optim_type = config.get('optim', 'adam')
    if optim_type == 'sgd':
      self.optim = optim.SGD(trainables, 
                             momentum=0.9,
                             lr=self.lr)
    else:
      self.optim = optim.Adam(trainables,
                              lr=self.lr, betas=(0.5, 0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)
    
  def train(self, save_embedding=False):
    self.set_mode('train')
    preprocessor = self.data_loader['train'].dataset.preprocessor
    total_loss = 0.
    total_step = 0.
    total_word_loss = 0.
    total_phone_loss = 0.
    total_pair_loss = 0.
    n_merges = 0

    for e in range(self.epoch):
      if e > 1 and self.debug:
        break
      self.global_epoch += 1

      progress = tqdm(total=len(self.data_loader['train']), ncols=80, desc=f'Training Epoch {e}')
      for idx, batch in enumerate(self.data_loader['train']):
        if idx > 2 and self.debug:
          break
        self.global_iter += 1
        
        if self.config.n_positives > 0:
            assert(batch[0].ndim == 4)
            audios = batch[0][:, 0]
            pos_audios = batch[0][:, 1:]
            spans_ids = batch[1][:, 0]
            pos_spans_ids = batch[1][:, 1:]
            span_masks = batch[4][:, 0]
            pos_span_masks = batch[4][:, 1:]
            x = torch.stack([audios[i, span_ids] for i, span_ids in enumerate(spans_ids)])
            x_pos = torch.stack(
                [torch.stack(
                    [pos_audios[i, pos_id, pos_span_id] 
                        for pos_id, pos_span_id in enumerate(pos_span_ids)]
                ) 
                for i, pos_span_ids in enumerate(pos_spans_ids)]
            )
            x_pos = cuda(x_pos, self.cuda)
            pos_span_masks = cuda(pos_span_masks, self.cuda)
        else:
            audios = batch[0]
            spans_ids = batch[1]
            span_masks = batch[4]
            # (batch size, segment num, feat dim)
            x = torch.stack([audios[i, span_ids] for i, span_ids in enumerate(spans_ids)])

        x = cuda(x, self.cuda)

        word_labels = batch[2].squeeze(-1)        
        data_indices = batch[-1]

        # (batch size,)
        word_labels = cuda(word_labels, self.cuda)
        
        # (batch size, segment num)
        span_masks = cuda(span_masks, self.cuda)

        # (batch size, segment num, n visual class)
        word_logits, quantized, phone_loss = self.audio_net(x, masks=span_masks)

        # (batch size * segment num, n visual class)
        segment_word_logits = (word_logits\
                              * span_masks.unsqueeze(-1)).sum(-2)
        word_loss = F.cross_entropy(segment_word_logits,
                                    word_labels,
                                    ignore_index=self.ignore_index)
        if torch.isnan(word_loss):
            print(f'word loss is nan for example of size {x.size()}, word_logits of size {word_logits.size()}')
            print(torch.any(torch.isnan(x)))
        loss = phone_loss + word_loss
        if self.config.n_positives > 0:
            word_logits = word_logits * span_masks.unsqueeze(-1)
            pos_word_logits, _, _ = self.audio_net(x_pos, masks=pos_span_masks)
            pos_word_logits = pos_word_logits * pos_span_masks.unsqueeze(-1)
            pair_loss = F.mse_loss(word_logits.unsqueeze(1), pos_word_logits)
            loss += pair_loss
            total_pair_loss += pair_loss.cpu().detach().numpy()
        total_phone_loss += phone_loss.cpu().detach().numpy()
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.
        
        if loss == 0:
          continue
        self.optim.zero_grad()
        loss.backward()        
        self.optim.step()

        if (self.global_iter-1) % 1000 == 0:
          avg_loss = total_loss / total_step
          avg_phone_loss = total_phone_loss / total_step
          # print(f'Itr {self.global_iter:d}\tAvg Loss (Total Loss):{avg_loss:.2f} ({total_loss:.2f})\tAvg Phone Loss:{avg_phone_loss:.2f}')
        progress.update(1)
      progress.close()
      avg_loss = total_loss / total_step
      avg_phone_loss = total_phone_loss / total_step
      avg_pair_loss = total_pair_loss / total_step
      print(f'Epoch {self.global_epoch}\tTraining Loss: {avg_loss:.3f}\tTraining Phone Loss: {avg_phone_loss:.3f}\tTraining Pairwise Phone Loss: {avg_pair_loss:.3f}')

      # Refine segments
      if (self.config.phone_label != 'groundtruth') and (self.debug or self.global_epoch >= 3):
        self.train_segment('train')
      
      if (self.global_epoch % 2) == 0:
        self.scheduler.step()
      self.test(save_embedding=save_embedding)
      if self.oos_dataset_name:
        self.test_out_of_sample(save_embedding=save_embedding)

  @torch.no_grad()
  def train_segment(self, split):
    dataset = self.data_loader[split].dataset
    
    segment_dir = Path(self.ckpt_dir / f'predicted_segmentations/{split}')
    if not segment_dir.exists():
      segment_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(self.data_loader[split]), ncols=80, desc=f'Segmenting {split} set')
    for batch in self.data_loader[split]:
      audios = cuda(batch[0], self.cuda)
      audio_masks = cuda(batch[3], self.cuda)
      word_labels = [None]*len(audios)
      if split in ['train', 'test']:
        word_labels = batch[2]
      span_masks = cuda(batch[4], self.cuda)
      phoneme_nums = batch[5]
      segment_nums = batch[6]
      data_indices = batch[-1]

      pred_peaks = []
      gold_peaks = []
      word_logits, quantized, phone_indices = self.audio_net.encode(audios, masks=audio_masks)
      for idx, global_idx in enumerate(data_indices):
        audio_path = dataset.dataset[global_idx][0]
        audio_id = os.path.basename(audio_path).split('.')[0]
        new_spans = self.viterbi_segment(
            word_logits[idx], 
            quantized[idx],
            audio_masks[idx],
            phoneme_nums[idx],
            segment_nums[idx],
            word_label=word_labels[idx]
        )
        dataset.update_spans(global_idx, new_spans)
        new_segments = dataset.span_to_segment(global_idx)
        seed_segments = dataset.dataset[global_idx][3]
        phonemes = dataset.dataset[global_idx][4]        
        utt_begin = phonemes[0]['begin'] 
        gold_peak = [round(phn['end']-utt_begin,3) for phn in phonemes]
        gold_peak_str = ' '.join(['0']+[str(g) for g in gold_peak])
        pred_peak = [segment['end'] for segment in new_segments]
        pred_peak_str = ' '.join(['0']+[str(p) for p in pred_peak])
        seed_peak_str = ' '.join([str(s['end']) for s in seed_segments])
        with open(segment_dir / f'{audio_id}.txt', 'w') as f:
          f.write(f'Pred: {pred_peak_str}\nGold: {gold_peak_str}\nSeed: {seed_peak_str}')
        gold_peaks.append(gold_peak[:-1])
        pred_peaks.append(pred_peak[:-1])
        if self.debug:
          print('seed_segments: ', seed_peak_str) # XXX
          print('Gold peaks: ', gold_peak)
          print('Pred peaks: ', pred_peak)
      progress.update(1)
    progress.close()
    
    boundary_prec, boundary_rec, boundary_f1 = compute_boundary_f1(pred_peaks, gold_peaks)
    info = f'{split} set result\tBoundary precision: {boundary_prec*100:.2f}\tBoundary recall: {boundary_rec*100:.2f}\tBoundary F1: {boundary_f1*100:.2f}'
    with open(self.ckpt_dir / f'results_file_{self.config.seed}.txt', 'a') as f_result:
        f_result.write(info+'\n')
    print(info)

  @torch.no_grad()
  def test(self, save_embedding=False, out_prefix='predictions'): 
    self.set_mode('eval')
    testset = self.data_loader['test'].dataset
    preprocessor = testset.preprocessor

    total_loss = 0.
    total_step = 0.

    pred_word_labels = []
    pred_word_labels_quantized = []
    gold_word_labels = []
    embeds = dict()
    embed_labels = dict()

    gold_phone_file = os.path.join(testset.data_path, f'{testset.splits[0]}/{testset.splits[0]}_nonoverlap.item')
    word_readable_f = open(self.ckpt_dir.joinpath(f'{out_prefix}_visual_word.readable'), 'w') 
    phone_file = self.ckpt_dir.joinpath(f'{out_prefix}_phoneme.txt')
    embed_file = self.ckpt_dir.joinpath(f'{out_prefix}_embeddings.npz')
    embed_label_file = self.ckpt_dir.joinpath(f'{out_prefix}_embedding_labels.json')
    print(f'gold_phone_file: {gold_phone_file}\npred_phone_file: {phone_file}')
    phone_f = open(phone_file, 'w')
    B = 0
    progress = tqdm(total=len(self.data_loader['test']), ncols=80, desc=f'Testing on test set')
    for b_idx, batch in enumerate(self.data_loader['test']):
      if b_idx > 2 and self.debug:
        break
      audios = batch[0]
      spans_ids = batch[1]
      if self.config.n_positives > 0:
        assert(batch[0].ndim == 4)
        audios = batch[0][:, 0]
        spans_ids = batch[1][:, 0]
        span_masks = batch[4][:, 0]
      else:
        audios = batch[0]
        spans_ids = batch[1]
        span_masks = batch[4]
      
      # (batch size, max segment num, feat dim)
      x = torch.stack([audios[i, span_ids] for i, span_ids in enumerate(spans_ids)])
      x = cuda(x, self.cuda)
          
      word_labels = batch[2].squeeze(-1)
      data_indices = batch[-1]
      if b_idx == 0: 
        B = audios.size(0)

      # (batch size,)
      word_labels = cuda(word_labels, self.cuda)

      # (batch size, max segment num) or (batch size, max segment num, max segment len)
      span_masks = cuda(span_masks, self.cuda)
        
      # (batch size, max segment num, n visual class)
      word_logits, quantized, phone_loss = self.audio_net(x, masks=span_masks)
      segment_word_logits = (word_logits\
                            * span_masks.unsqueeze(-1)).sum(-2)

      word_loss = F.cross_entropy(segment_word_logits,
                             word_labels,
                             ignore_index=self.ignore_index)
      loss = phone_loss + word_loss
      total_loss += loss.cpu().detach().numpy()
      total_step += 1.
      
      _, _, phone_indices = self.audio_net.encode(x, masks=span_masks)
      for idx, global_idx in enumerate(data_indices):
        audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0].split('.')[0]
        pred_phonemes = testset.span_to_segment(global_idx)
        pred_phone_label = phone_indices[idx]
        pred_phone_label = testset.unsegment(phone_indices[idx] + 1, pred_phonemes).long()

        embed = F.softmax(word_logits[idx, :len(pred_phonemes)], dim=-1)
        if save_embedding and global_idx < 1000:
          embed_id = f'{audio_id}_{global_idx}'
          embeds[embed_id] = embed.cpu().numpy()
          embed_labels[embed_id] = {'phoneme_text': [s['text'] for s in pred_phonemes],
                                    'word_text': [word_labels[idx].detach().cpu().numpy().tolist()]*len(pred_phonemes)}

        if int(self.hop_len_ms / 10) * self.audio_net.ds_ratio > 1:
          us_ratio = int(self.hop_len_ms / 10) * self.audio_net.ds_ratio
          pred_phone_label = pred_phone_label.unsqueeze(-1)\
                             .expand(-1, us_ratio).flatten()

        pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
        pred_phone_names = ','.join([str(p) for p in pred_phone_label_list])
        phone_f.write(f'{audio_id} {pred_phone_names}\n')
         
        gold_word_label = word_labels[idx].cpu().detach().numpy().tolist()
        pred_word_label = segment_word_logits[idx].max(-1)[1].cpu().detach().numpy().tolist()
        pred_word_label_quantized = quantized[idx, :len(pred_phonemes)].prod(-2).max(-1)[1].cpu().detach().numpy().tolist()
         
        gold_word_labels.append(gold_word_label)
        pred_word_labels.append(pred_word_label)
        pred_word_labels_quantized.append(pred_word_label_quantized)
        pred_word_name = preprocessor.to_word_text([pred_word_label])[0]
        pred_word_name_quantized = '' # XXX preprocessor.to_word_text([pred_word_label_quantized])[0]
        gold_word_name = preprocessor.to_word_text([gold_word_label])[0]
        word_readable_f.write(f'Utterance id: {audio_id}\n'
                              f'Gold word label: {gold_word_name}\n'
                              f'Pred word label: {pred_word_name}\n'
                              f'Pred word label by quantizer: {pred_word_name_quantized}\n\n')
      progress.update(1)
    progress.close()
    phone_f.close()
    word_readable_f.close()
    if save_embedding:
      np.savez(embed_file, **embeds)
      json.dump(embed_labels, open(embed_label_file, 'w'), indent=2)

    avg_loss = total_loss / total_step
    # Refine segments
    if (self.config.phone_label != 'groundtruth') and (self.debug or self.global_epoch >= 3):
      self.train_segment('test')

    # Compute word accuracy and word token F1
    print('[TEST RESULT]')
    word_acc = compute_accuracy(gold_word_labels, pred_word_labels)
    word_prec,\
    word_rec,\
    word_f1, _ = precision_recall_fscore_support(np.asarray(gold_word_labels),
                                                 np.asarray(pred_word_labels),
                                                 average='macro',
                                                 zero_division=0)

    word_prec_quantized,\
    word_rec_quantized,\
    word_f1_quantized, _ = precision_recall_fscore_support(np.asarray(gold_word_labels),
                                                           np.asarray(pred_word_labels_quantized),
                                                           average='macro',
                                                           zero_division=0) 

    token_f1,\
    token_prec,\
    token_recall = compute_token_f1(phone_file,
                                    gold_phone_file,
                                    self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))
    info = f'Epoch {self.global_epoch}\tLoss: {avg_loss:.4f}\n'\
           f'WER: {1-word_acc:.3f}\tWord Acc.: {word_acc:.3f}\n'\
           f'Word Precision: {word_prec:.3f}\tWord Recall: {word_rec:.3f}\tWord F1: {word_f1:.3f}\n'\
           f'(By Quantizer) Word Precision: {word_prec_quantized:.3f}\tWord Recall: {word_rec_quantized:.3f}\tWord F1: {word_f1_quantized:.3f}\n'\
           f'Token Precision: {token_prec:.3f}\tToken Recall: {token_recall:.3f}\tToken F1: {token_f1:.3f}\n'
    print(info) 

    save_path = self.ckpt_dir.joinpath(f'results_file_{self.config.seed}.txt')
    if not self.debug:
      with open(save_path, 'a') as file:
        file.write(info)

    if not self.debug and self.history['token_result'][-1] < token_f1:
      self.history['token_result'] = [token_prec, token_recall, token_f1]
      self.history['word_acc'] = word_acc
      self.history['loss'] = avg_loss
      self.history['iter'] = self.global_iter
      self.history['epoch'] = self.global_epoch
      self.save_checkpoint(f'best_acc_{self.config.seed}.tar')
      best_phone_file = self.ckpt_dir.joinpath(f'outputs_quantized_{self.config.seed}.txt')
      shutil.copyfile(phone_file, best_phone_file)
    self.set_mode('train') 

  @torch.no_grad()
  def test_out_of_sample(self, save_embedding=False):
    self.set_mode('eval')
    test_loader = self.data_loader['test_oos']
    testset = test_loader.dataset
    batch_size = test_loader.batch_size

    splits = '_'.join(testset.splits)
    phone_file = self.ckpt_dir.joinpath(f'{self.oos_dataset_name}_{splits}_outputs_quantized_{self.config.seed}.txt')
    embed_file = self.ckpt_dir.joinpath(f'{self.oos_dataset_name}_{splits}_embeddings.npz')
    embed_label_file = self.ckpt_dir.joinpath(f'{self.oos_dataset_name}_{splits}_embedding_labels.json')
    split = testset.splits[0]
    embed_zrc_dir = self.ckpt_dir.joinpath(f'outputs_{self.oos_dataset_name}_{self.embed_type}/phonetic/{splits}')
    if not os.path.exists(embed_zrc_dir):
      print(f'Create directory for embeddings: {embed_zrc_dir}')
      os.makedirs(embed_zrc_dir)
    
    embeds = dict()
    embed_labels = dict()

    progress = tqdm(total=len(self.data_loader['test_oos']), ncols=80, desc=f'Testing on test oos set')
    phone_f = open(phone_file, 'w')
    for b_idx, batch in tqdm(enumerate(test_loader)):
      audios = batch[0]
      spans_ids = batch[1]
      span_masks = batch[4]
      
      x = torch.stack([audios[i, span_ids] for i, span_ids in enumerate(spans_ids)])
      x = cuda(x, self.cuda)
      span_masks = cuda(span_masks, self.cuda)

      word_logits, quantized_word_proba, phone_indices = self.audio_net.encode(x, masks=span_masks) 
      B = phone_indices.size(0)
      for idx in range(B):
        global_idx = b_idx * batch_size + idx
        audio_path = test_loader.dataset.dataset[global_idx][0]
        pred_phonemes = testset.span_to_segment(global_idx)
        audio_id = os.path.basename(audio_path).split('.')[0]
        embedding = F.softmax(word_logits[idx, :len(pred_phonemes)], dim=-1)
        
        if save_embedding:
          embed_id = f'{audio_id}_{global_idx}' 
          # XXX embeds[embed_id] = embedding.detach().cpu().numpy()
          embed_labels[embed_id] = {'phoneme_text': [s['text'] for s in pred_phonemes],
                                    'word_text': [BLANK]*len(pred_phonemes)}
          embedding_frame = testset.unsegment(embedding, pred_phonemes)
          embed_path = embed_zrc_dir.joinpath(f'{audio_id}.ark.gz')
          with WriteHelper(f'ark:| gzip -c > {embed_path}') as writer:
            writer('arr_0', embedding_frame.detach().cpu().numpy())

        pred_phone_label = phone_indices[idx]
        pred_phone_label = testset.unsegment(phone_indices[idx] + 1, pred_phonemes).long()

        us_ratio = int(self.audio_net.ds_ratio * (self.hop_len_ms // 10))
        if us_ratio > 1:
          pred_phone_label = pred_phone_label.unsqueeze(-1)\
                             .expand(-1, us_ratio).flatten()
           
        pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
        pred_phone_names = ','.join([str(phn_idx) for phn_idx in pred_phone_label_list])
        phone_f.write(f'{audio_id} {pred_phone_names}\n')  
      progress.update(1)
    progress.close()
    phone_f.close()

    # Save embeddings
    if save_embedding:
      json.dump(embed_labels, open(embed_label_file, 'w') ,indent=2)

    # Refine segments
    if (self.config.phone_label != 'groundtruth') and (self.debug or self.global_epoch >= 3):
      self.train_segment('test_oos')

    # Evaluation with token F1
    gold_phone_files = [os.path.join(testset.data_path, f'{split}/{split}_nonoverlap.item') for split in testset.splits]
    print(f'gold_phone_files: {gold_phone_files}\npred_phone_file: {phone_file}')
    token_f1,\
    token_prec,\
    token_recall = compute_token_f1(phone_file,
                                    gold_phone_files,
                                    self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))

    print('[OOS TEST RESULT]')
    info = f'Out-of-Sample Dataset: {self.oos_dataset_name}\n'\
           f'Token Precision: {token_prec:.4f}\tToken Recall: {token_recall:.4f}\tToken F1: {token_f1:.4f}\n'

    save_path = os.path.join(self.ckpt_dir, f'results_file_{self.config.seed}.txt')
    with open(save_path, 'a') as f:
      f.write(info)
    print(info)
     
    if not self.debug and self.history['oos_token_result'][-1] < token_f1:
      best_phone_file = self.ckpt_dir.joinpath(f'{self.oos_dataset_name}_outputs_quantized_{self.config.seed}.txt')
      shutil.copyfile(phone_file, best_phone_file)
      self.history['oos_token_result'] = [token_prec, token_recall, token_f1]
      self.save_checkpoint(f'best_acc_oos_{self.config.seed}.tar')
    self.set_mode('train')

  @torch.no_grad()
  def test_zerospeech(self, save_embedding=False): # TODO
    self.set_mode('eval')
    test_loader = self.data_loader['test_oos']
    testset = test_loader.dataset
    batch_size = test_loader.batch_size

    zrc_dir = Path(self.ckpt_dir / f'outputs_zerospeech2021_{self.embed_type}')
    splits = '_'.join(testset.splits)
    task = self.config['oos_dset_dir'].split('/')[-1]
    print(f'{splits} for zerospeech task {task}')

    phone_file = zrc_dir / f'{task}/{splits}/quantized_outputs.txt'
    print(phone_file)
    split = testset.splits[0]
   
    if not zrc_dir.exists():
      os.makedirs(zrc_dir)
     
    if not (zrc_dir / f'{task}/{splits}').exists():
      os.makedirs(zrc_dir / f'{task}/{splits}')

    progress = tqdm(total=len(self.data_loader['test_oos']), ncols=80, desc=f'Testing on zerospeech test set')
    phone_f = open(phone_file, 'w')
    for b_idx, batch in tqdm(enumerate(test_loader)):
      audios = batch[0]
      spans_ids = batch[1]
      input_mask = batch[4]
      x = torch.stack([audios[i, span_ids] for i, span_ids in enumerate(spans_ids)])
      input_mask = cuda(input_mask, self.cuda)
      x = cuda(x, self.cuda)

      word_logits, quantized_word_proba, phone_indices = self.audio_net.encode(x, masks=input_mask) 
      B = phone_indices.size(0)
      for idx in range(B):
        global_idx = b_idx * batch_size + idx
        audio_path = test_loader.dataset.dataset[global_idx][0]
        pred_phonemes = test_loader.dataset.span_to_segment(global_idx)
        audio_id = os.path.basename(audio_path).split('.')[0]
        if save_embedding:
          embed_id = f'{audio_id}_{global_idx}'
          if self.embed_type == 'proba': 
            embedding = F.softmax(word_logits[idx, :len(pred_phonemes)], dim=-1)
          elif self.embed_type == 'quantized_proba':
            embedding = quantized_word_proba[idx, :len(pred_phonemes)]
          elif self.embed_type == 'continuous':
            embedding = word_logits[idx, :len(pred_phonemes)]
          elif self.embed_type == 'continuous_cpc':
            embedding = torch.cat([x[idx], word_logits[idx]], dim=-1)

          embedding_frame = testset.unsegment(embedding, pred_phonemes)
          embed_path = zrc_dir / f'{task}/{splits}/{audio_id}.ark.gz'

          with WriteHelper(f'ark:| gzip -c > {embed_path}') as writer:
            writer('arr_0', embedding_frame.detach().cpu().numpy())

        pred_phone_label = phone_indices[idx]
        pred_phone_label = testset.unsegment(phone_indices[idx] + 1, pred_phonemes).long()

        us_ratio = int(self.audio_net.ds_ratio * (self.hop_len_ms // 10))
        if us_ratio > 1:
          pred_phone_label = pred_phone_label.unsqueeze(-1)\
                             .expand(-1, us_ratio).flatten()
           
        pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
        pred_phone_names = ','.join([str(phn_idx) for phn_idx in pred_phone_label_list])
        phone_f.write(f'{audio_id}\t{pred_phone_names}\n')
      progress.update(1)
    progress.close()
    phone_f.close()
    self.set_mode('train') 

  def viterbi_segment(self, word_logits, 
                      quantized, audio_mask, 
                      phoneme_num, segment_num, 
                      word_label=None):
    """Globally optimal segmentation by viterbi decoding
    
    Args :
        word_logits : (num. of segments, num. of words) FloatTensor,
        quantized : (num. of segments, num. of words) FloatTensor,
        segment_mask : (num. of segments,) FloatTensor,
        phoneme_num : int,
        segment_num : int,
        word_label (optional) : int,
 
    Returns :
        best_spans: list of [int, int] lists
    """
    word_label = word_label.item() if word_label else word_label
    phoneme_num = segment_num if phoneme_num > segment_num else phoneme_num  
    I_ZY = self.mutual_information(word_logits, quantized, word_label=word_label) 

    pointers = cuda(-1*torch.ones(phoneme_num+1, segment_num+1, dtype=torch.long), self.cuda)
    scores = cuda(torch.zeros(phoneme_num+1, segment_num+1).log(), self.cuda)
    scores[0, 0] = 0.0
    for i in range(1, phoneme_num+1):
      for end in range(i, segment_num+1):
        new_scores = cuda(torch.zeros(end).log(), self.cuda)
        for begin in range(1, end+1):
          k = self.get_span_id(begin-1, end-1)
          if audio_mask[k]:
            new_scores[begin-1] = scores[i-1][begin-1] + I_ZY[k]
        scores[i, end], pointers[i, end] = new_scores.max(-1)
      
    end = segment_num
    best_spans = []
    for i in range(phoneme_num, 0, -1):
      begin = pointers[i, end].data.cpu().item()
      best_spans.append([begin, end-1])
      end = begin
      if end < 1 and i > 1:
        print(f'Warning: back pointer stops at phoneme {i} > 1')
    return best_spans[::-1]
            
  def mutual_information(self, word_logits, quantized, word_label=None):
    KL = masked_kl_div(quantized, word_logits, mask=None, reduction=None) 
    #if word_label is not None:
    #  return word_logits[:, word_label] - KL
    return -KL

  def get_span_id(self, begin, end):
    return self.data_loader['train'].dataset.get_span_id(begin, end)

  def set_mode(self, mode='train'): 
    if mode == 'train':
      self.audio_net.train()
      if self.audio_feature_net is not None:
        self.audio_feature_net.train()
    elif mode == 'eval':
      self.audio_net.eval()
      if self.audio_feature_net is not None:
        self.audio_feature_net.eval()
    else:
      raise('mode error. It should be either train or eval')

  def load_checkpoint(self, filename='best_acc.tar'):
    if self.config.mode == 'test':
      filename = f'best_acc_{self.config.seed}.tar' 
    else:
      filename = f'best_acc_oos_{self.config.seed}.tar'
    file_path = self.ckpt_dir.joinpath(filename)
    if file_path.is_file():
      print('=> loading checkpoint "{}"'.format(file_path))
      checkpoint = torch.load(file_path.open('rb'))
      self.global_epoch = checkpoint['epoch']
      self.global_iter = checkpoint['iter']
      self.history = checkpoint['history']
      self.audio_net.load_state_dict(checkpoint['model_states']['audio_net'], strict=False)
      print('=> loaded checkpoint "{} (iter {}, epoch {})"'.format(
                file_path, self.global_iter, self.global_epoch))
    else:
      print('=> no checkpoint found at "{}"'.format(file_path))

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'audio_net': self.audio_net.state_dict()    
    }
    optim_states = {
      'optim': self.optim.state_dict()  
    }
    states = {
      'iter': self.global_iter,
      'epoch': self.global_epoch,
      'history': self.history,
      'config': self.config,
      'model_states': model_states,
      'optim_states': optim_states  
    }
    file_path = self.ckpt_dir.joinpath(filename)
    torch.save(states, file_path.open('wb+'))
    print('=> saved checkpoint "{}" (iter {}, epoch {})'.format(file_path, self.global_iter, self.global_epoch)) 
 

def main(argv):
  print(f'I am process {os.getpid()}')
  parser = argparse.ArgumentParser(description='Visual word information quantizer')
  parser.add_argument('CONFIG', type=str)
  parser.add_argument('-s', '--setting', default='basic')
  args = parser.parse_args(argv)

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

  config = ConfigFactory.parse_file(args.CONFIG)[args.setting]
  if not config.dset_dir:
    config.dset_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic' 
  
  if config.debug:
    ckpt_dir = 'checkpoints/debug'
  else:
    ckpt_dir = os.path.split(args.CONFIG)[-1].split('.')[0]
    ckpt_dir = f'checkpoints/{ckpt_dir}_{config.model_type}_{args.setting}'
  config['ckpt_dir'] = ckpt_dir
  config.ckpt_dir = ckpt_dir
  word_accs = []
  token_precs = []
  token_recs = []
  token_f1s = []
  for seed in config.get('seeds', [config.seed]):
    config.seed = seed
    config['seed'] = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    print()
    print('[CONFIGS]')
    print(config)
    print()

    net = Solver(config)
    save_embedding = config.get('save_embedding', False)
    if config.mode == 'train':
      net.train(save_embedding=save_embedding)
    elif config.mode == 'test':
      net.test(save_embedding=save_embedding) 
    elif config.mode == 'test_oos':
      net.test_out_of_sample(save_embedding=save_embedding)
    elif config.mode == 'test_zerospeech':
      net.test_zerospeech(save_embedding=save_embedding)
    else:
      return 0
    word_accs.append(net.history['word_acc'])
    token_precs.append(net.history['token_result'][0])
    token_recs.append(net.history['token_result'][1])
    token_f1s.append(net.history['token_result'][2])

  word_accs = np.asarray(word_accs)
  token_precs = np.asarray(token_precs)
  token_recs = np.asarray(token_recs)
  token_f1s = np.asarray(token_f1s)

  mean_word_acc, std_word_acc = np.mean(word_accs), np.std(word_accs)
  mean_token_prec, std_token_prec = np.mean(token_precs), np.std(token_precs)
  mean_token_rec, std_token_rec = np.mean(token_recs), np.std(token_recs)
  mean_token_f1, std_token_f1 = np.mean(token_f1s), np.std(token_f1s) 
  print(f'Average Word Acc.: {mean_word_acc:.4f}+/-{std_word_acc:.4f}\n'
        f'Average Token Precision: {mean_token_prec:.4f}+/-{std_token_prec:.4f}\t'
        f'Recall: {mean_token_rec:.4f}+/-{std_token_rec:.4f}\t'
        f'F1: {mean_token_f1:.4f}+/-{std_token_f1:.4f}') 

if __name__ == '__main__':
  argv = sys.argv[1:]
  main(argv)
