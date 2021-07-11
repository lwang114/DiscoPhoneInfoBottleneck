import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import fairseq
import argparse
import os
import json
import time
import numpy as np
from pathlib import Path
from utils.utils import cuda
from model import BLSTM
from phone_model import UnigramPronunciator, LinearPositionAligner
from criterion import MicroTokenFLoss, MacroTokenFLoss
from datasets.datasets import return_data
from utils.evaluate import compute_accuracy, compute_token_f1, compute_edit_distance

class Solver(object):

  def __init__(self, config):
    self.config = config

    self.cuda = torch.cuda.is_available()
    self.beta = 1. # XXX
    self.epoch = config.epoch
    self.batch_size = config.batch_size
    self.lr = config.lr
    self.n_layers = config.get('num_layers', 3)
    self.eps = 1e-9
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
    else:
      raise ValueError(f"Feature type {config.audio_feature} not supported")
      
    self.K = config.K
    self.global_iter = 0
    self.global_epoch = 0
    self.audio_feature = config.audio_feature
    self.image_feature = config.image_feature
    self.debug = config.debug
    self.dataset = config.dataset
    self.max_normalize = config.get('max_normalize', False)
    self.loss_type = config.get('loss_type', 'macro_token_floss')
    self.beta_f_measure = config.get('beta_f_measure', 0.3)
    if self.loss_type == 'macro_token_floss':
      self.criterion = MacroTokenFLoss(beta=self.beta_f_measure)
    elif self.loss_type == 'micro_token_floss':
      self.criterion = MicroTokenFLoss(beta=self.beta_f_measure)
    else:
      raise ValueError(f'Invalid loss type {self.loss_type}')

    # Dataset
    self.data_loader = return_data(config)
    self.ignore_index = config.get('ignore_index', -100)
    self.n_visual_class = self.data_loader['train']\
                          .dataset.preprocessor.num_visual_words
    self.n_phone_class = self.data_loader['train'].dataset.preprocessor.num_tokens
    self.visual_words = self.data_loader['train'].dataset.preprocessor.visual_words
    self.phone_set = self.data_loader['train'].dataset.preprocessor.tokens
    self.max_feat_len = self.data_loader['train'].dataset.max_feat_len
    self.max_word_len = self.data_loader['train'].dataset.max_word_len

    print(f'Number of visual label classes = {self.n_visual_class}')
    print(f'Number of phone classes = {self.n_phone_class}')
    print(f'Max normalized: {self.max_normalize}')

    self.audio_net = cuda(BLSTM(self.K,
                                n_layers=self.n_layers,
                                n_class=self.n_phone_class,
                                input_size=self.input_size,
                                ds_ratio=1,
                                bidirectional=True), self.cuda)

    self.phone_net = cuda(UnigramPronunciator(self.visual_words,
                                              self.phone_set,
                                              ignore_index=self.ignore_index), self.cuda)
    self.align_net = cuda(LinearPositionAligner(scale=0.), self.cuda) # XXX 

    trainables = [p for p in self.audio_net.parameters()]
    optim_type = config.get('optim', 'adam')
    if optim_type == 'sgd':
      self.optim = optim.SGD(trainables, lr=self.lr)
    else:
      self.optim = optim.Adam(trainables,
                              lr=self.lr, betas=(0.5, 0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)
    self.ckpt_dir = Path(config.ckpt_dir)
    if not self.ckpt_dir.exists(): 
      self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    self.load_ckpt = config.load_ckpt
    if self.load_ckpt or config.mode in ['test', 'cluster']: 
      self.load_checkpoint()
    
    # History
    self.history = dict()
    self.history['token_f1']=0.
    self.history['visual_token_f1']=0. 
    self.history['loss']=0.
    self.history['epoch']=0
    self.history['iter']=0

  def train_phone_net(self):
    file_path = self.ckpt_dir.joinpath('pron_dict.json')
    begin_time = time.time()
    for split in ['train', 'test']: 
      for _, phoneme_labels, word_labels,\
          _, phone_masks, word_masks in self.data_loader[split]:
        self.phone_net.update(word_labels, phoneme_labels)
    print(f'Finish training pronounciation model after {time.time()-begin_time}s')
    self.phone_net.save_readable(file_path)

  def train(self, save_embedding=False):
    self.set_mode('train')
    preprocessor = self.data_loader['train'].dataset.preprocessor
    total_loss = 0.
    total_step = 0
    
    if not self.load_ckpt:
      self.train_phone_net()
    
    for e in range(self.epoch):
      self.global_epoch += 1
      pred_phone_labels = []
      gold_phone_labels = []
      for idx, (audios, phoneme_labels, word_labels,\
                audio_masks, phone_masks, word_masks)\
                in enumerate(self.data_loader['train']):
        if idx > 2 and self.debug:
          break
        self.global_iter += 1
    
        x = cuda(audios, self.cuda)
        if self.audio_feature == "wav2vec2":
          x = self.audio_feature_net.feature_extractor(x)
        phoneme_labels = cuda(phoneme_labels, self.cuda)
        word_labels = cuda(word_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)
        phone_masks = cuda(phone_masks, self.cuda)
        word_masks = cuda(word_masks, self.cuda)
   
        if self.audio_net.ds_ratio > 1:
          audio_masks = audio_masks[:, ::self.audio_net.ds_ratio]
          word_masks = word_masks[:, :, ::self.audio_net.ds_ratio]

        audio_lens = audio_masks.sum(-1).long()
        sent_lens = phone_masks.sum(-1).long()
        word_lens = word_masks.sum(dim=(-1, -2)).long()
        
        cluster_logits, embedding = self.audio_net(x, return_feat=True)
        ## Phone loss
        cluster_probs = F.softmax(cluster_logits, dim=-1)\
                        .view(-1, self.max_feat_len, self.n_phone_class) 
        if self.max_normalize:
          cluster_probs = cluster_probs / cluster_probs.max(-1, keepdim=True)[0]

        phoneme_labels_aligned = self.align_net(F.one_hot(phoneme_labels * phone_masks.long(), 
                                                          self.n_phone_class),
                                                phone_masks,
                                                audio_masks)

        loss = self.criterion(cluster_probs,
                              phoneme_labels_aligned,
                              audio_masks) # XXX
        loss = loss + F.ctc_loss(F.log_softmax(cluster_logits, dim =-1).permute(1, 0, 2),
                                 phoneme_labels,
                                 audio_lens,
                                 sent_lens)

        ## Word loss 
        word_cluster_logits = torch.matmul(word_masks, cluster_logits.unsqueeze(1))
        word_cluster_probs = F.softmax(word_cluster_logits, dim=-1)\
                             .view(-1, self.max_word_len, self.n_phone_class)
        if self.max_normalize:
          word_cluster_probs = word_cluster_probs / word_cluster_probs.max(-1, keepdim=True)[0]
        
        # (batch size, max num. words, num. phone classes)
        word_phone_probs = self.phone_net(word_labels.flatten())
        # (batch size x max num. words, max word len., num. phone classes)
        word_phone_probs = word_phone_probs\
                           .unsqueeze(1).expand(-1, self.max_word_len, -1)

        loss = loss + 0. * self.criterion(word_cluster_probs,
                                     word_phone_probs,
                                     word_masks.sum(-1).view(-1, self.max_word_len)) # XXX

        total_loss += loss.cpu().detach().numpy() 
        total_step += 1.

        if loss == 0:
          continue
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.global_iter % 1000 == 0:
          avg_loss = total_loss / total_step
          print(f'i:{self.global_iter:d} avg loss (total loss):{avg_loss:.2f} ({total_loss:.2f})')
      
      avg_loss = total_loss / total_step
      print(f'Epoch {self.global_epoch}\tTraining Loss: {avg_loss:.3f}')

      if (self.global_epoch % 2) == 0:
        self.scheduler.step()
      self.test(save_embedding=save_embedding)

  def test(self, save_embedding=False, out_prefix='predictions'):
    self.set_mode('eval')
    testset = self.data_loader['test'].dataset
    preprocessor = testset.preprocessor

    total_loss = 0.
    total_step = 0.

    gold_word_labels = []
    if not self.ckpt_dir.joinpath('outputs/phonetic/dev-clean').is_dir():
      os.makedirs(self.ckpt_dir.joinpath('outputs/phonetic/dev-clean'))

    gold_phone_file = os.path.join(testset.data_path, f'{testset.splits[0]}/{testset.splits[0]}_nonoverlap.item')
    gold_visual_phone_file = os.path.join(testset.data_path, f'{testset.splits[0]}/{testset.splits[0]}_visual.item')

    phone_file = self.ckpt_dir.joinpath(f'{out_prefix}_phoneme.{self.global_epoch}.txt')
    visual_phone_file = self.ckpt_dir.joinpath(f'{out_prefix}_visual_phoneme.{self.global_epoch}.txt')
    phone_f = open(phone_file, 'w')
    visual_phone_f = open(visual_phone_file, 'w')
    phone_readable_f = open(self.ckpt_dir.joinpath(f'{out_prefix}_phoneme.{self.global_epoch}.readable'), 'w')
    visual_phone_readable_f = open(self.ckpt_dir.joinpath(f'{out_prefix}_visual_phoneme.{self.global_epoch}.readable'), 'w')
    
    with torch.no_grad():
      B = 0
      for b_idx, (audios, phoneme_labels, word_labels,\
                  audio_masks, phone_masks, word_masks)\
                  in enumerate(self.data_loader['test']):
        if b_idx > 2 and self.debug:
          break
        if b_idx == 0: 
          B = audios.size(0)
        
        audios = cuda(audios, self.cuda)
        if self.audio_feature == 'wav2vec2':
          x = self.audio_feature_net.feature_extractor(audios)
        else:
          x = audios
        
        word_labels = cuda(word_labels, self.cuda)
        phoneme_labels = cuda(phoneme_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)
        phone_masks = cuda(phone_masks, self.cuda)
        word_masks = cuda(word_masks, self.cuda)

        audio_lens = audio_masks.sum(-1).long()
        sent_lens = phone_masks.sum(-1).long()
        word_lens = word_masks.sum(dim=(-2, -1)).long()
        word_num = (word_labels >= 0).long().sum(-1)
        cluster_logits, embedding = self.audio_net(x, return_feat=True)
        phoneme_labels_aligned = self.align_net(F.one_hot(phoneme_labels * phone_masks.long(), 
                                                          self.n_phone_class), 
                                                phone_masks, 
                                                audio_masks)

        cluster_probs = F.softmax(cluster_logits, dim=-1)\
                        .view(-1, self.max_feat_len, self.n_phone_class)
        if self.max_normalize:
          cluster_probs = cluster_probs / cluster_probs.max(-1, keepdim=True)[0] 

        phone_label_mask = F.one_hot(phoneme_labels * phone_masks.long(), self.n_phone_class).sum(1, keepdim=True)
        phone_label_mask = (phone_label_mask > 0).long()
        phoneme_labels_aligned_ctc = (cluster_probs * phone_label_mask).max(-1)[1]
        phoneme_labels_aligned_ctc = F.one_hot(phoneme_labels_aligned_ctc, self.n_phone_class).detach()
        loss = self.criterion(cluster_probs,
                              phoneme_labels_aligned_ctc,
                              audio_masks)

        word_cluster_logits = torch.matmul(word_masks, cluster_logits.unsqueeze(1))
        word_cluster_probs = F.softmax(word_cluster_logits, dim=-1)\
                             .view(-1, self.max_word_len, self.n_phone_class)
        # XXX word_cluster_probs = word_cluster_probs / word_cluster_probs.max(-1, keepdim=True)[0] 

        word_phone_probs = self.phone_net(word_labels.flatten())
        word_phone_probs = word_phone_probs\
                           .unsqueeze(1).expand(-1, self.max_word_len, -1)
        
        loss = loss + self.criterion(word_cluster_probs,
                                     word_phone_probs,
                                     word_masks.sum(-1).view(-1, self.max_word_len))
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.
        
        for idx in range(audios.size(0)):
          global_idx = b_idx * B + idx
          audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0]
          gold_phone_label = phoneme_labels[idx, :sent_lens[idx]]
          pred_phone_label = cluster_logits[idx, :audio_lens[idx]].max(-1)[1]
          gold_phone_names = ','.join(preprocessor.to_text(gold_phone_label))
          pred_phone_names = ','.join(preprocessor.tokens_to_text(pred_phone_label))
          phone_readable_f.write(f'Utterance id: {audio_id}\n'
                                 f'Gold transcript: {gold_phone_names}\n'
                                 f'Pred transcript: {pred_phone_names}\n\n')
          us_ratio = int(self.hop_len_ms / 10) * self.audio_net.ds_ratio > 1
          if us_ratio > 1:
            pred_phone_label = pred_phone_label.unsqueeze(-1)\
                               .expand(-1, us_ratio).flatten()
          pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
          pred_phone_names = ','.join([str(p) for p in pred_phone_label_list])
          phone_f.write(f'{audio_id} {pred_phone_names}\n')

          if word_num[idx] > 0:
            pred_phone_word_masked_label = pred_phone_label\
                                           * word_masks[idx, :, :, :audio_lens[idx]].long().sum(dim=(0, 1))
            pred_phone_word_masked_label_list = pred_phone_word_masked_label.detach().cpu().numpy().tolist()
            pred_phone_word_masked_names = ','.join([str(p) for p in pred_phone_word_masked_label_list])
            visual_phone_f.write(f'{audio_id} {pred_phone_word_masked_names}\n')

            gold_word_label = word_labels[idx, :word_num[idx]].cpu().detach().numpy().tolist()
            gold_word_names = preprocessor.to_word_text(gold_word_label)
            pred_visual_phone_label = word_cluster_logits[idx, :word_num[idx]].max(-1)[1] 
            for word_idx in range(word_num[idx]):
              gold_word_name = gold_word_names[word_idx]
              pred_label = pred_visual_phone_label[word_idx, :word_lens[idx, word_idx]]
              pred_label = pred_label.cpu().detach().numpy().tolist()
              pred_visual_phone_names = ','.join(preprocessor.tokens_to_text(pred_label))
              visual_phone_readable_f.write(f'Utterance id: {audio_id}\n'
                                            f'Gold word label: {gold_word_name}\n'
                                            f'Pred transcript: {pred_visual_phone_names}\n\n') 
      phone_f.close()
      visual_phone_f.close()
      phone_readable_f.close()
      visual_phone_readable_f.close()

      avg_loss = total_loss / total_step
      
      # Token F1
      token_f1,\
      token_prec,\
      token_recall = compute_token_f1(phone_file,
                                      gold_phone_file,
                                      self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))

      visual_token_f1,\
      visual_token_prec,\
      visual_token_recall = compute_token_f1(visual_phone_file,
                                             gold_visual_phone_file,
                                             self.ckpt_dir.joinpath(f'visual_confusion.{self.global_epoch}.png'))
      print('[TEST RESULT]')
      print(f'Epoch {self.global_epoch}\tLoss: {avg_loss:.4f}\tToken F1: {token_f1:.3f}\tVisual Token F1: {visual_token_f1:.3f}') 

      if self.history['visual_token_f1'] < visual_token_f1:
        self.history['token_f1'] = token_f1
        self.history['visual_token_f1'] = visual_token_f1
        self.history['loss'] = avg_loss
        self.history['iter'] = self.global_iter
        self.history['epoch'] = self.global_epoch
        self.save_checkpoint('best_acc.tar')
      self.set_mode('train') 

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
    file_path = self.ckpt_dir.joinpath(filename)
    if file_path.is_file():
      print('=> loading checkpoint "{}"'.format(file_path))
      checkpoint = torch.load(file_path.open('rb'))
      self.global_epoch = checkpoint['epoch']
      self.global_iter = checkpoint['iter']
      self.history = checkpoint['history']

      self.audio_net.load_state_dict(checkpoint['model_states']['audio_net'])
      self.audio_net.load_state_dict(checkpoint['model_states']['phone_net'])

      print('=> loaded checkpoint "{} (iter {}, epoch {})"'.format(
                file_path, self.global_iter, self.global_epoch))
    else:
      print('=> no checkpoint found at "{}"'.format(file_path))

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'audio_net': self.audio_net.state_dict(),
      'phone_net': self.phone_net.state_dict()
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
     
