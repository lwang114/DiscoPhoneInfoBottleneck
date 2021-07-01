import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import fairseq
import argparse
import os
import json
import math
from itertools import groupby
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.clustering import KMeans
from utils.utils import cuda
from model import GumbelBLSTM, GaussianBLSTM
from datasets.datasets import return_data
from utils.evaluate import compute_accuracy, compute_token_f1, compute_edit_distance

class Solver(object):

  
  def __init__(self, config):
    self.config = config

    self.cuda = torch.cuda.is_available()
    self.epoch = config.epoch
    self.batch_size = config.batch_size
    self.beta = config.beta
    self.lr = config.lr
    self.n_layers = config.get('num_layers', 1)
    self.weight_word_loss = config.get('weight_word_loss', 1.)
    self.weight_evidence = config.get('weight_evidence', 1.)
    self.anneal_rate = config.get('anneal_rate', 3e-6)
    self.num_sample = config.get('num_sample', 1)
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
    else:
      raise ValueError(f"Feature type {config.audio_feature} not supported")
  
    self.K = config.K
    self.global_iter = 0
    self.global_epoch = 0
    self.audio_feature = config.audio_feature
    self.image_feature = config.image_feature
    self.debug = config.debug
    self.dataset = config.dataset

    # Dataset
    self.data_loader = return_data(config)
    self.ignore_index = config.get('ignore_index', -100)
    self.n_visual_class = self.data_loader['train']\
                          .dataset.preprocessor.num_visual_words
    self.n_phone_class = 50
    self.visual_words = self.data_loader['train'].dataset.preprocessor.visual_words
    print(f'Number of visual label classes = {self.n_visual_class}')
    
    self.audio_net = cuda(GaussianBLSTM(
                            self.K,
                            input_size=self.input_size,
                            n_layers=self.n_layers,
                            n_class=self.n_visual_class+2*self.input_size,
                            ds_ratio=1,
                            bidirectional=True
                          ), self.cuda)
    trainables = [p for p in self.audio_net.parameters()]
    optim_type = config.get('optim', 'adam')
    if optim_type == 'sgd':
      self.optim = optim.SGD(trainables, lr=self.lr)
    else:
      self.optim = optim.Adam(trainables,
                              lr=self.lr, betas=(0.5, 0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)

    # History
    self.history = dict()
    self.history['acc'] = 0.
    self.history['token_f1'] = 0.
    self.history['loss'] = 0.
    self.history['epoch'] = 0
    self.history['iter'] = 0 
    self.history['temp'] = 1.

    self.ckpt_dir = Path(config.ckpt_dir)
    if not self.ckpt_dir.exists():
      self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    self.load_ckpt = config.load_ckpt
    if self.load_ckpt or config.mode == 'test':
      self.load_checkpoint()

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

  def train(self, save_embedding=False):
    self.set_mode('train')
    preprocessor = self.data_loader['train'].dataset.preprocessor
    temp_min = 0.1
    anneal_rate = self.anneal_rate
    temp = self.history['temp']
        
    total_loss = 0.
    total_step = 0.
    for e in range(self.epoch):
      self.global_epoch += 1
      pred_word_labels = []
      gold_word_labels = []
      for idx, (audios, _, word_labels,\
                audio_masks, _, word_masks)\
          in enumerate(self.data_loader['train']):
        if idx > 2 and self.debug:
          break
        self.global_iter += 1

        audios = cuda(audios, self.cuda)

        if self.audio_feature == 'wav2vec2':
          x = self.audio_feature_net.feature_extractor(audios)
        word_labels = cuda(word_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)
        word_masks = cuda(word_masks, self.cuda)
        audio_lens = audio_masks.sum(-1).long()
        word_lens = (word_labels > 0).long().sum(-1)
        (mu, std),\
        outputs,\
        embedding = self.audio_net(x,
                                   masks=audio_masks,
                                   temp=temp,
                                   num_sample=self.num_sample,
                                   return_feat=True
                                                ) 
        word_logits = outputs[:, :, :self.n_visual_class]
        word_logits = torch.matmul(word_masks, word_logits)
        mu_x = outputs[:, :, self.n_visual_class:self.n_visual_class+self.input_size]
        std_x = outputs[:, :, self.n_visual_class+self.input_size:self.n_visual_class+2*self.input_size]
        std_x = F.softplus(std_x-5, beta=1)

        word_loss = F.cross_entropy(word_logits.permute(0, 2, 1), word_labels,\
                                     ignore_index=-100,
                                     ).div(math.log(2))
        info_loss = -0.5 * (1 + 2*std.log() - mu.pow(2) - std.pow(2)).sum(-1).mean().div(math.log(2)) 
        evidence_loss = self.weight_evidence * F.mse_loss(x.permute(0, 2, 1), mu_x)

        loss = self.weight_word_loss * word_loss + evidence_loss + self.beta * info_loss
        izy_bound = math.log(self.n_visual_class, 2) - word_loss
        izx_bound = info_loss
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        for i in range(audios.size(0)):
          word_len = word_lens[i]
          if word_len > 0:
            gold_word_labels.append(word_labels[i, :word_len].cpu().detach().numpy().tolist())
            pred_word_label = word_logits[i, :word_len].max(-1)[1]
            pred_word_labels.append(pred_word_label.cpu().detach().numpy().tolist())

        if self.global_iter % 1000 == 0:
          temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
          avg_loss = total_loss / total_step
          print(f'i:{self.global_iter:d} temp:{temp} avg loss (total loss):{avg_loss:.2f} ({total_loss:.2f}) '
                  f'IZY:{izy_bound:.2f} IZX:{izx_bound:.2f} Evidence:{evidence_loss}')

      acc = compute_accuracy(gold_word_labels, pred_word_labels)
      print(f'Epoch {self.global_epoch}\ttraining visual word accuracy: {acc:.3f}')
      if (self.global_epoch % 2) == 0:
        self.scheduler.step()
      self.test(save_embedding=save_embedding)

  def test(self, save_embedding=False, out_prefix='predictions'):
    self.set_mode('eval')
    testset = self.data_loader['test'].dataset 
    preprocessor = testset.preprocessor
    temp = self.history['temp']
    
    total_loss = 0.
    total_neg_evidence = 0.
    total_num = 0.
    gold_word_labels = []
    pred_word_labels = []
    gold_word_names = []
    if not self.ckpt_dir.joinpath('outputs/phonetic/dev-clean').is_dir():
      os.makedirs(self.ckpt_dir.joinpath('outputs/phonetic/dev-clean'))

    gold_path = os.path.join(os.path.join(testset.data_path, f'{testset.splits[0]}'))
    out_word_file = os.path.join(
                      self.ckpt_dir,
                      f'{out_prefix}_word.{self.global_epoch}.readable'
                     )
    word_f = open(out_word_file, 'w')
    word_f.write('Image ID\tGold label\tPredicted label\n')

    with torch.no_grad():
      B = 0
      for b_idx, (audios, _, word_labels,\
                  audio_masks, _, word_masks)\
                  in enumerate(self.data_loader['test']):
        if b_idx > 2 and self.debug:
          break
        if b_idx == 0: 
          B = audios.size(0)

        audios = cuda(audios, self.cuda)
        if self.audio_feature == 'wav2vec2':
          x = self.audio_feature_net.feature_extractor(audios)
        
        word_labels = cuda(word_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)
        word_masks = cuda(word_masks, self.cuda)

        audio_lens = audio_masks.sum(-1).long()
        word_lens = (word_labels >= 0).long().sum(-1)
        (mu, std),\
        outputs,\
        embedding = self.audio_net(x,
                                   masks=audio_masks,
                                   temp=temp,
                                   num_sample=self.num_sample,
                                   return_feat=True)
         
        word_logits = outputs[:, :, :self.n_visual_class]
        word_logits = torch.matmul(word_masks, word_logits)
        mu_x = outputs[:, :, self.n_visual_class:self.n_visual_class+self.input_size]
        std_x = outputs[:, :, self.n_visual_class+self.input_size:self.n_visual_class+2*self.input_size]
        std_x = F.softplus(std_x-5, beta=1)

        word_loss = F.cross_entropy(word_logits.permute(0, 2, 1),
                                    word_labels,
                                    ignore_index=-100)\
                                    .div(math.log(2))

        info_loss = -0.5 * (1 + 2*std.log() - mu.pow(2) - std.pow(2)).sum(-1).mean().div(math.log(2))
        evidence_loss = self.weight_evidence * F.mse_loss(x.permute(0, 2, 1), mu_x).div(math.log(2))

        total_loss += (self.weight_word_loss * word_loss + evidence_loss + self.beta * info_loss).cpu().detach().numpy()
        total_neg_evidence += evidence_loss.cpu().detach().numpy()
        total_num += 1.
        
        for idx in range(audios.size(0)): 
          global_idx = b_idx * B + idx
          audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0]
          if word_lens[idx] > 0:
            gold_words = word_labels[idx, :word_lens[idx]]
            pred_words = word_logits[idx, :word_lens[idx]].max(-1)[1]

            gold_words = gold_words.cpu().detach().numpy().tolist()
            pred_words = pred_words.cpu().detach().numpy().tolist()
            gold_word_labels.append(gold_words)
            pred_word_labels.append(pred_words)       
            gold_word_names.append(preprocessor.to_word_text(gold_words))
            gold_word_str = ','.join(gold_word_names[-1])
            pred_word_str = ','.join(preprocessor.to_word_text(pred_words))
 
            word_f.write(f'{audio_id}\t{gold_word_str}\t{pred_word_str}\n')
          self.ckpt_dir.joinpath(f'outputs/phonetic/dev-clean/{audio_id}.txt')
          if save_embedding:
            np.savetxt(feat_fn, embedding[idx, :audio_lens[idx]][::2].cpu().detach().numpy()) # XXX
    word_f.close()
    avg_loss = total_loss / total_num 
    avg_neg_evidence = total_neg_evidence / total_num
    acc = compute_accuracy(gold_word_labels, pred_word_labels)
    
    print('[TEST RESULT]')
    print('Epoch {}\tLoss: {:.4f}\tEvidence Loss: {:.4f}\tWord Acc.: {:.3f}'\
          .format(self.global_epoch, avg_loss, avg_neg_evidence, acc))
    if self.history['acc'] < acc:
      self.history['acc'] = acc
      self.history['loss'] = avg_loss
      self.history['epoch'] = self.global_epoch
      self.history['iter'] = self.global_iter
    self.set_mode('train')

  def cluster(self, 
              n_clusters=50,
              out_prefix='quantized_outputs'):
    us_ratio = int(self.hop_len_ms / 10) * self.audio_net.ds_ratio 
    with torch.no_grad():
      B = 0
      utt_ids = []
      X = []      
      for b_idx, (audios, phoneme_labels, word_labels,\
                  audio_masks, phone_masks, word_masks)\
                  in enumerate(self.data_loader['test']):
        if b_idx > 2 and self.debug:
          break
        if b_idx == 0:
          B = audios.size(0)
        
        if self.audio_feature == 'wav2vec2':
          x = self.audio_feature_net.feature_extractor(audios)
          audio_masks = cuda(audio_masks, self.cuda)
          audio_lens = audio_masks.sum(-1).long()
          _, _, embedding = self.audio_net(x,
                                           masks=audio_masks,
                                           temp=temp,
                                           num_sample=self.num_sample,
                                           return_feat=True)
        
        for idx in range(audios.size(0)): 
          global_idx = b_idx * B + idx
          utt_id = os.path.splitext(os.path.basename(testset.dataset[global_idx][0]))[0] 
          X.extend(embedding[idx, :audio_lens[idx]].cpu().detach().numpy().tolist())
          utt_ids.extend([utt_id]*audio_lens[idx])
          print(utt_id) # XXX
      X = np.asarray(X)
      clusterer = KMeans(n_clusters=n_clusters).fit(X) 
      np.save(self.ckpt_dir.joinpath('cluster_means.npy'), clusterer.cluster_centers_)
      
      ys = clusterer.predict(X)
      filename = self.ckpt_dir.joinpath(out_prefix+'.txt')
      out_f = open(filename, 'w')
      for _, (utt_id, group) in itertools.groupby(list(zip(utt_ids, ys)), lambda x:x[0]):
        print(utt_id, list(group)) # XXX
        y = ' '.join([str(g[1]) for g in group for _ in range(us_ratio)])
        out_f.write(f'{utt_id} {y}\n') 
      out_f.close()
      gold_path = os.path.join(os.path.join(testset.data_path, f'{testset.splits[0]}'))
      token_f1, token_prec, token_recall = compute_token_f1(
                                             filename,
                                             gold_path,
                                             self.ckpt_dir.joinpath(f'confusion.png'),
                                           ) 

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

  def load_checkpoint(self, filename='best_acc.tar'):
    file_path = self.ckpt_dir.joinpath(filename)
    if file_path.is_file():
      print(f'=> loading checkpoint "{file_path}"')
      checkpoint = torch.load(file_path.open('rb'))
      self.global_epoch = checkpoint['epoch']
      self.global_iter = checkpoint['iter']
      self.history = checkpoint['history']

      self.audio_net.load_state_dict(checkpoint['model_states']['audio_net'])
      print('=> loaded checkpoint "{} (iter {}, epoch {})"'.format(
                file_path, self.global_iter, self.global_epoch))
    else:
      print('=> no checkpoint found at "{}"'.format(file_path))
      print('=> no checkpoint found at "{}"'.format(file_path))
