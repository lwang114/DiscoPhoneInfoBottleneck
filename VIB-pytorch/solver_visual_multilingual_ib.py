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
from utils.utils import cuda
from pathlib import Path
from sklearn.metrics import accuracy_score
from model import GumbelBLSTM, GumbelMLP, GumbelTDS, VQMLP, BLSTM
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
    self.weight_phone_loss = config.get('weight_phone_loss', 1.)
    self.weight_word_loss = config.get('weight_word_loss', 1.)
    self.anneal_rate = config.get('anneal_rate', 3e-6)
    self.num_sample = config.get('num_sample', 1)
    self.eps = 1e-9
    self.max_grad_norm = config.get('max_grad_norm', None)
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
    self.n_visual_class = self.data_loader['train']\
                          .dataset.preprocessor.num_visual_words
    self.n_phone_class = self.data_loader['train']\
                         .dataset.preprocessor.num_tokens
    self.visual_words = self.data_loader['train']\
                        .dataset.preprocessor.visual_words 
    print(f'Number of visual label classes = {self.n_visual_class}')
    print(f'Number of phone classes = {self.n_phone_class}')
  
    self.model_type = config.model_type 
    if config.model_type == 'gumbel_blstm':
      self.audio_net = cuda(GumbelBLSTM(
                              self.K,
                              input_size=self.input_size,
                              n_layers=self.n_layers,
                              n_class=self.n_visual_class,
                              n_gumbel_units=self.n_phone_class,
                              ds_ratio=1,
                              bidirectional=True), self.cuda)
      self.K = 2 * self.K
    elif config.model_type == 'blstm':
      self.audio_net = cuda(BLSTM(
        self.K,
        input_size=self.input_size,
        n_layers=self.n_layers,
        n_class=self.n_visual_class+self.n_phone_class,
        bidirectional=True), self.cuda)
      self.K = 2 * self.K
    elif config.model_type == 'mlp':
      self.audio_net = cuda(GumbelMLP(
                                self.K,
                                input_size=self.input_size,
                                n_class=self.n_visual_class,
                                n_gumbel_units=self.n_phone_class,
                            ), self.cuda)
    elif config.model_type == 'tds':
      self.audio_net = cuda(GumbelTDS(
                              input_size=self.input_size,
                              n_class=self.n_visual_class,
                              n_gumbel_units=self.n_phone_class,
                            ), self.cuda)
    elif config.model_type == 'vq-mlp':
      self.audio_net = cuda(VQMLP(
                              self.K,
                              input_size=self.input_size,
                              n_class=self.n_visual_class,
                              n_embeddings=self.n_phone_class
                            ), self.cuda) 
  
    trainables = [p for p in self.audio_net.parameters()]
    optim_type = config.get('optim', 'adam')
    if optim_type == 'sgd':
      self.optim = optim.SGD(trainables, lr=self.lr)
    else:
      self.optim = optim.Adam(trainables,
                              lr=self.lr, betas=(0.5,0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)
    self.ckpt_dir = Path(config.ckpt_dir)
    if not self.ckpt_dir.exists(): 
      self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    self.load_ckpt = config.load_ckpt
    if self.load_ckpt or config.mode == 'test': 
      self.load_checkpoint()
    
    # History
    self.history = dict()
    self.history['acc']=0. 
    self.history['token_f1']=0.
    self.history['loss']=0.
    self.history['epoch']=0
    self.history['iter']=0
 
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
    temp = 1.

    total_loss = 0.
    total_step = 0
    for e in range(self.epoch):
      self.global_epoch += 1
      pred_word_labels = []
      gold_word_labels = []
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
        word_lens = (word_labels >= 0).long().sum(-1)

        if self.model_type == "blstm":
          out_logits, embedding = self.audio_net(x, return_feat=True)
          word_logits = out_logits[:, :, :self.n_visual_class]
          phone_logits = out_logits[:, :, self.n_visual_class:]
        else:
          gumbel_logits, out_logits, _, embedding = self.audio_net(
            x, masks=audio_masks,
            temp=temp,
            num_sample=self.num_sample,
            return_feat=True)
          phone_logits = gumbel_logits
          word_logits = out_logits
        quantized = None
        if self.model_type == 'vq-mlp':
          word_logits = out_logits[:, :, :self.n_visual_class]
          quantized = out_logits[:, :, self.n_visual_class:]

        word_logits = torch.matmul(word_masks, word_logits)

        word_loss = F.cross_entropy(word_logits.permute(0, 2, 1), word_labels,\
                                    ignore_index=-100,
                                    ).div(math.log(2))
        phone_loss = F.ctc_loss(F.log_softmax(phone_logits, dim=-1)\
                                  .permute(1, 0, 2),
                                phoneme_labels,
                                audio_lens,
                                sent_lens) 
        info_loss = (F.softmax(phone_logits, dim=-1)\
                      * F.log_softmax(phone_logits, dim=-1)
                    ).sum().div(sent_lens.sum()*math.log(2)) 
        loss = self.weight_phone_loss * phone_loss +\
               self.weight_word_loss * word_loss +\
               self.beta * info_loss
        if self.model_type == 'vq-mlp':
          loss += self.audio_net.quantize_loss(embedding, quantized,
                                               masks=audio_masks)

        izy_bound = math.log(self.n_visual_class, 2) - word_loss
        izx_bound = info_loss
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.

        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
          torch.nn.utils.clip_grad_norm_(
            self.audio_net.parameters(),
            self.max_grad_norm
          )
        self.optim.step()
  
        for i in range(audios.size(0)):
          audio_len = audio_lens[i]
          sent_len = sent_lens[i]
          word_len = word_lens[i]

          gold_phone_label = phoneme_labels[i, :sent_len]
          pred_phone_label = phone_logits[i, :audio_len].max(-1)[1]
          gold_phone_labels.append(gold_phone_label.cpu().detach().numpy().tolist())
          pred_phone_labels.append(pred_phone_label.cpu().detach().numpy().tolist())
          if word_len > 0:
            gold_word_labels.append(word_labels[i, :word_len].cpu().detach().numpy().tolist())
            pred_word_label = word_logits[i, :word_len].max(-1)[1]
            pred_word_labels.append(pred_word_label.cpu().detach().numpy().tolist())
          
        if self.global_iter % 1000 == 0:
          temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
          avg_loss = total_loss / total_step
          print(f'i:{self.global_iter:d} temp:{temp} avg loss (total loss):{avg_loss:.2f} ({total_loss:.2f}) '
                f'IZY:{izy_bound:.2f} IZX:{izx_bound:.2f}')
      
      # Evaluate training visual word classification accuracy and phone token error rate
      acc = compute_accuracy(gold_word_labels, pred_word_labels)
      dist, n_tokens = compute_edit_distance(pred_phone_labels, gold_phone_labels, preprocessor)
      pter = float(dist) / float(n_tokens)
      print(f'Epoch {self.global_epoch}\ttraining visual word accuracy: {acc:.3f}\ttraining phone token error rate: {pter:.3f}')

      if (self.global_epoch % 2) == 0:
        self.scheduler.step()
      self.test(save_embedding=save_embedding)

  def test(self, save_embedding=False, out_prefix='predictions'):
    self.set_mode('eval')
    testset = self.data_loader['test'].dataset 
    preprocessor = testset.preprocessor
    
    total_loss = 0.
    total_num = 0.
    gold_labels = []
    pred_labels = []
    if not self.ckpt_dir.joinpath('outputs/phonetic/dev-clean').is_dir():
      os.makedirs(self.ckpt_dir.joinpath('outputs/phonetic/dev-clean'))

    gold_path = os.path.join(os.path.join(testset.data_path, f'{testset.splits[0]}'))
    out_word_file = os.path.join(
                      self.ckpt_dir,
                      f'{out_prefix}_word.{self.global_epoch}.readable'
                    )
    out_phone_readable_file = os.path.join(
                      self.ckpt_dir,
                      f'{out_prefix}_phoneme.{self.global_epoch}.readable'
                    )
    out_phone_file = os.path.join(
                       self.ckpt_dir,
                       f'{out_prefix}_phoneme.{self.global_epoch}.txt'
                     )

    word_f = open(out_word_file, 'w')
    word_f.write('Image ID\tGold label\tPredicted label\n')
    phone_readable_f = open(out_phone_readable_file, 'w')
    phone_f = open(out_phone_file, 'w')
    
    gold_word_labels = []
    gold_phone_labels = []
    pred_word_labels = []
    pred_phone_labels = []
     
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
          audios = self.audio_feature_net.feature_extractor(audios)
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
        word_lens = (word_labels >= 0).long().sum(-1)

        if self.model_type == 'blstm':
          out_logits, embedding = self.audio_net(audios, return_feat=True)
          word_logits = out_logits[:, :, :self.n_visual_class]
          phone_logits = out_logits[:, :, self.n_visual_class:]
        else:
          gumbel_logits, out_logits, encoding, embedding = self.audio_net(
            audios, masks=audio_masks,
            return_feat=True)
          phone_logits = gumbel_logits
          word_logits = out_logits

        if self.model_type == 'vq-mlp':
          word_logits = out_logits[:, :, :self.n_visual_class]

        word_logits = torch.matmul(word_masks, word_logits)
        word_loss = F.cross_entropy(word_logits.permute(0, 2, 1), 
                                    word_labels,
                                    ignore_index=-100)\
                                    .div(math.log(2)) 
        phone_loss = F.ctc_loss(F.log_softmax(phone_logits, dim=-1)\
                                  .permute(1, 0, 2),
                                phoneme_labels,
                                audio_lens,
                                sent_lens)
        info_loss = (F.softmax(phone_logits, dim=-1)\
                      * F.log_softmax(phone_logits, dim=-1)
                    ).sum().div(sent_lens.sum() * math.log(2))
        total_loss += word_loss + phone_loss + self.beta * info_loss
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

            gold_word_names = ','.join(preprocessor.to_word_text(
                                    gold_words)
                                  )
            pred_word_names = ','.join(preprocessor.to_word_text(
                                    pred_words)
                                  )
            word_f.write(f'{audio_id}\t{gold_word_names}\t{pred_word_names}\n')

          feat_fn = self.ckpt_dir.joinpath(f'outputs/phonetic/dev-clean/{audio_id}.txt')
          if save_embedding:
            np.savetxt(feat_fn, embedding[idx, :audio_lens[idx]][::2].cpu().detach().numpy()) # XXX
           
          gold_phone_label = phoneme_labels[idx, :sent_lens[idx]]
          pred_phone_label = phone_logits[idx, :audio_lens[idx]].max(-1)[1]
          gold_phone_names = ','.join(preprocessor.to_text(gold_phone_label))
          pred_phone_names = ','.join(preprocessor.tokens_to_text(pred_phone_label))
          phone_readable_f.write(f'Utterance id: {audio_id}\n'
                                 f'Gold transcript: {gold_phone_names}\n'
                                 f'Pred transcript: {pred_phone_names}\n\n')

          gold_phone_label = gold_phone_label.cpu().detach().numpy().tolist()
          if int(self.hop_len_ms / 10) * self.audio_net.ds_ratio > 1:
            us_ratio = int(self.hop_len_ms / 10) * self.audio_net.ds_ratio
            pred_phone_label = pred_phone_label.unsqueeze(-1)\
                                 .expand(-1, us_ratio).flatten()
          pred_phone_label = pred_phone_label.cpu().detach().numpy().tolist()
          gold_phone_labels.append(gold_phone_label)
          pred_phone_labels.append(pred_phone_label) 
          
          pred_phone_label = preprocessor.to_index(preprocessor.to_text(pred_phone_label))
          pred_phone_label = pred_phone_label.cpu().detach().numpy().tolist()
          pred_phone_names = ','.join([str(p) for p in pred_phone_label])
          phone_f.write(f'{audio_id} {pred_phone_names}\n')  
   
    word_f.close()
    phone_f.close()              
   
    avg_loss = total_loss / total_num 
    acc = compute_accuracy(gold_word_labels, pred_word_labels)
    dist, n_tokens = compute_edit_distance(pred_phone_labels, gold_phone_labels, preprocessor)
    pter = float(dist) / float(n_tokens)
    print('[TEST RESULT]')
    print('Epoch {}\tLoss: {:.4f}\tWord Acc.: {:.3f}\tPTER: {:.3f}'\
          .format(self.global_epoch, avg_loss, acc, pter))
    token_f1, token_prec, token_recall = compute_token_f1(
                                           out_phone_file,
                                           gold_path,
                                           os.path.join(
                                             self.ckpt_dir,
                                             f'confusion.{self.global_epoch}.png'
                                           )
                                         )
    if self.history['acc'] < acc:
      self.history['acc'] = acc
      self.history['loss'] = avg_loss
      self.history['epoch'] = self.global_epoch
      self.history['iter'] = self.global_iter
      self.history['token_f1'] = token_f1
      self.save_checkpoint('best_acc.tar')
    self.set_mode('train')

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'net': self.audio_net.state_dict()  
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
      print('=> loading checkpoint "{}"'.format(file_path))
      checkpoint = torch.load(file_path.open('rb'))
      self.global_epoch = checkpoint['epoch']
      self.global_iter = checkpoint['iter']
      self.history = checkpoint['history']

      self.audio_net.load_state_dict(checkpoint['model_states']['net'])
      print('=> loaded checkpoint "{} (iter {}, epoch {})"'.format(
                file_path, self.global_iter, self.global_epoch))
    else:
      print('=> no checkpoint found at "{}"'.format(file_path))
