import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import fairseq
import argparse
import sys
import os
import json
import time
import numpy as np
import argparse
from copy import deepcopy
from pyhocon import ConfigFactory
from pathlib import Path
import shutil
from sklearn.metrics import precision_recall_fscore_support
from utils.utils import cuda, str2bool
from model import GumbelBLSTM, GumbelMLP, InfoQuantizer
from datasets.datasets import return_data
from utils.evaluate import compute_accuracy, compute_token_f1, compute_edit_distance

EPS = 1e-10
NULL = '###NULL###'

class Solver(object):
  
  def __init__(self, config):
    self.config = config
    self.cuda = torch.cuda.is_available()
    self.mode = config.mode

    self.debug = debug
    self.get_dataset_config(config)
    self.get_feature_config(config)
    self.get_model_config(config)
    self.get_optim_config(config)
    
    self.ckpt_dir = Path(config.ckpt_dir)
    self.history = dict()
    self.history['token_result'] = [0., 0., 0.]
    self.history['predict_acc'] = [0.] * self.bag_size
    self.history['loss'] = 0.
    self.history['epoch'] = 0
    self.history['iter'] = 0

  def get_dataset_config(self):
    config = self.config
    self.dataset_name = config['dataset']
    self.oos_dataset_name = None
    self.data_loaders = return_data(config)
    self.ignore_index = config.get('ignore_index', -100)

    self.n_phone_class = self.data_loader['train'].dataset.preprocessor.num_tokens
    self.phone_set = self.data_loader['train'].dataset.preprocessor.tokens
    self.max_feat_len = self.data_loader['train'].dataset.max_feat_len
    self.max_segment_num = self.data_loader['train'].dataset.max_segment_num
    print(f'Number of phone classes = {self.n_phone_class}')
    print(f'Number of clusters = {self.n_clusters}')

  def get_feature_config(self):
    config = self.config
    if config.audio_feature == 'mfcc':
      self.audio_feature_net = None
      self.input_size = 80
      self.hop_len_ms = 10
    elif config.audio_feature == 'cpc':
      self.audio_feature_net = None
      self.input_size = 256
      self.hop_len_ms = 10
    else:
      raise ValueError(f"Feature type {config.audio_feature} not supported")
  
  def get_model_config(self):
    config = self.config
    self.n_clusters = config.get("n_clusters", self.n_phone_class)
    self.bag_width = config.bag_width
    self.bag_size = self.bag_width * 2 + 1

    z_dim = self.bag_size * self.n_phone_class
    alpha = [100] * self.n_clusters
    init_embedding = np.random.dirichlet(alpha, size=(self.n_clusters, bag_size)).reshape(self.n_clusters, -1) 
    if self.debug:
      print('init_embedding.sum(-1), expected', init_embedding.sum(-1), self.bag_size)
    self.audio_net = cuda(InfoQuantizer(in_channels=self.input_size,
                                        channels=config.K,
                                        n_embeddings=self.n_clusters,
                                        z_dim=z_dim,
                                        init_embedding=init_embedding))
    if self.mode in ['test'] or self.load_ckpt:
      self.load_checkpoint(f'best_acc_{config.seed}.tar') 
  
  def get_optim_config(self):
    config = self.config
    self.batch_size = config.batch_size
    self.lr = config.lr

    trainables = [p for p in self.audio_net.parameters()]
    self.optim = torch.optim.Adam(trainables,
                                  lr=self.lr,
                                  betas=(0.5, 0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)
    self.epoch = config.epoch
    self.global_iter = 0
    self.global_epoch = 0

  def create_bag_of_phone_labels(phn_labels):
    bag_of_phn_labels = []
    for i in range(-self.bag_width, self.bag_width+1):
      if i > 0:
        phn_labels_padded = torch.pad(phn_labels, (0, i), 'constant', 0)
        if self.debug:
          print('phn_labels_padded[0], i: ', phn_labels_padded[0], i) 
        bag_of_phn_labels.append(phn_labels_padded[i:])
      elif i < 0:
        phn_labels_padded = torch.pad(phn_labels, (abs(i), 0), 'constant', 0)
        if self.debug:
          print('phn_labels_padded[0], i: ', phn_labels_padded[0], i) 
        bag_of_phn_labels.append(phn_labels_padded[:i])
      else:
        bag_of_phn_labels.append(phn_labels)
    bag_of_phn_labels = torch.stack(bag_of_phn_labels, dim=-2)
    return bag_of_phn_labels

  def train(self, save_embedding=False):
    self.set_mode('train')
    
    total_loss = 0.
    total_quantizer_loss = 0.
    total_steps = 0.
    for e in range(self.epoch):
      if e > 1 and self.debug:
        break
      self.global_epoch += 1

      for b_idx, batch in enumerate(self.data_loader['train']):
        if b_idx > 2 and self.debug:
          break
        self.global_iter += 1

        audios = batch[0]
        phn_labels = batch[1]
        audio_masks = batch[3]
        
        x = cuda(audios, self.cuda)
        phn_labels = cuda(phn_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)

        # Create bag-of-phone labels
        bag_of_phn_labels = self.create_bag_of_phone_labels(phn_labels)

        # Quantization loss
        phone_logits, quantized, quantizer_loss = self.audio_net(x, masks=audio_masks)

        # Prediction loss
        pred_loss = F.cross_entropy(phone_logits.view(-1, self.n_phone_class), 
                                    bag_of_phn_labels.view(-1, self.n_phone_class), 
                                    ignore_index=self.ignore_index)
                                    
        loss = pred_loss + quantizer_loss
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.

        if loss == 0:
          continue
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if (self.global_iter - 1) % 1000 == 0:
          avg_loss = total_loss / total_step
          avg_quantizer_loss = total_quantizer_loss / total_step
          print(f'Itr {self.global_iter:d}\tAvg Loss (Total Loss):{avg_loss:.2f} ({total_loss:.2f})\tAvg Quantizer Loss:{avg_quantizer_loss:.2f}')

      avg_loss = total_loss / total_step
      avg_quantizer_loss = total_quantizer_loss / total_step
      print(f'Epoch {self.global_epoch}\tTraining Loss: {avg_loss:.3f}\tTraining Quantizer Loss: {avg_quantizer_loss:.3f}')

      if self.global_epoch % 2 == 0:
        self.scheduler.step()
      self.test(save_embedding=save_embedding)

  def test(self, save_embedding=False, out_prefix='predictions'):
    self.set_mode('eval')
    test_loader = self.data_loader['test']
    batch_size = test_loader.batch_size
    testset = test_loader.dataset
    
    total_loss = 0.
    total_steps = 0.
    
    pred_labels = []
    pred_labels_quantized = []
    gold_labels = []
    
    embeds = dict()
    embed_labels = dict()

    out_file = self.ckpt_dir.joinpath(f'{out_prefix}_{self.global_epoch}.readable')
    out_f = open(out_file, 'w')
    readable_f = open(f'{out_prefix}_{self.global_epoch}.readable', 'w')
    with torch.no_grad():
      for b_idx, batch in enumerate(test_loader):
        if b_idx > 2 and self.debug:
          break
        audios = batch[0]
        phn_labels = batch[1]
        audio_mask = batch[3]
   
        x = cuda(audios, self.cuda)
        phn_labels = cuda(phn_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)

        # Create bag-of-phone labels
        bag_of_phn_labels = self.create_bag_of_phone_labels(phn_labels)

        # Quantizer loss
        phone_logits, quantized, quantizer_loss = self.audio_net(x, masks=audio_masks)
        
        # Prediction loss
        pred_loss = F.cross_entropy(phone_logits.view(-1, self.n_phone_class), 
                                    bag_of_phn_labels.view(-1, self.n_phone_class), 
                                    ignore_index=self.ignore_index)

        loss = pred_loss + quantizer_loss
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.
         
        _, _, cluster_inds = self.audio_net.encode(x, masks=audio_masks)
        for idx in range(audios.size(0)):
          global_idx = b_idx * batch_size + idx
          audio_path = testset.dataset[global_idx][0]
          audio_id = os.path.split(audio_path)[1].split('.')[0]
          segments = testset.dataset[global_idx][-1]
          pred_phn_labels = testset.unsegment(cluster_inds[idx], segments).long()
          pred_phn_names = ','.join([str(p) for p in pred_phn_labels.cpu().detach().numpy().tolist()])
          out_f.write(f'{audio_id} {pred_phn_names}\n')

          pred_label = phone_logits[idx, :len(segments)].view(-1, self.bag_size, self.n_phone_class).max(-1)[1]
          pred_label_quantized = quantized[idx, :len(segments)].max(-1)[1].cpu().detach().numpy().tolist()
          gold_label = bag_of_phn_labels[idx: :len(segments)].cpu().detach().numpy().tolist()

          pred_labels.append(pred_label)
          pred_labels_quantized.append(pred_label_quantized)
          gold_labels.append(gold_label)
          
          pred_name = ','.join([str(pred_label[i]) for i in range(self.bag_size)])
          pred_name_quantized = ','.join([str(pred_label_quantized[i]) for i in range(self.bag_size)])
          gold_name = ','.join([str(gold_label[i]) for i in range(self.bag_size)])
          readable_f.write(f'Utterance id: {audio_id}\n'
                           f'Gold phone labels: {gold_name}\n'
                           f'Pred phone labels: {pred_name}\n'
                           f'Pred phone labels by quantizer: {pred_name_quantized}') 

          if save_embedding:
            embed_id = f'{audio_id}_{global_idx}'
            embeds[embed_id] = F.softmax(phone_logits[idx, :len(segments)], dim=-1).detach().cpu().numpy()
            embed_labels[embed_id] = {'phoneme_text': [s['text'] for s in segments],
                                      'word_text': [NULL]*len(segments)}
    out_f.close()
    readable_f.close()
    if save_embedding:
      np.savez(embed_file, **embeds)
      json.dump(embed_labels, open(embed_label_file, 'w'), indent=2)

    avg_loss = total_loss / total_step
    # Compute prediction accuracy and token F1
    print('[TEST RESULT]')
    pred_labels = np.asarray(pred_labels)
    pred_labels_quantized = np.asarray(pred_labels_quantized)
    gold_labels = np.asarray(gold_labels)
    pred_accs = np.zeros(self.bag_size)
    pred_accs_quantized = np.zeros(self.bag_size)
    for i in range(self.bag_size):
      pred_accs[i] = compute_accuracy(gold_labels[:, i], pred_labels[:, i])
      pred_accs_quantized[i] = compute_accuracy(gold_labels[:, i], pred_labels_quantized[:, i])

    gold_file = os.path.join(testset.data_path, f'{testset.splits[0]}/{testset.splits[0]}_nonoverlap.item')
    token_f1,\
    token_prec,\
    token_recall = compute_token_f1(out_file,
                                    gold_file,
                                    self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))
    info = f'Epoch {self.global_epoch}\tLoss: {avg_loss:.4f}\n'\
           f'Prediction Accuracies: {pred_accs}\n'\
           f'(By Quantizer) Prediction Accuracies: {pred_accs_quantized}\n'\
           f'Token Precision: {token_prec:.3f}\tToken Recall: {token_recall:.3f}\tToken F1: {token_f1:.3f}\n'
    print(info)

    save_path = self.ckpt_dir.joinpath(f'results_file_{self.config.seed}.txt')
    with open(save_path, 'a') as file:
      file.write(info)

    if self.history['token_result'][-1] < token_f1:
      self.history['token_result'] = [token_prec, token_recall, token_f1]
      self.history['loss'] = avg_loss
      self.history['iter'] = self.global_iter
      self.history['epoch'] = self.global_epoch
      self.save_checkpoint(f'best_acc_{self.config.seed}.tar')
      shutil.copyfile(out_file, self.ckpt_dir.joinpath(f'quantized_outputs_{self.config.seed}.txt'))
    self.set_mode('train')


  def set_mode(self, mode='train'): 
    if mode == 'train':
      self.audio_net.train()
    elif mode == 'eval':
      self.audio_net.eval()
    else:
      raise('mode error. It should be either train or eval')

  def load_checkpoint(self, filename='best_acc.tar'):
    filename = f'best_acc_{self.config.seed}.tar' 
    file_path = self.ckpt_dir.joinpath(filename)
    if file_path.is_file():
      print('=> loading checkpoint "{}"'.format(file_path))
      checkpoint = torch.load(file_path.open('rb'))
      self.global_epoch = checkpoint['epoch']
      self.global_iter = checkpoint['iter']
      self.history = checkpoint['history']
      self.audio_net.load_state_dict(checkpoint['model_states']['audio_net'])
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

