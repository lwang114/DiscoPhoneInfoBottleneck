import numpy as np
import torch
import argparse
import os
import json
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support 
from datasets.datasets import return_data
from utils import cuda
from model import GumbelBLSTM, GumbelMLP
from position_model import PositionPredictor
from pathlib import Path
from image_model import Resnet34 
from evaluate import evaluate, compute_token_f1
import cpc.eval as ev


class Solver(object):


  def __init__(self, config):
      self.config = config

      self.cuda = torch.cuda.is_available()
      self.epoch = config.epoch
      self.batch_size = config.batch_size
      self.beta = config.beta
      self.lr = config.lr
      self.anneal_rate = config.get('anneal_rate', 3e-6)
      self.num_sample = config.get('num_sample', 1)
      self.use_segment = config.get('use_segment', False)
      self.ds_method = config.get('downsample_method', 'average')
      self.eps = 1e-9
      if config.audio_feature == 'mfcc':
        self.input_size = 80
      elif config.audio_feature == 'cpc':
        self.input_size = 256
      else: Exception(f'Audio feature type {config.audio_feature_type} not supported')
      
      if self.use_segment and (self.ds_method == 'resample'): # input size is the concatenation of 10 frames for resample 
        self.input_size = 10 * self.input_size

      self.K = config.K
      self.global_iter = 0
      self.global_epoch = 0
      self.audio_feature = config.audio_feature
      self.image_feature = config.image_feature
      self.debug = config.debug
      self.dataset = config.dataset

      # Dataset
      self.data_loader = return_data(config)
      self.n_class = self.data_loader['train']\
                     .dataset.preprocessor.num_tokens
      self.class_names = self.data_loader['train']\
                         .dataset.preprocessor.tokens
      print(f'visual label class: {self.n_class}')
  
      self.image_net = Resnet34(pretrained=True, n_class=self.n_class)
      '''
      self.image_net.load_state_dict(
                       torch.load(
                         os.path.join(config.image_model_dir,
                                      'best_image_model.pth'
                         )
                       )
                     ) 
      '''
      self.image_net = cuda(self.image_net, self.cuda)

      if config.model_type == 'blstm': 
        self.audio_net = cuda(GumbelBLSTM(
                                self.K, 
                                input_size=self.input_size,
                                n_layers=1,
                                n_class=self.n_class, 
                                ds_ratio=1,
                                bidirectional=True), self.cuda)
        self.K = 2 * self.K 
      elif config.model_type == 'mlp':
        self.audio_net = cuda(GumbelMLP(
                                self.K,
                                input_size=self.input_size,
                                n_class=self.n_class
                              ), self.cuda)
      else: Exception(f'Model type {config.model_type} not defined')

      self.position_net = cuda(PositionPredictor(
                                 input_size=self.K,
                                 vocab_size=self.n_class,
                                 embedding_size=50
                               ), self.cuda) # TODO

      trainables = [p for p in self.audio_net.parameters()]             
      self.optim = optim.Adam(trainables,
                              lr=self.lr,betas=(0.5,0.999))
      self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)

      self.ckpt_dir = Path(config.ckpt_dir)
      if not self.ckpt_dir.exists(): 
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
      self.load_ckpt = config.load_ckpt
      if self.load_ckpt: 
        self.load_checkpoint()
      
      # History
      self.history = dict()
      self.history['f1']=0. 
      self.history['token_f1']=0.
      self.history['abx']=0.5
      self.history['total_loss']=0.
      self.history['avg_loss']=0.
      self.history['epoch']=0
      self.history['iter']=0
      
  def set_mode(self, mode='train'):
      if mode == 'train':
        self.audio_net.train()
        self.image_net.train()
      elif mode == 'eval':
        self.audio_net.eval()
        self.image_net.eval()
      else: 
        raise('mode error. It should be either train or eval')
  
  def train(self):
      self.set_mode('train')
      temp_min = 0.1
      anneal_rate = self.anneal_rate # 3e-6
      temp = 1.

      total_loss = 0.
      total_step = 0
      for e in range(self.epoch):
        self.global_epoch += 1
        pred_labels = []
        gold_labels = []
        for idx, (audios, images, labels, audio_masks, image_masks)\
            in enumerate(self.data_loader['train']):
          if idx > 2 and self.debug:
            break
          self.global_iter += 1
          x = cuda(audios, self.cuda)
          audio_masks = cuda(audio_masks, self.cuda)
          labels = cuda(labels, self.cuda)
          images = cuda(images, self.cuda)
          image_masks = cuda(image_masks.unsqueeze(-1), self.cuda)
          image_size = images.size()

          '''
          if images.ndim in [2, 4]:
            image_logit, _ = self.image_net(images, return_score=True) 
          else:
            image_logit_flat, _ = self.image_net(
                                      images.view(-1, *image_size[2:]),
                                      return_score=True
                                      )
            image_logit = image_logit_flat.view(*image_size[:2], -1)

          if self.image_feature == 'label':                 
            if image_logit.ndim == 2:
              y = image_logit.max(-1)[1]
            else:
              y = F.one_hot(image_logit.max(-1)[1], self.n_class)
              y = (y * image_masks).max(1)[0]
          elif self.image_feature == 'multi_label':
            if image_logit.ndim == 2:
              y = (image_logit > 0).float() 
            else:
              y = ((image_logit > 0).float() * image_masks).max(1)[0]
          '''
          y = labels  

          # Compute IB loss
          in_logit, logits, _, embedding = self.audio_net(
                                 x, masks=audio_masks, 
                                 temp=temp,
                                 num_sample=self.num_sample, 
                                 return_feat=True)
          logit = (logits * audio_masks.unsqueeze(-1)).sum(dim=1)
          pred_label = F.one_hot(logit.max(-1)[1], self.n_class)
          gold_label = F.one_hot(y, self.n_class)
          pred_labels.append(pred_label.cpu())
          gold_labels.append(gold_label.cpu())

          class_loss = F.cross_entropy(logit, y).div(math.log(2))
          pred_position = self.position_model(x).squeeze(-1)
          true_position = torch.range(embedding.size(1)).unsqueeze(0).expand(embedding.size(0), -1) 
          position_loss = F.mse_loss(pred_position * audio_masks, 
                                     true_position * audio_masks)
          info_loss = (F.softmax(in_logit, dim=-1)\
                        * F.log_softmax(in_logit, dim=-1)
                      ).sum(1).mean().div(math.log(2))
          loss = class_loss + self.beta * info_loss + position_loss

          izy_bound = math.log(self.n_class, 2) - class_loss
          izx_bound = info_loss
          total_loss += loss.cpu().detach().numpy()
          total_step += audios.size(0)

          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

          if self.global_iter % 1000 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
            avg_loss = total_loss / total_step
            print(f'i:{self.global_iter:d} temp:{temp} avg loss (total loss):{avg_loss:.2f} ({total_loss:.2f}) '
                  f'IZY:{izy_bound:.2f} IZX:{izx_bound:.2f}')
        pred_labels = torch.cat(pred_labels).detach().numpy()
        gold_labels = torch.cat(gold_labels).detach().numpy()
        _, _, f1s, _ = precision_recall_fscore_support(
                           gold_labels.flatten(),
                           pred_labels.flatten()
                       )
        print(f'Epoch {self.global_epoch}\ttraining F1: {f1s[1]}')

        if (self.global_epoch % 2) == 0: 
          self.scheduler.step()
        compute_abx = False
        # TODO if self.global_epoch % 5 == 0:
        #   compute_abx = True
        self.test(compute_abx=compute_abx)
          
  def test(self, compute_abx=False, out_prefix='predictions'):
      self.set_mode('eval')
      testset = self.data_loader['test'].dataset

      total_loss = 0
      izy_bound = 0
      izx_bound = 0
      total_num = 0
      seq_list = []
      pred_labels = []
      gold_labels = []
      if not self.ckpt_dir.joinpath('feats').is_dir():
        self.ckpt_dir.joinpath('feats').mkdir()

      gold_path = os.path.join(os.path.join(testset.data_path, 'test/'))
      out_word_file = os.path.join(
                   self.ckpt_dir,
                   f'{out_prefix}_word.{self.global_epoch}.readable'
                 )
      out_phone_file = os.path.join(
                         self.ckpt_dir,
                         f'{out_prefix}_phoneme.{self.global_epoch}.txt'
                       )

      word_f = open(out_word_file, 'w')
      word_f.write('Image ID\tGold label\tPredicted label\n')      
      phone_f = open(out_phone_file, 'w')
      with torch.no_grad():
        B = 0
        for b_idx, (audios, images, labels, audio_masks, image_masks) in enumerate(self.data_loader['test']):
          if b_idx > 2 and self.debug:
            break
          if b_idx == 0:
            B = audios.size(0)
          audios = cuda(audios, self.cuda)
          labels = cuda(labels, self.cuda)
          images = cuda(images, self.cuda)
          audio_masks = cuda(audio_masks, self.cuda)
          image_masks = cuda(image_masks.unsqueeze(-1), self.cuda)
          image_size = images.size()
          '''
          if images.ndim in [2, 4]:
            image_logit, _ = self.image_net(images, return_score=self.image_feature)
          else:
            image_logit_flat, _ = self.image_net(
                                      images.view(-1, *image_size[2:]),
                                      return_score=True
                                      )
            image_logit = image_logit_flat.view(*image_size[:2], -1)
          
          if self.image_feature == 'label':
            if image_logit.ndim == 2:
              y = image_logit.max(-1)[1]
            else:
              y = F.one_hot(image_logit.max(-1)[1], self.n_class)
              y = (y * image_masks).max(1)[0]
          elif self.image_feature == 'multi_label':
            if image_logit.ndim == 2:
              y = (image_logit > 0).float()          
            else:
              y = ((image_logit > 0).float() * image_masks).max(1)[0]
          '''
          y = labels
          in_logit, logits, encoding, embedding = self.audio_net(
                                                    audios, masks=audio_masks, 
                                                    return_feat=True
                                                  )

          # Word prediction
          logit = (logits * audio_masks.unsqueeze(-1)).sum(dim=1)
          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0]
            golds = [y[idx].cpu().detach().numpy()] 
            preds = [logit[idx].max(-1)[1].cpu().detach().numpy()]
            pred_phones = encoding[idx].max(-1)[1]
            if self.use_segment:
              pred_phones = testset.unsegment(pred_phones, testset.dataset[global_idx][3]).long()
            pred_phones = pred_phones.cpu().detach().numpy().tolist()
            gold_names = ','.join([self.class_names[c] for c in golds])
            pred_names = ','.join([self.class_names[c] for c in preds])
            pred_phones_str = ','.join([str(phn) for phn in pred_phones])
            word_f.write(f'{audio_id}\t{gold_names}\t{pred_names}\n')
            phone_f.write(f'{audio_id} {pred_phones_str}\n')

          pred_labels.append(F.one_hot(y, self.n_class).cpu())
          gold_labels.append(F.one_hot(logit.max(-1)[1], self.n_class).cpu())
          cur_class_loss = F.cross_entropy(logit, y).div(math.log(2)) 
          cur_info_loss = (F.softmax(in_logit, dim=-1)\
                            * F.log_softmax(in_logit, dim=-1)
                          ).sum(1).mean().div(math.log(2))
          cur_ib_loss = cur_class_loss + self.beta * cur_info_loss
          izy_bound = izy_bound + y.size(0) * math.log(self.n_class, 2) - cur_class_loss
          izx_bound = izx_bound + cur_info_loss
                    
          total_loss += cur_ib_loss.item()
          total_num += audios.size(0)
          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            example_id = testset.dataset[global_idx][0].split('/')[-1]
            text = testset.dataset[global_idx][1]
            feat_id = f'{example_id}_{global_idx}'      
            units = encoding[idx].max(-1)[1]

            if global_idx <= 100:
              feat_fn = self.ckpt_dir.joinpath(f'feats/{feat_id}.npy')
              np.save(feat_fn, embedding[idx].cpu().detach().numpy())
              seq_list.append((feat_id, feat_fn))
      word_f.close()
      phone_f.close()

      pred_labels = torch.cat(pred_labels).detach().numpy()
      gold_labels = torch.cat(gold_labels).detach().numpy()
      ps, rs, f1s, _ =  precision_recall_fscore_support(
                          gold_labels.flatten(),
                          pred_labels.flatten())
      p, r, f1 = ps[1], rs[1], f1s[1]

      class_f1s = np.zeros(self.n_class)
      for c in range(self.n_class):
        _, _, class_f1, _ = precision_recall_fscore_support(
                              gold_labels[:, c], 
                              pred_labels[:, c]
                            )
        class_f1s[c] = class_f1[-1]
      izy_bound /= total_num
      izx_bound /= total_num
      
      avg_loss = total_loss / total_num
      print('[TEST RESULT]')
      print('Epoch {}\tLoss: {:.2f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF1: {:.2f}'\
            .format(self.global_epoch, avg_loss, p, r, f1))
      print('Top 10 F1: {:.2f}\tBest F1: {:.2f}'\
            .format(class_f1s[:10].mean(), self.history['f1']))
      token_f1, token_prec, token_recall = compute_token_f1(
                                             out_phone_file, 
                                             gold_path,
                                             os.path.join(
                                               self.ckpt_dir, 
                                               f'confusion.{self.global_epoch}.png'
                                             ),
                                           )
      if self.history['f1'] < f1:
        self.history['f1'] = f1
        self.history['loss'] = avg_loss
        self.history['epoch'] = self.global_epoch
        self.history['iter'] = self.global_iter
        self.history['token_f1'] = token_f1
        self.save_checkpoint('best_acc.tar')
      
      if compute_abx:
        # Compute ABX score
        path_item_file = os.path.join(testset.data_path, 'abx_triplets.item')
        abx_score = ev.ABX(load_feature,
                           path_item_file,
                           seq_list,
                           distance_mode='cosine',
                           step_feature=160,
                           modes=['within'])
        abx_score = abx_score['within']
        if self.history['abx'] > abx_score:
          self.history['abx'] = abx_score 
        # print('abx error:{:.4f} abx acc:{:.4f} best abx acc:{:.4f}'.format(abx_score.item(), 1-abx_score.item(), self.history['abx_acc']))   
      self.set_mode('train')

  def cluster(self, test_loader=None, 
              out_prefix='predictions', 
              save_embedding=False):
    self.load_checkpoint()
    X = []
    audio_files = []

    if test_loader is not None:
      B = test_loader.batch_size
      testset = test_loader.dataset
      split = testset.splits[0]
      gold_path = os.path.join(testset.data_path, split)
    else:
      test_loader = self.data_loader['test']
      B = test_loader.batch_size
      testset = test_loader.dataset
      split = testset.splits[0]
      gold_path = os.path.join(testset.data_path, split)
    
    embed_path = os.path.join(testset.data_path, f'{split}_embeddings')
    if save_embedding and not os.path.exists(embed_path):
      os.makedirs(embed_path)

    for b_idx, (audios, _, _, audio_masks, _) in enumerate(test_loader):
      if b_idx > 2 and self.debug:
        break
      audios = cuda(audios, self.cuda)
      audio_masks = cuda(audio_masks, self.cuda)
      _, _, _, embedding = self.audio_net(
                               audios,
                               masks=audio_masks,
                               return_feat=True
                               )
      # Concatenate the hidden vector with the input feature
      concat_embedding = torch.cat([audios.permute(0, 2, 1), embedding], axis=-1)
      if save_embedding:
        for idx in range(audios.size(0)):
          audio_id = os.path.splitext(os.path.split(testset.dataset[b_idx*B+idx][0])[1])[0]
          np.savetxt(os.path.join(embed_path, f'{audio_id}.txt'),
                     concat_embedding[idx].cpu().detach().numpy())

      X.append(concat_embedding.cpu().detach().numpy())
      audio_files.extend([testset.dataset[b_idx*B+i][0] for i in range(audios.size(0))]) 
    X = np.concatenate(X, axis=0)

    shape = X.shape
    kmeans = KMeans(n_clusters=50).fit(X.reshape(shape[0]*shape[1], -1))
    np.save(os.path.join(self.ckpt_dir, f'kmeans_centroids.npy'), kmeans.cluster_centers_)
    encodings = kmeans.labels_.reshape(shape[0], shape[1])

    out_file = os.path.join(self.ckpt_dir, f'{out_prefix}_clustering.txt')
    out_f = open(out_file, 'w')
    for idx, (audio_file, encoding) in enumerate(zip(audio_files, encodings)):
      audio_id = os.path.splitext(os.path.split(audio_file)[1])[0]
      pred_phonemes = ','.join([str(phn) for phn in encodings[idx]])
      out_f.write(f'{audio_id} {pred_phonemes}\n')
    out_f.close()
    
    compute_token_f1(
      out_file,
      gold_path,
      os.path.join(
        self.ckpt_dir,
        'confusion_cluster.png'
      )
    )

  def phone_level_cluster(self, out_prefix='predictions'):
    self.load_checkpoint()
    X_a = np.zeros((self.n_class, self.K))
    norm = np.zeros((self.n_class, 1))
    audio_files = []
    encodings = []
    # Find the centroid of each phone-level cluster
    B = self.data_loader['test'].batch_size
    testset = self.data_loader['test'].dataset
    for b_idx, (audios, _, _, audio_masks, _) in enumerate(self.data_loader['test']): 
      if b_idx > 2 and self.debug:
        break
      audios = cuda(audios, self.cuda)
      audio_masks = cuda(audio_masks, self.cuda)
      _, _, encoding, embedding = self.audio_net(
                                      audios, 
                                      mask=audio_masks,
                                      return_feat=True
                                  )
      encoding = encoding.permute(0, 2, 1).cpu().detach().numpy()
      embedding = embedding.cpu().detach().numpy()
      X_a += encoding @ embedding
      norm += encoding.sum(axis=-1, keepdims=True)
      audio_files.extend([testset.dataset[b_idx*B+i][0] for i in range(audios.size(0))]) 
      encodings.append(encoding.T)
    encodings = np.concatenate(encodings)

    X_a /= norm
    X_s = self.audio_net.bottleneck.weight +\
                 self.audio_net.bottleneck.bias
    X = np.concatenate([X_a, X_s], axis=1)

    kmeans = KMeans(n_clusters=50).fit(X)
    phoneme_labels = kmeans.labels_
   
    out_file = os.path.join(self.ckpt_dir, f'{out_prefix}_phone_level_clustering.txt')
    out_f = open(out_file, 'w')
    pred_phones = encodings.max(-1)[0]
    for idx, (audio_file, encoding) in enumerate(zip(audio_files, encodings)):
      audio_id = os.path.splitext(os.path.split(audio_file)[1])[0]
      pred_phonemes = ','.join([str(phoneme_labels[phn]) for phn in pred_phones[idx]])
      out_f.write('{audio_id} {pred_phonemes}\n')
    out_f.close()
    
    gold_path = os.path.join(os.path.join(testset.data_path, 'test/'))
    compute_token_f1(
      out_file,
      gold_path,
      os.path.join(
        self.ckpt_dir,
        'confusion_phone_level_cluster.png'
      )
    ) 

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'net':self.audio_net.state_dict()
    }
    optim_states = {
                'optim':self.optim.state_dict(),
                }
    states = {
      'iter':self.global_iter,
      'epoch':self.global_epoch,
      'history':self.history,
      'config':self.config,
      'model_states':model_states,
      'optim_states':optim_states,
    }

    file_path = self.ckpt_dir.joinpath(filename)
    torch.save(states,file_path.open('wb+'))
    print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))

  def load_checkpoint(self, filename='best_acc.tar'):
    file_path = self.ckpt_dir.joinpath(filename)
    if file_path.is_file():
      print("=> loading checkpoint '{}'".format(file_path))
      checkpoint = torch.load(file_path.open('rb'))
      self.global_epoch = checkpoint['epoch']
      self.global_iter = checkpoint['iter']
      self.history = checkpoint['history']

      self.audio_net.load_state_dict(checkpoint['model_states']['net'])
      print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
    else:
      print("=> no checkpoint found at '{}'".format(file_path))

def load_feature(path):
  feat = np.load(path)
  return torch.FloatTensor(feat).unsqueeze(0)
