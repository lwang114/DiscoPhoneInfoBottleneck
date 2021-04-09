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
from datasets.datasets import return_data
from utils import cuda
import cpc.criterion as cr
import cpc.eval as ev
from model import GumbelBLSTM, GumbelPyramidalBLSTM, GumbelMarkovBLSTM
from pathlib import Path
from evaluate import evaluate

class Solver(object):

  def __init__(self, args):
      self.args = args

      self.cuda = (args.cuda and torch.cuda.is_available())
      self.epoch = args.epoch
      self.batch_size = args.batch_size
      self.beta = args.beta
      self.lr = args.lr
      self.eps = 1e-9
      self.K = args.K
      self.n_predicts = args.n_predicts # Number of prediction steps
      self.n_negatives = args.n_negatives # Number of negative samples per step
      self.global_iter = 0
      self.global_epoch = 0
      self.cpc_feature = args.cpc_feature
      self.loss_type = args.loss_type
      self.dataset = args.dataset

      # Dataset
      self.data_loader = return_data(args)
      self.n_class = self.data_loader['train'].dataset.preprocessor.num_tokens
      
      if args.model_type == 'gumbel_blstm':
        self.ds_ratio = 1
        self.net = cuda(GumbelBLSTM(self.K, n_class=self.n_class, ds_ratio=self.ds_ratio), self.cuda)
        self.K = 2*self.K
      elif args.model_type == 'pyramidal_blstm':
        self.ds_ratio = 4
        self.net = cuda(GumbelPyramidalBLSTM(self.K, n_class=self.n_class, ds_ratio=self.ds_ratio), self.cuda)
        self.net.weight_init()
      elif args.model_type == 'gumbel_markov_blstm':
        self.ds_ratio = 1
        self.net = cuda(GumbelMarkovBLSTM(self.K), self.cuda)
        self.net.weight_init()

      self.cpc_criterion = cr.CPCUnsupervisedCriterion(nPredicts=self.n_predicts,
                                                       dimOutputAR=self.K if self.cpc_feature=='rnn' else 49,
                                                       dimOutputEncoder=80,
                                                       negativeSamplingExt=self.n_negatives,
                                                       rnnMode=False)
      self.cpc_criterion = cuda(self.cpc_criterion, self.cuda)

      trainables = [p for p in self.net.parameters()]
      trainables += [p for p in self.cpc_criterion.parameters()]        
      
      self.optim = optim.Adam(trainables,
                              lr=self.lr,betas=(0.5,0.999))
      self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

      self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
      if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
      self.load_ckpt = args.load_ckpt
      if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)
      
      # History
      self.history = dict()
      self.history['acc']=0. 
      self.history['token_f1']=0.
      self.history['abx']=0.5
      self.history['total_loss']=0.
      self.history['avg_loss']=0.
      self.history['epoch']=0
      self.history['iter']=0
      
  def set_mode(self,mode='train'):
      if mode == 'train':
        self.net.train()
      elif mode == 'eval':
        self.net.eval()
      else : raise('mode error. It should be either train or eval')
  
  def train(self):
      self.set_mode('train')
      temp_min = 0.1
      anneal_rate = 3e-6
      temp = 1.

      total_loss = 0.
      total_step = 0
      pred_dicts = []
      for e in range(self.epoch) :
        self.global_epoch += 1
        
        for idx, (audios,labels,masks) in enumerate(self.data_loader['train']):
          self.global_iter += 1
          x = Variable(cuda(audios, self.cuda))
          y = Variable(cuda(labels, self.cuda))
          masks = Variable(cuda(masks, self.cuda))

          # Compute IB loss
          in_logit, logits, c_feature = self.net(x, masks=masks, temp=temp, return_feat=self.cpc_feature)
          logit = logits.sum(-2)
          class_loss = F.cross_entropy(logit,y).div(math.log(2))
          info_loss = (F.softmax(in_logit,dim=-1) * F.log_softmax(in_logit,dim=-1)).sum(1).mean().div(math.log(2))
          ib_loss = class_loss + self.beta * info_loss
          
          izy_bound = math.log(self.n_class,2) - class_loss
          izx_bound = info_loss
          prediction = F.softmax(logit,dim=1).max(1)[1]
          word_acc = torch.eq(prediction,y).float().mean()
          
          # Compute CPC loss
          spk_labels = torch.zeros((x.size(0),), dtype=torch.int, device=x.device)
          cpc_loss, cpc_acc = self.cpc_criterion(c_feature, x.permute(0, 2, 1), spk_labels)
          cpc_loss = cpc_loss.sum()
          
          cpc_acc = cpc_acc.mean(0).cpu().numpy()

          if self.loss_type == 'IB-only':
            loss = ib_loss
          elif self.loss_type == 'CPC-only':
            loss = cpc_loss
          else:
            loss = ib_loss + cpc_loss
          total_loss += loss.cpu().detach().numpy()
          total_step += audios.size(0)

          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

          if self.global_iter % 100 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
            print(f'i:{self.global_iter:d} avg loss (total loss):{total_loss / total_step:.2f} ({total_loss:.2f}) IZY:{izy_bound:.2f} IZX:{izx_bound:.2f} '
                  f'word acc:{word_acc:.4f} word err:{1-word_acc:.4f} '
                  f'cpc acc at step 1:{cpc_acc[0]:.4f} cpc acc at step last:{cpc_acc[-1]:.4f}')

        if (self.global_epoch % 2) == 0 : self.scheduler.step()
        compute_abx = False
        if self.global_epoch % 5 == 0:
          compute_abx = True
        self.test(compute_abx=compute_abx)

          
  def test(self, save_ckpt=True, compute_abx=False):
      self.set_mode('eval')

      total_loss = 0
      izy_bound = 0
      izx_bound = 0
      cpc_correct = 0
      word_correct = 0
      total_num = 0
      pred_dicts = []
      seq_list = []
      if not self.ckpt_dir.joinpath('feats').is_dir():
        self.ckpt_dir.joinpath('feats').mkdir()
      
      with torch.no_grad():
        B = 0
        for b_idx, (audios,labels,masks) in enumerate(self.data_loader['test']):
          if b_idx == 0:
            B = audios.size(0)
          x = Variable(cuda(audios, self.cuda))
          y = Variable(cuda(labels, self.cuda))
          masks = Variable(cuda(masks, self.cuda))

          in_logit, logits, encoding = self.net(x, masks=masks, return_feat='bottleneck')
          _, _, c_feature = self.net(x, masks=masks, return_feat=self.cpc_feature)
          logit = logits.sum(dim=-2)

          # Word prediction
          cur_class_loss = F.cross_entropy(logit,y,size_average=False).div(math.log(2))
          cur_info_loss = (F.softmax(in_logit,dim=-1) * F.log_softmax(in_logit,dim=-1)).sum(1).mean().div(math.log(2))
          cur_ib_loss = cur_class_loss + self.beta * cur_info_loss
          izy_bound = izy_bound + y.size(0) * math.log(self.n_class,2) - cur_class_loss
          izx_bound = izx_bound + cur_info_loss
          word_prediction = F.softmax(logit,dim=1).max(1)[1]
          word_correct += torch.eq(word_prediction,y).float().sum()
          
          # CPC prediction
          spk_labels = torch.zeros((audios.size(0),), dtype=torch.int, device=x.device)
          cur_cpc_loss, cpc_acc = self.cpc_criterion(c_feature, x.permute(0, 2, 1), spk_labels) 
          cpc_correct += cpc_acc.sum(dim=0).mean().item() * x.size(0)

          total_loss += cur_cpc_loss.sum().item() + cur_ib_loss.item()
          total_num += x.size(0)

          _, _, c_feature = self.net(x, masks=masks, return_feat='rnn')
          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            example_id = self.data_loader['test'].dataset.dataset[global_idx][0].split('/')[-1]
            text = self.data_loader['test'].dataset.dataset[global_idx][1]
            feat_id = f'{example_id}_{global_idx}'      
            units = encoding[idx].max(-1)[1]
            pred_dicts.append({'sent_id': example_id,
                               'units': units.cpu().detach().numpy().tolist(),  
                               'text': text})

            if global_idx <= 100:
              feat_fn = self.ckpt_dir.joinpath(f'feats/{feat_id}.npy')
              np.save(feat_fn, c_feature[idx].cpu().detach().numpy())
              seq_list.append((feat_id, feat_fn))
      izy_bound /= total_num
      izx_bound /= total_num
      
      avg_loss = total_loss / total_num
      word_acc = word_correct / total_num
      avg_cpc_acc = cpc_correct / total_num
      token_f1, conf_df, token_prec, token_recall = evaluate(pred_dicts, self.data_loader['test'].dataset.gold_dicts, ds_rate=self.ds_ratio)
      if self.history['acc'] < word_acc.item():
        self.history['acc'] = word_acc.item()
        self.history['loss'] = avg_loss
        self.history['epoch'] = self.global_epoch
        self.history['iter'] = self.global_iter
        self.history['token_f1'] = token_f1
        if save_ckpt : self.save_checkpoint('best_acc.tar')
        conf_df.to_csv(self.ckpt_dir.joinpath('confusion_matrix.csv'))
      print('[TEST RESULT]')
      print('e:{} IZY:{:.2f} IZX:{:.4f}'
                .format(self.global_epoch, izy_bound.item(), izx_bound.item()), end=' ')
      print('token precision:{:.4f} token recall:{:.4f} token f1:{:.4f} word acc:{:.4f} word err:{:.4f} best word acc:{:.4f}'
                .format(token_prec, token_recall, token_f1, word_acc.item(), 1-word_acc.item(), self.history['acc']), end=' ')   
      print('e:{} loss:{:.2f} cpc acc:{:.4f} cpc err:{:.4f}'.format(self.global_epoch, avg_loss, avg_cpc_acc, 1 - avg_cpc_acc))
      if compute_abx:
        # Compute ABX score
        path_item_file = os.path.join(self.data_loader['test'].dataset.data_path, 'abx_triplets.item')
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

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'net':self.net.state_dict(),
      'cpc_criterion':self.cpc_criterion.state_dict()
    }
    optim_states = {
                'optim':self.optim.state_dict(),
                }
    states = {
      'iter':self.global_iter,
      'epoch':self.global_epoch,
      'history':self.history,
      'args':self.args,
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

      self.net.load_state_dict(checkpoint['model_states']['net'])
      self.cpc_criterion.load_state_dict(checkpoint['model_states']['cpc_criterion'])
      print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
    else:
      print("=> no checkpoint found at '{}'".format(file_path))

def load_feature(path):
  feat = np.load(path)
  return torch.FloatTensor(feat).unsqueeze(0)
