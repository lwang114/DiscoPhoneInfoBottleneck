import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from datasets.datasets import return_data
import cpc.criterion as cr
import cpc.eval as ev 
from model import BLSTM
from pathlib import Path
import json

class Solver(object):

  def __init__(self, args):
      self.args = args

      self.cuda = (args.cuda and torch.cuda.is_available())
      self.epoch = args.epoch
      self.batch_size = args.batch_size
      self.lr = args.lr
      self.eps = 1e-9
      self.K = args.K
      self.beta = args.beta
      self.n_predicts = args.n_predicts # Number of prediction steps
      self.global_iter = 0
      self.global_epoch = 0
      self.ds_ratio = args.ds_ratio
 
      if args.model_type == 'blstm':
        self.encoder = cuda(BLSTM(self.K), self.cuda)

      self.criterion = cr.CPCUnsupervisedCriterion(nPredicts=self.n_predicts,
                                              dimOutputAR=self.K,
                                              dimOutputEncoder=self.K,
                                              negativeSamplingExt=8)

      # History
      self.history = dict()
      self.history['abx_acc']=0.
      self.history['total_loss']=0.
      self.history['avg_loss']=0.
      self.history['epoch']=0
      self.history['iter']=0

  def set_mode(self,mode='train'):
      if mode == 'train':
        self.encoder.train()
      elif mode == 'eval':
        self.encoder.eval()
      else : raise('mode error. It should be either train or eval')
  
  def train(self):
      self.set_mode('train')
      
      total_loss = 0.
      total_step = 0
      pred_dicts = []
      for e in range(self.epoch) :
        self.global_epoch += 1
        
        for idx, (audios,labels,masks) in enumerate(self.data_loader['train']):
          self.global_iter += 1
          x = Variable(cuda(audios, self.cuda))
          masks = Variable(cuda(masks, self.cuda))
          spk_labels = torch.zeros((x.size(0),), dtype=torch.int, device=x.device)
          c_feature = self.encoder(x)
          loss, acc = self.criterion(c_feature, x, spk_labels) # TODO Add masks
          loss = loss.sum()
          acc = acc.mean(dim=0).cpu().numpy()
          total_loss += loss.cpu().detach().numpy()
          total_step += audios.size(0)

          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

          if self.global_iter % 100 == 0:
            print('i:{} avg loss (total loss):{.2f} ({.2f}) acc:{.4f} err:{.4f}'.format(self.global_iter, total_loss / total_step, total_loss, acc, 1 - acc))

          # TODO Save tensorboard
          # if self.global_iter % 10 == 0:
          #   if self.tensorboard:
  
  def test(self):
      self.set_mode('eval')

      total_loss = 0
      correct = 0
      total_num = 0
      seq_list = []
      with torch.no_grad():
        B = 0
        for b_idx, (audios,labels,masks) in enumerate(self.data_loader['test']):
          if b_idx == 0:
            B = audios.size(0)
          x = Variable(cuda(audios, self.cuda))
          y = Variable(cuda(labels, self.cuda))
          masks = Variable(cuda(masks, self.cuda))

          spk_labels = torch.zeros((audios.size(0),), dtype=torch.int, device=x.device)
          c_feature = self.encoder(x)
          loss, acc = self.criterion(c_feature, x, spk_labels) 
          total_loss += loss.sum().cpu().detach().numpy()
          correct += acc.sum(dim=0).cpu().numpy() 
          total_num += x.size(0)

          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            example_id = self.data_loader['test'].dataset.dataset[global_idx][0].split('/')[-1]
            text = self.data_loader['test'].dataset.dataset[global_idx][1]
            feat_id = f'{example_id}_{global_idx}'
            feat_fn = self.ckpt_dir.joinpath(f'{feat_id}.pt')
            torch.save(c_feature[idx], feat_fn)
            seq_list.append((feat_id, feat_fn))
      avg_loss = total_loss / total_num
      avg_acc = correct / total_num  

      # Compute ABX score
      path_item_file = os.path.join(self.data_loader['test'].dataset.data_path, 'abx_triplets.item')
      abx_score = ev.ABX(torch.load,
                       path_item_file,
                       seq_list,
                       'cosine',
                       step_feature,
                       ['within']) # TODO Include speaker identities 

      if self.history['acc'] < avg_acc:
        self.history['acc'] = avg_acc

      if self.history['abx_acc'] < (1-abx_score).item():
        self.history['abx_acc'] = (1-abx_score).item() 
        self.history['loss'] = loss.item()
        self.history['epoch'] = self.global_epoch
        self.history['iter'] = self.global_iter
        if save_ckpt : self.save_checkpoint('best_acc.tar')
        np.savez(self.ckpt_dir.joinpath('best_rnn_feats.npz'), **pred_features)
      print('[TEST RESULT]')
      print('e:{} loss:{:.2f} acc:{:.4f} best acc:{:.4f}'.format(self.global_epoch, avg_loss, avg_acc, 1 - avg_acc, self.history['acc']))
      print('abx error:{:.4f} abx acc:{:.4f} best abx acc:{:.4f}'.format(abx_score, 1-abx_score, self.history['abx_acc']))

      self.set_mode('train')

    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.encoder.state_dict(),
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

            self.encoder.load_state_dict(checkpoint['model_states']['net'])
            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
