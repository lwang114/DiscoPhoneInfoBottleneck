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
from utils import cuda
import cpc.criterion as cr
import cpc.eval as ev
from model import BLSTM, GumbelBLSTM, VQCPCEncoder
from pathlib import Path
import os
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
      self.n_predicts = args.n_predicts # Number of prediction steps
      self.n_negatives = args.n_negatives # Number of negative samples per step
      self.global_iter = 0
      self.global_epoch = 0
      self.D = 80
      self.ds_ratio = 1
      
      if args.model_type == 'blstm':
        self.encoder = cuda(BLSTM(self.K), self.cuda)
        self.K = 2*self.K
      if args.model_type == 'gumbel_blstm':
        self.encoder = cuda(GumbelBLSTM(self.K), self.cuda)
        self.K = 49
      if args.model_type == 'vq_lstm':
        self.encoder = cuda(VQCPCEncoder(80,
                                         channels=self.K,
                                         n_embeddings=512,
                                         z_dim=self.K,
                                         c_dim=self.K),
                            self.cuda)
        self.D = 256
        self.K = 256
        self.ds_ratio = 2
      self.model_type = args.model_type
      self.criterion = cr.CPCUnsupervisedCriterion(nPredicts=self.n_predicts,
                                                   dimOutputAR=self.K,
                                                   dimOutputEncoder=self.D,
                                                   negativeSamplingExt=self.n_negatives)
      self.criterion = cuda(self.criterion, self.cuda)

      trainables = [p for p in self.encoder.parameters()]
      trainables += [p for p in self.criterion.parameters()]        
      
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
      self.history['abx_acc']=0.
      self.history['total_loss']=0.
      self.history['avg_loss']=0.
      self.history['epoch']=0
      self.history['iter']=0

      # Dataset
      self.data_loader = return_data(args)
      
  def set_mode(self,mode='train'):
      if mode == 'train':
        self.encoder.train()
      elif mode == 'eval':
        self.encoder.eval()
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
          masks = Variable(cuda(masks, self.cuda))
          spk_labels = torch.zeros((x.size(0),), dtype=torch.int, device=x.device)
          if self.model_type == 'blstm':
            c_feature,  = self.encoder(x, masks=masks)
          elif self.model_type == 'vq_lstm':
            x, c_feature, vq_loss = self.encoder(x)
            x = x.permute(0, 2, 1)
          else:
            _, _, c_feature = self.encoder(x, masks=masks, return_feat='rnn')
          
          loss, acc = self.criterion(c_feature, x.permute(0, 2, 1), spk_labels)
          loss = loss.sum()
          if self.model_type == 'vq_lstm':
            loss = loss + vq_loss

          acc = acc.mean(0).cpu().numpy()
          total_loss += loss.cpu().detach().numpy()
          total_step += audios.size(0)

          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

          if self.global_iter % 100 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
            print(f'i:{self.global_iter:d} avg loss (total loss):{total_loss / total_step:.2f} ({total_loss:.2f}) acc at step 1:{acc[0]:.4f} acc at step last:{acc[-1]:.4f}')

        if (self.global_epoch % 2) == 0 : self.scheduler.step()
        compute_abx = False
        if self.global_epoch % 5 == 0:
          compute_abx = True
        self.test(compute_abx=compute_abx)

          
  def test(self, save_ckpt=True, compute_abx=False):
      self.set_mode('eval')

      total_loss = 0
      correct = 0
      total_num = 0
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

          spk_labels = torch.zeros((audios.size(0),), dtype=torch.int, device=x.device)
          if self.model_type == 'blstm':
            c_feature = self.encoder(x, masks=masks)
          elif self.model_type == 'vq_lstm':
            x, c_feature, vq_loss = self.encoder(x)
            x = x.permute(0, 2, 1)
          else:
            _, _, c_feature = self.encoder(x, masks=masks, return_feat='bottleneck')
          loss, acc = self.criterion(c_feature, x.permute(0, 2, 1), spk_labels) 
          if self.model_type == 'vq_lstm':
            loss = loss + vq_loss
          
          total_loss += loss.sum().cpu().detach().numpy()
          correct += acc.sum(dim=0).mean().item() * x.size(0)
          total_num += x.size(0)

          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            example_id = self.data_loader['test'].dataset.dataset[global_idx][0].split('/')[-1]
            text = self.data_loader['test'].dataset.dataset[global_idx][1]
            feat_id = f'{example_id}_{global_idx}'
            feat_fn = self.ckpt_dir.joinpath(f'feats/{feat_id}.npy')
            np.save(feat_fn, c_feature[idx].cpu().detach().numpy())
            seq_list.append((feat_id, feat_fn))
      avg_loss = total_loss / total_num
      avg_acc = correct / total_num  
      if self.history['acc'] < avg_acc:
        self.history['acc'] = avg_acc
        self.history['loss'] = avg_loss.item()
        self.history['epoch'] = self.global_epoch
        self.history['iter'] = self.global_iter
        if save_ckpt : self.save_checkpoint('best_acc.tar')
      
      print('[TEST RESULT]')
      print('e:{} loss:{:.2f} acc:{:.4f} err:{:.4f} best acc:{:.4f}'.format(self.global_epoch, avg_loss, avg_acc, 1 - avg_acc, self.history['acc']))
      if compute_abx:
        # Compute ABX score
        path_item_file = os.path.join(self.data_loader['test'].dataset.data_path, 'abx_triplets.item')
        abx_score = ev.ABX(load_feature,
                           path_item_file,
                           seq_list,
                           distance_mode='cosine',
                           step_feature=160*self.ds_ratio,
                           modes=['within'])
        abx_score = abx_score['within']
        # XXX if self.history['abx_acc'] < (1-abx_score).item():
        #   self.history['abx_acc'] = (1-abx_score).item() 
        # print('abx error:{:.4f} abx acc:{:.4f} best abx acc:{:.4f}'.format(abx_score.item(), 1-abx_score.item(), self.history['abx_acc']))
        
              
      self.set_mode('train')

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'net':self.encoder.state_dict(),
      'criterion':self.criterion.state_dict()
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
      self.criterion.load_state_dict(checkpoint['model_states']['criterion'])
      print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
    else:
      print("=> no checkpoint found at '{}'".format(file_path))

def load_feature(path):
  feat = np.load(path)
  return torch.FloatTensor(feat).unsqueeze(0)
