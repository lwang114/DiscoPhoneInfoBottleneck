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
from sklearn.metrics import precision_recall_fscore_support 
from datasets.datasets import return_data
from utils import cuda
import cpc.criterion as cr
import cpc.eval as ev
from model import GumbelBLSTM, GumbelPyramidalBLSTM, GumbelMarkovBLSTM
from pathlib import Path
from image_model import Resnet34 
from evaluate import evaluate

class Solver(object):


  def __init__(self, args):
      self.args = args

      self.cuda = torch.cuda.is_available()
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
      self.image_feature = args.image_feature
      self.loss_type = args.loss_type
      self.dataset = args.dataset

      # Dataset
      self.data_loader = return_data(args)
      self.n_class = self.data_loader['train']\
                     .dataset.preprocessor.num_tokens
      self.class_names = self.data_loader['train']\
                         .dataset.preprocessor.tokens
      self.pos_weight = cuda(args.pos_weight * torch.ones(self.n_class), self.cuda)
      print(f'visual label class: {self.n_class}') # XXX
  
      self.image_net = Resnet34(pretrained=True, n_class=self.n_class)
      self.image_net.load_state_dict(
                       torch.load(
                         os.path.join(args.image_model_dir,
                                      'best_image_model.pth'
                         )
                       )
                     ) 
      self.image_net = cuda(self.image_net, self.cuda)

      if args.model_type == 'gumbel_blstm':
        self.ds_ratio = 1
        bidirectional = False if 'CPC' in args.loss_type else True
        self.audio_net = cuda(GumbelBLSTM(
                          self.K, 
                          n_class=self.n_class, 
                          ds_ratio=self.ds_ratio,
                          bidirectional=bidirectional),
                     self.cuda)
        self.K = 2*self.K if bidirectional else self.K 
      elif args.model_type == 'pyramidal_blstm':
        self.ds_ratio = 4
        bidirectional = False if 'CPC' in args.loss_type else True
        self.audio_net = cuda(GumbelPyramidalBLSTM(
                          self.K, 
                          n_class=self.n_class, 
                          ds_ratio=self.ds_ratio, 
                          bidirectional=bidirectional), self.cuda)
        self.K = 2*self.K if bidirectional else self.K
        self.audio_net.weight_init()
      elif args.model_type == 'gumbel_markov_blstm':
        self.ds_ratio = 1
        self.audio_net = cuda(GumbelMarkovBLSTM(self.K), self.cuda)
        self.audio_net.weight_init()

      self.cpc_criterion = cr.CPCUnsupervisedCriterion(
                             nPredicts=self.n_predicts,
                             dimOutputAR=self.K if self.cpc_feature=='rnn' 
                                                else 49,
                             dimOutputEncoder=80,
                             negativeSamplingExt=self.n_negatives,
                             rnnMode=False)
      self.cpc_criterion = cuda(self.cpc_criterion, self.cuda)

      trainables = [p for p in self.audio_net.parameters()]
      trainables += [p for p in self.cpc_criterion.parameters()]        
      
      self.optim = optim.Adam(trainables,
                              lr=self.lr,betas=(0.5,0.999))
      self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)

      self.ckpt_dir = Path(args.ckpt_dir)
      if not self.ckpt_dir.exists(): 
        self.ckpt_dir.mkdir(parents=True,exist_ok=True)
      self.load_ckpt = args.load_ckpt
      if self.load_ckpt != '': 
        self.load_checkpoint(self.load_ckpt)
      
      # History
      self.history = dict()
      self.history['f1']=0. 
      self.history['token_f1']=0.
      self.history['abx']=0.5
      self.history['total_loss']=0.
      self.history['avg_loss']=0.
      self.history['epoch']=0
      self.history['iter']=0
      
  def set_mode(self,mode='train'):
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
      anneal_rate = 3e-6
      temp = 1.

      total_loss = 0.
      total_step = 0
      for e in range(self.epoch):
        self.global_epoch += 1
        pred_labels = []
        gold_labels = []
        pred_dicts = []
        for idx, (audios, images, _, audio_masks, image_masks)\
            in enumerate(self.data_loader['train']):
          if idx > 2: # XXX
            break
          self.global_iter += 1
          x = cuda(audios, self.cuda)
          audio_masks = cuda(audio_masks, self.cuda)
          images = cuda(images, self.cuda)
          image_masks = cuda(image_masks.unsqueeze(-1), self.cuda)
          image_size = images.size()

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

          # Compute IB loss
          in_logit, logits, c_feature = self.audio_net(
                                          x, masks=audio_masks, 
                                          temp=temp, 
                                          return_feat=self.cpc_feature)
          logit = logits.max(-2)[0]
          
          if self.image_feature == 'label' and images.ndim in [2, 4]:
              class_loss = F.cross_entropy(logit, y).div(math.log(2))
          else:
            class_loss = F.binary_cross_entropy_with_logits(
                            logit, y, pos_weight=self.pos_weight
                         ).div(math.log(2))

          info_loss = (F.softmax(in_logit, dim=-1)\
                        * F.log_softmax(in_logit, dim=-1)
                      ).sum(1).mean().div(math.log(2))
          ib_loss = class_loss + self.beta * info_loss
          
          izy_bound = math.log(self.n_class,2) - class_loss
          izx_bound = info_loss

          if self.image_feature == 'label' and images.ndim in [2, 4]: 
            pred_label = F.one_hot(logit.max(-1)[1], self.n_class)
            gold_label = F.one_hot(y, self.n_class)
            pred_labels.append(pred_label.cpu())
            gold_labels.append(gold_label.cpu())
          else:
            pred_labels.append((logit > 0).long().cpu())
            gold_labels.append((y > 0).long().cpu())
          
          # Compute CPC loss
          spk_labels = torch.zeros(x.size(0), 
                                   dtype=torch.int, 
                                   device=x.device)
          cpc_loss, cpc_acc = self.cpc_criterion(c_feature, 
                                                 x.permute(0, 2, 1), 
                                                 spk_labels)
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

          if self.global_iter % 1000 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
            avg_loss = total_loss / total_step
            print(f'i:{self.global_iter:d} avg loss (total loss):{avg_loss:.2f} ({total_loss:.2f}) '
                  f'IZY:{izy_bound:.2f} IZX:{izx_bound:.2f} '
                  f'cpc acc at step 1:{cpc_acc[0]:.4f} cpc acc at step last:{cpc_acc[-1]:.4f}')
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
        # XXX if self.global_epoch % 5 == 0:
        #   compute_abx = True
        self.test(compute_abx=compute_abx)

          
  def test(self, save_ckpt=True, compute_abx=False, out_prefix='predictions'):
      self.set_mode('eval')
      testset = self.data_loader['test'].dataset

      total_loss = 0
      izy_bound = 0
      izx_bound = 0
      cpc_correct = 0
      total_num = 0
      pred_dicts = []
      gold_dicts = self.data_loader['test'].dataset.gold_dicts
      seq_list = []
      pred_labels = []
      gold_labels = []
      if not self.ckpt_dir.joinpath('feats').is_dir():
        self.ckpt_dir.joinpath('feats').mkdir()

      out_file = os.path.join(
                   self.ckpt_dir,
                   f'{out_prefix}.{self.global_epoch}.readable'
                 )
      f = open(out_file, 'w')
      f.write('Image ID\tGold label\tPredicted label\n')      
      with torch.no_grad():
        B = 0
        for b_idx, (audios, images, _, audio_masks, image_masks) in enumerate(self.data_loader['test']):
          if b_idx > 2: # XXX
            break
          if b_idx == 0:
            B = audios.size(0)
          audios = cuda(audios, self.cuda)
          images = cuda(images, self.cuda)
          audio_masks = cuda(audio_masks, self.cuda)
          image_masks = cuda(image_masks.unsqueeze(-1), self.cuda)
          image_size = images.size()

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
          
          in_logit, logits, encoding = self.audio_net(
                                         audios, masks=audio_masks, 
                                         return_feat='bottleneck'
                                       )
          _, _, c_feature = self.audio_net(
                              audios, masks=audio_masks, 
                              return_feat=self.cpc_feature
                            )

          # Word prediction
          logit = logits.max(-2)[0]
          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            image_id = testset.dataset[global_idx][1]
            if (self.image_feature == 'label') and (images.ndim in [2, 4]):
              golds = [y[idx].cpu().detach().numpy()] 
              preds = [logit[idx].max(-1)[1].cpu().detach().numpy()]
            else:
              golds = y[idx].nonzero()\
                      .squeeze(-1).cpu().detach().numpy()
              preds = (logit[idx] > 0)\
                      .long().nonzero().squeeze(-1).cpu().detach().numpy() 

            gold_names = ','.join([self.class_names[c] for c in golds])  
            pred_names = ','.join([self.class_names[c] for c in preds])
            f.write(f'{image_id}\t{gold_names}\t{pred_names}\n')

          if (self.image_feature == 'label') and (images.ndim in [2, 4]): 
            pred_labels.append(F.one_hot(y, self.n_class).cpu())
            gold_labels.append(F.one_hot(logit.max(-1)[1], self.n_class).cpu())
            cur_class_loss = F.cross_entropy(logit, y).div(math.log(2))
          else:
            pred_labels.append((logit > 0).long().cpu())
            gold_labels.append((y > 0).long().cpu())
            cur_class_loss = F.binary_cross_entropy_with_logits(
                                logit, y,
                                size_average=False).div(math.log(2))
                    
          cur_info_loss = (F.softmax(in_logit,dim=-1)\
                            * F.log_softmax(in_logit, dim=-1)
                          ).sum(1).mean().div(math.log(2))
          cur_ib_loss = cur_class_loss + self.beta * cur_info_loss
          izy_bound = izy_bound\
                      + y.size(0) * math.log(self.n_class, 2)\
                      - cur_class_loss
          izx_bound = izx_bound + cur_info_loss
                    
          # CPC prediction
          spk_labels = torch.zeros(audios.size(0), 
                                   dtype=torch.int, 
                                   device=audios.device)
          cur_cpc_loss, cpc_acc = self.cpc_criterion(
                                    c_feature, 
                                    audios.permute(0, 2, 1), 
                                    spk_labels
                                  ) 
          cpc_correct += cpc_acc.sum(dim=0).mean().item() * audios.size(0)
          total_loss += cur_cpc_loss.sum().item() + cur_ib_loss.item()
          total_num += audios.size(0)
          _, _, c_feature = self.audio_net(audios, masks=audio_masks, return_feat='rnn')
          for idx in range(audios.size(0)):
            global_idx = b_idx * B + idx
            example_id = testset.dataset[global_idx][0].split('/')[-1]
            text = testset.dataset[global_idx][1]
            feat_id = f'{example_id}_{global_idx}'      
            units = encoding[idx].max(-1)[1]
            pred_dicts.append({'sent_id': example_id,
                               'units': units.cpu().detach().numpy().tolist(),  
                               'text': text})

            if global_idx <= 100:
              feat_fn = self.ckpt_dir.joinpath(f'feats/{feat_id}.npy')
              np.save(feat_fn, c_feature[idx].cpu().detach().numpy())
              seq_list.append((feat_id, feat_fn))
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
      avg_cpc_acc = cpc_correct / total_num
      token_f1, conf_df, token_prec, token_recall = evaluate(
                                                      pred_dicts, 
                                                      gold_dicts, 
                                                      ds_rate=self.ds_ratio)

      if self.history['f1'] < f1:
        self.history['f1'] = f1
        self.history['loss'] = avg_loss
        self.history['epoch'] = self.global_epoch
        self.history['iter'] = self.global_iter
        self.history['token_f1'] = token_f1
        if save_ckpt : self.save_checkpoint('best_acc.tar')
        conf_df.to_csv(self.ckpt_dir.joinpath('confusion_matrix.csv'))
      print('[TEST RESULT]')
      print('Epoch {}\tLoss: {:.2f}\tPrecision: {:.2f}\tRecall: {:.2f}\tF1: {:.2f}'\
            .format(self.global_epoch, avg_loss, p, r, f1))
      print('Top 10 F1: {:.2f}\tBest F1: {:.2f}'\
            .format(class_f1s[:10].mean(), self.history['f1']))
      print('Phone Token Precision: {:.4f}\tPhone Token Recall: {:.4f}\tPhone Token F1: {:.4f}'\
            .format(token_prec, token_recall, token_f1)) 
      print('CPC Average Accuracy: {:.4f}\tCPC Average Error: {:.4f}'\
            .format(avg_cpc_acc, 1 - avg_cpc_acc))
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

  def save_checkpoint(self, filename='best_acc.tar'):
    model_states = {
      'net':self.audio_net.state_dict(),
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

      self.audio_net.load_state_dict(checkpoint['model_states']['net'])
      self.cpc_criterion.load_state_dict(checkpoint['model_states']['cpc_criterion'])
      print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
    else:
      print("=> no checkpoint found at '{}'".format(file_path))

def load_feature(path):
  feat = np.load(path)
  return torch.FloatTensor(feat).unsqueeze(0)
