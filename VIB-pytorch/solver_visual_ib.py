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
from pyhocon import ConfigFactory
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from utils.utils import cuda, str2bool
from model import GumbelBLSTM
from criterion import MacroTokenFLoss
from datasets.datasets import return_data
from utils.evaluate import compute_accuracy, compute_token_f1, compute_edit_distance

EPS = 1e-10
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
    self.max_normalize = config.get('max_normalize', False)
    self.loss_type = config.get('loss_type', 'macro_token_floss')
    self.beta_f_measure = config.get('beta_f_measure', 0.3)
    self.ckpt_dir = Path(config.ckpt_dir)
    if not self.ckpt_dir.exists(): 
      self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    self.get_feature_config(config)
    self.get_dataset_config(config)
    self.get_model_config(config)
    self.get_loss_config(config)
    self.get_optim_config(config)

    self.load_ckpt = config.load_ckpt
    if self.load_ckpt or config.mode in ['test', 'cluster']: 
      self.load_checkpoint(f'best_acc_{self.config.seed}.tar')
    
    # History
    self.history = dict()
    self.history['token_f1']=0.
    self.history['word_acc']=0. 
    self.history['loss']=0.
    self.history['epoch']=0
    self.history['iter']=0

  def get_dataset_config(self, config):
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
    else:
      raise ValueError(f"Feature type {config.audio_feature} not supported")

  def get_model_config(self, config):
    self.audio_net = cuda(GumbelBLSTM(self.K,
                                n_layers=self.n_layers,
                                n_gumbel_units=self.n_phone_class,
                                n_class=self.n_visual_class,
                                input_size=self.input_size,
                                ds_ratio=1,
                                bidirectional=True), self.cuda)
    # self.image_net = cuda(Resnet34(pretrained=True, n_class=self.n_visual_class)
    # if config.get(image_net_path, None):
    #   self.image_net.load_state_dict(config.image_net_path)

  def get_loss_config(self, config):
    if self.loss_type == 'macro_token_floss':
      self.criterion = MacroTokenFLoss(beta=self.beta_f_measure)
    elif self.loss_type == 'cross_entropy':
      self.criterion = MacroTokenFLoss(beta=self.beta_f_measure)
    else:
      raise ValueError(f'Invalid loss type {self.loss_type}')

  def get_optim_config(self, config):
    trainables = [p for p in self.audio_net.parameters()]
    optim_type = config.get('optim', 'adam')
    if optim_type == 'sgd':
      self.optim = optim.SGD(trainables, lr=self.lr)
    else:
      self.optim = optim.Adam(trainables,
                              lr=self.lr, betas=(0.5, 0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)

  def train(self, save_embedding=False):
    self.set_mode('train')
    preprocessor = self.data_loader['train'].dataset.preprocessor
    total_loss = 0.
    total_word_loss = 0.
    total_phone_loss = 0.
    total_step = 0
        
    for e in range(self.epoch):
      self.global_epoch += 1
      pred_phone_labels = []
      gold_phone_labels = []
      for idx, batch in enumerate(self.data_loader['train']):
        if idx > 2 and self.debug:
          break
        self.global_iter += 1
        
        audios = batch[0]
        word_labels = batch[2]
        audio_masks = batch[3]
        word_masks = batch[5]
        # images = batch[6]

        x = cuda(audios, self.cuda)
        if self.audio_feature == "wav2vec2":
          x = self.audio_feature_net.feature_extractor(x)
        word_labels = cuda(word_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)
        # (batch size, max word num, max word len, max audio len)
        word_masks = cuda(word_masks, self.cuda)
        # (batch size x max word num, max word len)
        word_masks_2d = torch.where(word_masks.sum(-1).view(-1, self.max_word_len) > 0,
                                    torch.tensor(1, device=x.device),
                                    torch.tensor(0, device=x.device))
        # (batch size x max word num)
        word_masks_1d = torch.where(word_masks_2d.sum(-1) > 0,
                                    torch.tensor(1, device=x.device),
                                    torch.tensor(0, device=x.device))
        # images = cuda(images, self.cuda)

        if self.audio_net.ds_ratio > 1:
          audio_masks = audio_masks[:, ::self.audio_net.ds_ratio]
          word_masks = word_masks[:, :, ::self.audio_net.ds_ratio]
        audio_lens = audio_masks.sum(-1).long()
        word_lens = word_masks.sum(dim=(-1, -2)).long()
        sent_phone_logits,\
        word_logits,\
         _,\
         embedding = self.audio_net(x,
                                    masks=audio_masks,
                                    return_feat=True)

        # (batch size, max word num, max word len, n phone class)
        phone_logits = torch.matmul(word_masks, sent_phone_logits.unsqueeze(1))
        # (batch size x max word num, max word len, n phone class)
        phone_logits = phone_logits.view(-1, self.max_word_len, self.n_phone_class)
       
        # (batch size, max word num, max word len, n word class)
        word_logits = torch.matmul(word_masks, word_logits.unsqueeze(1))
        # (batch size, max word num, n word class)
        word_logits = (word_logits * word_masks.sum(-1, keepdim=True)).sum(-2) # / (word_lens.unsqueeze(-1) + EPS) # Average over frames 
        # (batch size x max word num, n word class)
        word_logits_2d = word_logits.view(-1, self.n_visual_class)

        loss = F.cross_entropy(word_logits_2d,
                               word_labels.flatten(),
                               ignore_index=self.ignore_index)
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.

        if loss == 0:
          continue
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.global_iter % 1000 == 0:
          avg_loss = total_loss / total_step
          print(f'Itr {self.global_iter:d}\tAvg Loss (Total Loss):{avg_loss:.2f} ({total_loss:.2f})')
      
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
    total_word_loss = 0.
    total_phone_loss = 0.
    total_step = 0.

    pred_word_labels = []
    gold_word_labels = []
    if not self.ckpt_dir.joinpath('outputs/phonetic/dev-clean').is_dir():
      os.makedirs(self.ckpt_dir.joinpath('outputs/phonetic/dev-clean'))

    gold_phone_file = os.path.join(testset.data_path, f'{testset.splits[0]}/{testset.splits[0]}_nonoverlap.item')
    word_readable_f = open(self.ckpt_dir.joinpath(f'{out_prefix}_visual_word.{self.global_epoch}.readable'), 'w') 
    phone_file = self.ckpt_dir.joinpath(f'{out_prefix}_phoneme.{self.global_epoch}.txt')
    phone_f = open(phone_file, 'w')

    with torch.no_grad():
      B = 0
      for b_idx, batch in enumerate(self.data_loader['test']):        
        audios = batch[0]
        word_labels = batch[2]
        audio_masks = batch[3]
        word_masks = batch[5]
        if b_idx > 2 and self.debug:
          break
        if b_idx == 0: 
          B = audios.size(0)
        
        x = cuda(audios, self.cuda)
        if self.audio_feature == "wav2vec2":
          x = self.audio_feature_net.feature_extractor(x)  
        word_labels = cuda(word_labels, self.cuda)
        audio_masks = cuda(audio_masks, self.cuda)
        # (batch size, max word num, max word len, max audio len)
        word_masks = cuda(word_masks, self.cuda)
        # (batch size x max word num, max word len)
        word_masks_2d = torch.where(word_masks.sum(-1).view(-1, self.max_word_len) > 0,
                                    torch.tensor(1, device=x.device),
                                    torch.tensor(0, device=x.device))

        # (batch size x max word num)
        word_masks_1d = torch.where(word_masks_2d.sum(-1) > 0,
                                    torch.tensor(1, device=x.device),
                                    torch.tensor(0, device=x.device))
        # images = cuda(images, self.cuda)

        if self.audio_net.ds_ratio > 1:
          audio_masks = audio_masks[:, ::self.audio_net.ds_ratio]
          word_masks = word_masks[:, :, ::self.audio_net.ds_ratio]
        audio_lens = audio_masks.sum(-1).long()
        # (batch size, max word num)
        word_lens = word_masks.sum(dim=(-1, -2)).long()
        # (batch size,)
        word_nums = torch.where(word_lens > 0,
                                torch.tensor(1, device=x.device),
                                torch.tensor(0, device=x.device)).sum(-1)

        sent_phone_logits,\
        word_logits,\
         _,\
         embedding = self.audio_net(x,
                                    masks=audio_masks,
                                    return_feat=True)
        
        # (batch size, max word num, max word len, n phone class)
        phone_logits = torch.matmul(word_masks, sent_phone_logits.unsqueeze(1))
        # (batch size x max word num, max word len, n phone class)
        phone_logits = phone_logits.view(-1, self.max_word_len, self.n_phone_class)
                
        # (batch size, max word num, max word len, n word class)
        word_logits = torch.matmul(word_masks, word_logits.unsqueeze(1))
        # (batch size, max word num, n word class)
        word_logits = (word_logits * word_masks.sum(-1, keepdim=True)).sum(-2) # / (word_lens.unsqueeze(-1) + EPS) # Average over frames 
        # (batch size x max word num, n word class)
        word_logits_2d = word_logits.view(-1, self.n_visual_class)

        loss = F.cross_entropy(word_logits_2d,
                               word_labels.flatten(),
                               ignore_index=self.ignore_index)
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.
        
        for idx in range(audios.size(0)):
          global_idx = b_idx * B + idx
          audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0]
          pred_phone_label = sent_phone_logits[idx, :audio_lens[idx]].max(-1)[1]
          pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
          pred_phone_names = ','.join([str(p) for p in pred_phone_label_list])
          phone_f.write(f'{audio_id} {pred_phone_names}\n')

          if word_nums[idx] > 0:
            gold_word_label = word_labels[idx, :word_nums[idx]].cpu().detach().numpy().tolist()
            pred_word_label = word_logits[idx, :word_nums[idx]].max(-1)[1].cpu().detach().numpy().tolist() 
         
            gold_word_labels.extend(gold_word_label)
            pred_word_labels.extend(pred_word_label)
            pred_word_names = preprocessor.to_word_text(pred_word_label)
            gold_word_names = preprocessor.to_word_text(gold_word_label)
            

            for word_idx in range(word_nums[idx]):
              pred_word_name = pred_word_names[word_idx]
              gold_word_name = gold_word_names[word_idx]
              word_readable_f.write(f'Utterance id: {audio_id}\n'
                                    f'Gold word label: {gold_word_name}\n'
                                    f'Pred word label: {pred_word_name}\n\n') 
      phone_f.close()
      word_readable_f.close()
      avg_loss = total_loss / total_step
      # Compute word accuracy and word token F1
      print('[TEST RESULT]')
      word_acc = compute_accuracy(gold_word_labels, pred_word_labels)
      word_prec,\
      word_rec,\
      word_f1, _ = precision_recall_fscore_support(np.asarray(gold_word_labels),
                                                   np.asarray(pred_word_labels),
                                                   average='macro')
      print(f'Epoch {self.global_epoch}\tLoss: {avg_loss:.4f}\tWord Loss: {avg_word_loss}\tPhone Loss: {avg_phone_loss}\n'
            f'WER: {1-word_acc:.3f}\tWord Acc.: {word_acc:.3f}\n'
            f'Word Precision: {word_prec:.3f}\tWord Recall: {word_rec:.3f}\tWord F1: {word_f1:.3f}') 
      token_f1,\
      token_prec,\
      token_recall = compute_token_f1(phone_file,
                                      gold_phone_file,
                                      self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))


      if self.history['word_acc'] < word_acc:
        self.history['token_f1'] = token_f1
        self.history['word_acc'] = word_acc
        self.history['loss'] = avg_loss
        self.history['iter'] = self.global_iter
        self.history['epoch'] = self.global_epoch
        self.save_checkpoint(f'best_acc_{self.config.seed}.tar')
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

def main(argv):
  parser = argparse.ArgumentParser(description='Visual macro token F1 maximizer')
  parser.add_argument('CONFIG', type=str)
  args = parser.parse_args(argv)

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

  config = ConfigFactory.parse_file(args.CONFIG)
  if not config.dset_dir:
    config.dset_dir = "/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic" 
  
  avg_word_acc = []
  avg_token_f1 = []
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
    else:
      return 0
    avg_word_acc.append(net.history['word_acc'])
    avg_token_f1.append(net.history['token_f1'])

  avg_word_acc = np.asarray(avg_word_acc)
  avg_token_f1 = np.asarray(avg_token_f1)
  print(f'Average Word Acc.: {np.mean(avg_word_acc)}+/-{np.std(avg_word_acc)}\n'
        f'Average Token F1: {np.mean(avg_token_f1)}+/-{np.std(avg_token_f1)}') 


if __name__ == '__main__':
  argv = sys.argv[1:]
  main(argv)    
