import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from copy import deepcopy
import fairseq
import argparse
import sys
import os
import json
import time
import shutil
import numpy as np
import argparse
from pyhocon import ConfigFactory
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from utils.utils import cuda, str2bool
from model import GumbelBLSTM, GumbelMLP
from criterion import MacroTokenFLoss
from datasets.datasets import return_data
from utils.evaluate import compute_accuracy, compute_token_f1, compute_edit_distance

EPS = 1e-10
NULL = "###NULL###"
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
    self.beta = config.beta # degree of compression
    self.global_iter = 0
    self.global_epoch = 0
    self.audio_feature = config.audio_feature
    self.image_feature = config.image_feature
    self.debug = config.debug
    self.dataset = config.dataset
    self.ckpt_dir = Path(config.ckpt_dir)
    if not self.ckpt_dir.exists(): 
      self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    self.get_feature_config(config)
    self.get_dataset_config(config)
    self.get_model_config(config)
    self.get_optim_config(config)

    self.load_ckpt = config.load_ckpt
    if self.load_ckpt or config.mode in ['test', 'cluster', 'test_oos']: 
      self.load_checkpoint(f'best_acc_{self.config.seed}.tar')
    
    # History
    self.history = dict()
    self.history['token_result']=[0., 0., 0.]
    self.history['word_acc']=0. 
    self.history['loss']=0.
    self.history['epoch']=0
    self.history['iter']=0

  def get_dataset_config(self, config):
    self.data_loader = return_data(config)
    self.oos_dataset_name = config.get('oos_dataset', None) 
    if self.oos_dataset_name:
      oos_config = deepcopy(config)
      oos_config['dataset'] = oos_config['oos_dataset']
      oos_config['dset_dir'] = oos_config['oos_dset_dir']
      oos_config['splits'] = {'train': oos_config['splits']['test_oos'],
                              'test': oos_config['splits']['test_oos']} 
      oos_data_loader = return_data(oos_config)
      self.data_loader['test_oos'] = oos_data_loader['test']
    self.dataset_name = config.dataset
    self.ignore_index = config.get('ignore_index', -100)

    if self.config.get('n_visual_class', None):
      self.n_visual_class = self.config['n_visual_class']
    else:
      self.n_visual_class = self.data_loader['train']\
                            .dataset.preprocessor.num_visual_words
    if self.config.get('n_clusters', None):
      self.n_phone_class = self.config['n_clusters']   
    else:
      self.n_phone_class = self.data_loader['train'].dataset.preprocessor.num_tokens
    self.visual_words = self.data_loader['train'].dataset.preprocessor.visual_words
    self.phone_set = self.data_loader['train'].dataset.preprocessor.tokens
    self.max_feat_len = self.data_loader['train'].dataset.max_feat_len
    self.max_word_len = self.data_loader['train'].dataset.max_word_len
    self.max_segment_num = self.data_loader['train'].dataset.max_segment_num
    self.n_clusters = config.get("n_clusters", self.n_phone_class)
    print(f'Number of visual label classes = {self.n_visual_class}')
    print(f'Number of phone classes = {self.n_phone_class}')
    print(f'Number of clusters = {self.n_clusters}')

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
    if config.model_type.split('_')[0] == 'pd':
      position_dependent = True
    else:
      position_dependent = False

    if (config.model_type in ['gumbel_blstm', 'pd_gumbel_blstm']) and (config.downsample_method == 'no-op'):
      self.audio_net = cuda(GumbelBLSTM(self.K,
                                        n_layers=self.n_layers,
                                        n_gumbel_units=self.n_clusters,
                                        n_class=self.n_visual_class,
                                        input_size=self.input_size,
                                        ds_ratio=1,
                                        bidirectional=True,
                                        max_len=self.max_segment_num,
                                        position_dependent=position_dependent), self.cuda)
    elif (config.model_type in ['gumbel_mlp', 'pd_gumbel_mlp']) and (config.downsample_method != 'no-op'):
      self.audio_net = cuda(GumbelMLP(self.K,
                                      n_layers=self.n_layers,
                                      n_gumbel_units=self.n_phone_class,
                                      n_class=self.n_visual_class,
                                      input_size=self.input_size,
                                      max_len=self.max_segment_num,
                                      position_dependent=position_dependent), self.cuda)
    else:
      raise ValueError(f'Invalid model type: {config.model_type} for downsample method {config.downsample_method}')

  def get_optim_config(self, config):
    trainables = [p for p in self.audio_net.parameters()]
    optim_type = config.get('optim', 'adam')
    if optim_type == 'sgd':
      self.optim = optim.SGD(trainables, 
                             momentum=0.9,
                             lr=self.lr)
    else:
      self.optim = optim.Adam(trainables,
                              lr=self.lr, betas=(0.5, 0.999))
    self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.97)

  def extract_wav2vec2(self, x, mask):
    B = x.size(0)
    T = x.size(1)
    out = self.audio_feature_net.feature_extractor(x.view(B*T, -1)).permute(0, 2, 1)
    out = out.view(B, T, out.size(-2), out.size(-1))
    out = (out * mask.unsqueeze(-1)).sum(-2) / (mask.sum(-1, keepdim=True) + torch.tensor(1e-10, device=x.device))
    out_mask = (mask.sum(-1) > 0).float()
    return out, out_mask 

  def train(self, save_embedding=False):
    self.set_mode('train')
    preprocessor = self.data_loader['train'].dataset.preprocessor
    total_loss = 0.
    total_step = 0.
    total_word_loss = 0.
        
    for e in range(self.epoch):
      if self.debug and e > 1:
        break
      self.global_epoch += 1
      pred_phone_labels = []
      gold_phone_labels = []
      for idx, batch in enumerate(self.data_loader['train']):
        if idx > 2 and self.debug:
          break
        self.global_iter += 1
        
        audios = batch[0]
        word_labels = batch[2].squeeze(-1)
        audio_masks = batch[3]

        # (batch size, max segment num, feat dim) or (batch size, max segment num, max segment len, feat dim)
        x = cuda(audios, self.cuda)

        # (batch size,)
        word_labels = cuda(word_labels, self.cuda)
        
        # (batch size, max segment num) or (batch size, max segment num, max segment len)
        audio_masks = cuda(audio_masks, self.cuda)
        
        if self.audio_feature == "wav2vec2":
          x, audio_masks = self.extract_wav2vec2(x, audio_masks)
         
        # (batch size, max segment num)
        if audio_masks.dim() == 3:
          segment_masks = torch.where(audio_masks.sum(-1) > 0,
                                      torch.tensor(1., device=audio_masks.device),
                                      torch.tensor(0., device=audio_masks.device))
        else:
          segment_masks = audio_masks.clone()

        if self.audio_net.ds_ratio > 1:
          audio_masks = audio_masks[:, ::self.audio_net.ds_ratio]
          segment_masks = segment_masks[:, ::self.audio_net.ds_ratio]

        # (batch size, max segment num, n visual class)
        phone_logits, word_logits = self.audio_net(x, masks=audio_masks,
                                        return_feat=False)

        # (batch size * max segment num, n visual class)
        segment_word_logits = word_logits.sum(-2)

        loss = F.cross_entropy(segment_word_logits,
                               word_labels,
                               ignore_index=self.ignore_index)
        loss = loss + self.beta * (F.softmax(phone_logits, dim=-1) * F.log_softmax(phone_logits, dim=-1)).sum((-1, -2)).mean()
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.
        
        if loss == 0:
          continue
        self.optim.zero_grad()
        loss.backward()        
        self.optim.step()

        if (self.global_iter-1) % 1000 == 0:
          avg_loss = total_loss / total_step
          print(f'Itr {self.global_iter:d}\tAvg Loss (Total Loss):{avg_loss:.2f} ({total_loss:.2f})')
      
      avg_loss = total_loss / total_step
      print(f'Epoch {self.global_epoch}\tTraining Loss: {avg_loss:.3f}')

      if (self.global_epoch % 2) == 0:
        self.scheduler.step()
      self.test(save_embedding=save_embedding)
      if self.oos_dataset_name:
        self.test_out_of_sample(save_embedding=save_embedding)

  def test(self, save_embedding=False, out_prefix='predictions'):
    self.set_mode('eval')
    testset = self.data_loader['test'].dataset
    preprocessor = testset.preprocessor

    total_loss = 0.
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
        word_labels = batch[2].squeeze(-1)
        audio_masks = batch[3]
        word_masks = batch[5]
        if b_idx > 2 and self.debug:
          break
        if b_idx == 0: 
          B = audios.size(0)
 
        # (batch size, max segment num, feat dim) or (batch size, max segment num, max segment len, feat dim)
        x = cuda(audios, self.cuda)

        # (batch size,)
        word_labels = cuda(word_labels, self.cuda)

        # (batch size, max segment num) or (batch size, max segment num, max segment len)
        audio_masks = cuda(audio_masks, self.cuda)

        if self.audio_feature == "wav2vec2":
          x, audio_masks = self.extract_wav2vec2(x, audio_masks)
          
        # (batch size, max segment num)
        if audio_masks.dim() == 3: 
          segment_masks = torch.where(audio_masks.sum(-1) > 0,
                                      torch.tensor(1., device=audio_masks.device),
                                      torch.tensor(0., device=audio_masks.device))
        else:
          segment_masks = audio_masks.clone()
             
        if self.audio_net.ds_ratio > 1:
          audio_masks = audio_masks[:, ::self.audio_net.ds_ratio]
          segment_masks = segment_masks[:, ::self.audio_net.ds_ratio]

        # (batch size, max segment num, n visual class)
        phone_logits, word_logits, _, embedding = self.audio_net(x, masks=audio_masks,
                                                                 return_feat=True)

        # segment_word_labels = word_labels.unsqueeze(-1)\
        #                                  .expand(-1, self.max_segment_num)
        # segment_word_labels = (segment_word_labels * segment_masks).flatten().long()
        segment_word_logits = word_logits.sum(-2) # XXX word_logits.view(-1, self.n_visual_class)
        loss = F.cross_entropy(segment_word_logits,
                               word_labels,
                               ignore_index=self.ignore_index)
        total_loss += loss.cpu().detach().numpy()
        total_step += 1.
        
        for idx in range(audios.size(0)):
          global_idx = b_idx * B + idx
          audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0].split('.')[0]
          
          phone_logits = phone_logits.squeeze(1)
          segments = testset.dataset[global_idx][-1]
          phone_logit_frame_level = testset.unsegment(phone_logits[idx], segments)
          pred_phone_label = phone_logit_frame_level.max(-1)[1]

          if int(self.hop_len_ms / 10) * self.audio_net.ds_ratio > 1:
            us_ratio = int(self.hop_len_ms / 10) * self.audio_net.ds_ratio
            pred_phone_label = pred_phone_label.unsqueeze(-1)\
                               .expand(-1, us_ratio).flatten()

          pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
          pred_phone_names = ','.join([str(p) for p in pred_phone_label_list])
          phone_f.write(f'{audio_id} {pred_phone_names}\n')
          
          gold_word_label = word_labels[idx].cpu().detach().numpy().tolist()
          pred_word_label = word_logits[idx].sum(0).max(-1)[1].cpu().detach().numpy().tolist() 
          gold_word_labels.append(gold_word_label)
          pred_word_labels.append(pred_word_label)
          pred_word_name = preprocessor.to_word_text([pred_word_label])[0]
          gold_word_name = preprocessor.to_word_text([gold_word_label])[0]
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
      token_f1,\
      token_prec,\
      token_recall = compute_token_f1(phone_file,
                                      gold_phone_file,
                                      self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))
      info = f'Epoch {self.global_epoch}\tLoss: {avg_loss:.4f}\n'\
             f'WER: {1-word_acc:.3f}\tWord Acc.: {word_acc:.3f}\n'\
             f'Word Precision: {word_prec:.3f}\tWord Recall: {word_rec:.3f}\tWord F1: {word_f1:.3f}\n'\
             f'Token Precision: {token_prec:.3f}\tToken Recall: {token_recall:.3f}\tToken F1: {token_f1:.3f}\n'
      print(info) 

      save_path = self.ckpt_dir.joinpath('results_file.txt')
      with open(save_path, 'a') as file:
        file.write(info)

      if self.history['token_result'][-1] < token_f1:
        self.history['token_result'] = [token_prec, token_recall, token_f1]
        self.history['word_acc'] = word_acc
        self.history['loss'] = avg_loss
        self.history['iter'] = self.global_iter
        self.history['epoch'] = self.global_epoch
        best_phone_file = self.ckpt_dir.joinpath(f'outputs_quantized_{self.config.seed}.txt')
        shutil.copyfile(phone_file, best_phone_file)
        self.save_checkpoint(f'best_acc_{self.config.seed}.tar')
      self.set_mode('train') 

  def test_out_of_sample(self, save_embedding=False):
    self.set_mode('eval')
    test_loader = self.data_loader['test_oos']
    testset = test_loader.dataset
    batch_size = test_loader.batch_size

    phone_file = self.ckpt_dir.joinpath(f'{self.dataset_name}_outputs_quantized_{self.config.seed}.txt')
    embed_file = self.ckpt_dir.joinpath(f'{self.dataset_name}_embeddings.npz')
    embed_label_file = self.ckpt_dir.joinpath(f'{self.dataset_name}_embedding_labels.json')
    embeds = dict()
    embed_labels = dict()

    phone_f = open(phone_file, 'w')
    with torch.no_grad():
      for b_idx, batch in enumerate(test_loader):
        audios = batch[0]
        input_mask = batch[3]
        x = cuda(audios, self.cuda)
        input_mask = cuda(input_mask, self.cuda)
        if self.audio_feature == 'wav2vec2':
          x, input_mask = self.extract_wav2vec2(audios, input_mask)

        phone_logits,\
        word_logits,\
        _, embedding = self.audio_net(x, masks=input_mask,
                                      return_feat=True)
        
        segment_word_logits = word_logits.sum(-2) # XXX word_logits.view(-1, self.n_visual_class)
 
        for idx in range(audios.size(0)):
          global_idx = b_idx * batch_size + idx
          audio_id = os.path.splitext(os.path.split(testset.dataset[global_idx][0])[1])[0].split('.')[0]
          if save_embedding:
            embed_id = f'{audio_id}_{global_idx}'
            embeds[embed_id] = F.softmax(word_logits[idx, :len(phonemes)], dim=-1).detach().cpu().numpy()
            embed_labels[embed_id] = {'phoneme_text': [s['text'] for s in phonemes],
                                      'word_text': [NULL]*len(phonemes)}

          phone_logits = phone_logits.squeeze(1)
          segments = testset.dataset[global_idx][-1]
          phone_logit_frame_level = testset.unsegment(phone_logits[idx], segments)
          pred_phone_label = phone_logit_frame_level.max(-1)[1]

          if int(self.hop_len_ms / 10) * self.audio_net.ds_ratio > 1:
            us_ratio = int(self.hop_len_ms / 10) * self.audio_net.ds_ratio
            pred_phone_label = pred_phone_label.unsqueeze(-1)\
                               .expand(-1, us_ratio).flatten()

          pred_phone_label_list = pred_phone_label.cpu().detach().numpy().tolist()
          pred_phone_names = ','.join([str(p) for p in pred_phone_label_list])
          phone_f.write(f'{audio_id} {pred_phone_names}\n')
      phone_f.close()

      # Save embeddings
      if save_embedding:
        np.savez(embed_file, **embeds)
        json.dump(embed_labels, open(embed_label_file, 'w') ,indent=2)

    # Evaluation with token F1
    gold_phone_files = [os.path.join(testset.data_path, f'{split}/{split}_nonoverlap.item') for split in testset.splits]
    token_f1,\
    token_prec,\
    token_recall = compute_token_f1(phone_file,
                                    gold_phone_files,
                                    self.ckpt_dir.joinpath(f'confusion.{self.global_epoch}.png'))
    print('[OOS TEST RESULT]')
    info = f'Out-of-Sample Dataset: {self.oos_dataset_name}\n'\
           f'Token Precision: {token_prec:.4f}\tToken Recall: {token_recall:.4f}\tToken F1: {token_f1:.4f}\n'

    save_path = os.path.join(self.ckpt_dir, f'results_file_{self.config.seed}.txt')
    with open(save_path, 'a') as f:
      f.write(info)
    print(info)
     
    if self.history['token_result'][-1] < token_f1:
      self.history['token_result'] = [token_prec, token_recall, token_f1]
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
  
  word_accs = []
  token_precs = []
  token_recs = []
  token_f1s = []
  for seed in config.get('seeds', [config.seed]):
    config.seed = seed
    config['seed'] = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print(f'I am process {os.getpid()}')
    print('[CONFIGS]')
    print(config)
    print()

    net = Solver(config)
    save_embedding = config.get('save_embedding', False)
    if config.mode == 'train':
      net.train(save_embedding=save_embedding)
    elif config.mode == 'test':
      net.test(save_embedding=save_embedding) 
    elif config.mode == 'test_oos':
      net.test_out_of_sample(save_embedding=save_embedding) 
    else:
      return 0
    word_accs.append(net.history['word_acc'])
    token_precs.append(net.history['token_result'][0])
    token_recs.append(net.history['token_result'][1])
    token_f1s.append(net.history['token_result'][2])

  word_accs = np.asarray(word_accs)
  token_precs = np.asarray(token_precs)
  token_recs = np.asarray(token_recs)
  token_f1s = np.asarray(token_f1s)

  mean_word_acc, std_word_acc = np.mean(word_accs), np.std(word_accs)
  mean_token_prec, std_token_prec = np.mean(token_precs), np.std(token_precs)
  mean_token_rec, std_token_rec = np.mean(token_recs), np.std(token_recs)
  mean_token_f1, std_token_f1 = np.mean(token_f1s), np.std(token_f1s) 
  print(f'Average Word Acc.: {mean_word_acc:.4f}+/-{std_word_acc:.4f}\n'
        f'Average Token Precision: {mean_token_prec:.4f}+/-{std_token_prec:.4f}\t'
        f'Recall: {mean_token_rec:.4f}+/-{std_token_rec:.4f}\t'
        f'F1: {mean_token_f1:.4f}+/-{std_token_f1:.4f}') 


if __name__ == '__main__':
  argv = sys.argv[1:]
  main(argv)    
