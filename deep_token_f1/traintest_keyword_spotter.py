import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import numpy as np
import pickle
import json
from pyhocon import ConfigFactory
from sklearn.metrics import precision_recall_fscore_support
import pdb
import os
import argparse
import sys
from utils.util import *
from model import BLSTM, Davenet, DotProductClassAttender
from criterion import MacroTokenFLoss, MicroTokenFLoss
from datasets.datasets import return_data

class Solver(object):

  def __init__(self, config):
    self.config = config 
    self.debug = config.debug
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    self.global_epoch = 0
    self.get_feature_config(config)
    self.get_dataset_config(config)
    self.get_model_config(config)    
    self.global_epoch = 0
    self.history = dict()
    self.history['f1'] =  0.
    self.history['loss'] = 0.
    self.history['epoch'] = 0.
    if not os.path.exists(config.exp_dir): 
      os.makedirs(config.exp_dir)

  def get_feature_config(self, config):
    self.audio_feature = config.audio_feature
    if config.audio_feature == 'mfcc':
      self.audio_feature_net = None
      self.input_size = 80
      self.hop_len_ms = 10
    elif config.audio_feature == 'wav2vec2':
      self.audio_feature_net = fairseq.checkpoint_utils.load_model_ensemble_and_task([config.wav2vec_path])[0][0]
      self.audio_feature_net = self.audio_feature_net.to(self.device)

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
    self.max_normalize = config.get('max_normalize', False)

    print(f'Number of visual label classes = {self.n_visual_class}')
    print(f'Number of phone classes = {self.n_phone_class}')
    print(f'Max normalized: {self.max_normalize}')

  def get_model_config(self, config):
    if config.model_type == 'davenet':
      self.audio_model = Davenet(input_dim=self.input_size,
                                 embedding_dim=1024)
    elif config.model_type == 'blstm':
      self.audio_model = BLSTM(512,
                               input_size=self.input_size,
                               n_layers=config.num_layers)
    self.image_model = nn.Linear(2048, 1024)
    self.attention_model = DotProductClassAttender(input_dim=1024,
                                                   hidden_dim=1024,
                                                   n_class=self.n_visual_class)

  def train(self):
    device = self.device
    torch.set_grad_enabled(True)

    args = self.config
    audio_model = self.audio_model
    image_model = self.image_model
    attention_model = self.attention_model
    train_loader = self.data_loader['train']
    n_visual_class = self.n_visual_class

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
  
    if not isinstance(attention_model, torch.nn.DataParallel):
        attention_model = nn.DataParallel(attention_model)
    
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/audio_model.pth" % (exp_dir)))
        image_model.load_state_dict(torch.load("%s/image_model.pth" % (exp_dir)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    attention_model = attention_model.to(device)
    if args.loss_type == 'macro_token_floss':
      criterion = MacroTokenFLoss()
    elif args.loss_type == 'micro_token_floss':
      criterion = MicroTokenFLoss()
    elif args.loss_type == 'binary_cross_entropy':
      criterion = nn.BCEWithLogitsLoss()
    else:
      raise ValueError(f'Invalid loss type: {args.loss_type}') 

    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    attention_trainables = [p for p in attention_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables + attention_trainables
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/optim_state.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    image_model.train()
    attention_model.train()
    while epoch < args.epoch:
        self.global_epoch += 1
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        for i, batch in enumerate(train_loader): 
            if self.debug and i > 2:
              break
            audio_input = batch[0]
            word_label = batch[2]
            input_mask = batch[3]
            word_mask = batch[5] 

            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)
            audio_input = audio_input.to(device)
            if self.audio_feature == 'wav2vec2':
              audio_input = self.audio_feature_net.feature_extractor(audio_input)

            word_label = word_label.to(device)
            input_mask = input_mask.to(device)
            word_mask = word_mask.to(device)
            nframes = input_mask.sum(-1) 
            word_mask = torch.where(word_mask.sum(dim=(-2, -1)) > 0,
                                    torch.tensor(1, device=device),
                                    torch.tensor(0, device=device))
            nwords = word_mask.sum(-1)

            # (batch size, n word class)
            word_label_onehot = (F.one_hot(word_label, n_visual_class) * word_mask.unsqueeze(-1)).sum(-2) 
            word_label_onehot = torch.where(word_label_onehot > 0,
                                            torch.tensor(1, device=device),
                                            torch.tensor(0, device=device))
            optimizer.zero_grad()
            
            audio_output = audio_model(audio_input, masks=input_mask)
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-2))
            nframes = nframes // pooling_ratio
            input_mask_ds = input_mask[:, ::pooling_ratio]
            word_logit, attn_weights = attention_model(audio_output, input_mask_ds)

            if args.loss_type == 'binary_cross_entropy': 
              loss = criterion(word_logit, word_label_onehot.float())
            else:
              word_prob = torch.sigmoid(word_logit)
              if self.max_normalize:
                word_prob = word_prob / word_prob.max(-1, keepdim=True)[0]
              loss = criterion(word_prob,
                               word_label_onehot,
                               torch.ones(B, device=device))
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            global_step += 1
            if i % 500 == 0:
                info = 'Itr {} {loss_meter.val:.4f} ({loss_meter.avg:.4f} '.format(i,loss_meter=loss_meter)
                print(info)
            i += 1
        info = ('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})').format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter)
        print(info)

        end_time = time.time()

        if np.isnan(loss_meter.avg):
            print("training diverged...")
            return
        if epoch % 1 == 0:
            precision, recall, f1 = self.validate()
            avg_acc = f1

            torch.save(audio_model.state_dict(),
                    "%s/audio_model.pth" % (exp_dir))
            torch.save(image_model.state_dict(),
                    "%s/image_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/optim_state.pth" % (exp_dir))
            
            info = f' Epoch: [{epoch}] Loss: {loss_meter.val:.4f}  Token Precision: {precision:.4f} Recall: {recall:.4f}  F1: {f1:.4f}\n'
            save_path = os.path.join(exp_dir, 'result_file.txt')
            with open(save_path, "a") as file:
                file.write(info)

            if avg_acc > best_acc:
                self.history['f1'] = f1
                self.history['loss'] = loss_meter.avg
                self.history['epoch'] = self.global_epoch
                best_epoch = epoch
                best_acc = avg_acc
                shutil.copyfile("%s/audio_model.pth" % (exp_dir), 
                    "%s/best_audio_model.pth" % (exp_dir))
                shutil.copyfile("%s/image_model.pth" % (exp_dir), 
                    "%s/best_image_model.pth" % (exp_dir))
            _save_progress()
        epoch += 1


  def validate(self):
    device = self.device
    args = self.config
    audio_model = self.audio_model
    image_model = self.image_model
    attention_model = self.attention_model
    val_loader = self.data_loader['test']
    n_visual_class = self.n_visual_class
    epoch = self.global_epoch
    batch_time = AverageMeter()
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    if not isinstance(attention_model, torch.nn.DataParallel):
        attention_model = nn.DataParallel(attention_model)
  
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    attention_model = attention_model.to(device)

    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()
    attention_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    gold_labels = [] 
    pred_labels = [] 
    readable_f = open(os.path.join(args.exp_dir, f'keyword_predictions_{epoch}.txt'), 'w')
    readable_f.write('ID\tGold\tPred\n')
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if self.debug and i > 2:
              break
            audio_input = batch[0]
            word_label = batch[2]
            input_mask = batch[3]
            word_mask = batch[5] 
            B = audio_input.size(0)
            audio_input = audio_input.to(device)
            if self.audio_feature == 'wav2vec2':
              audio_input = self.audio_feature_net.feature_extractor(audio_input)

            word_label = word_label.to(device)
            input_mask = input_mask.to(device)
            word_mask = word_mask.to(device)
            nframes = input_mask.sum(-1) 
            word_mask = torch.where(word_mask.sum(dim=(-2, -1)) > 0,
                                    torch.tensor(1, device=device),
                                    torch.tensor(0, device=device))
            nwords = word_mask.sum(-1)

            # (batch size, n word class)
            word_label_onehot = (F.one_hot(word_label, n_visual_class) * word_mask.unsqueeze(-1)).sum(-2) 
            word_label_onehot = torch.where(word_label_onehot > 0,
                                            torch.tensor(1, device=device),
                                            torch.tensor(0, device=device))
            
            # compute output
            audio_output = audio_model(audio_input, masks=input_mask)
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-2))
            input_mask_ds = input_mask[:, ::pooling_ratio]
            word_logit, attn_weights = attention_model(audio_output, input_mask_ds)
            pred_label_onehot = (word_logit > 0).long()
            gold_labels.append(word_label_onehot.flatten().detach().cpu().numpy())
            pred_labels.append(pred_label_onehot.flatten().detach().cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
          
            for ex in range(B):
              global_idx = i * val_loader.batch_size + ex
              audio_id = os.path.splitext(os.path.split(val_loader.dataset.dataset[global_idx][0])[1])[0]
              pred_idxs = torch.nonzero(pred_label_onehot[ex], as_tuple=True)[0].detach().cpu().numpy().tolist()
              gold_idxs = torch.nonzero(word_label_onehot[ex], as_tuple=True)[0].detach().cpu().numpy().tolist()

              pred_word_names = '|'.join(val_loader.dataset.preprocessor.to_word_text(pred_idxs)) 
              gold_word_names = '|'.join(val_loader.dataset.preprocessor.to_word_text(gold_idxs))
              readable_f.write(f'{audio_id}\t{gold_word_names}\t{pred_word_names}\n') 
  
        gold_labels = np.concatenate(gold_labels)
        pred_labels = np.concatenate(pred_labels)
    readable_f.close()
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gold_labels, pred_labels, average='macro')
    print(f'Macro Precision: {macro_precision: .3f}, Recall: {macro_recall:.3f}, F1: {macro_f1:.3f}')
    precision, recall, f1, _ = precision_recall_fscore_support(gold_labels, pred_labels)
    precision = precision[1]
    recall = recall[1]
    f1 = f1[1]
    print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
    return precision, recall, f1

""" TODO
def align_attention(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter() 

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    if not args.audio_model_file or not args.image_model_file:
        audio_model.load_state_dict(torch.load('{}/models/best_audio_model.pth'.format(args.exp_dir)))
        image_model.load_state_dict(torch.load('{}/models/best_image_model.pth'.format(args.exp_dir)))
    else:
        audio_model.load_state_dict(torch.load(args.audio_model_file))
        image_model.load_state_dict(torch.load(args.image_model_file))

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    max_num_units = val_loader.dataset.max_nphones
    B = args.batch_size # TODO Add to config
    split_file = None # args.datasplit
    if not split_file:
      selected_indices = list(range(N_examples))
    else:
      with open(split_file, 'r') as f:
        selected_indices = [i for i, line in enumerate(f) if int(line)]
    
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    region_counts = []
    alignments = []
    with torch.no_grad():
        for i, (audio_input, image_input, nphones, nregions) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)

            # Compute output
            image_output = image_model(image_input).transpose(1, 2).unsqueeze(-1) # Make the image output 4D
            audio_output = audio_model(audio_input)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            n = image_output.size(0)
           
            # Optionally segment the acoustic features 
            boundaries = None
            if nphones.dim() > 1:
                print('Segmentations detected, segment the acoustic features...')
                boundaries = nphones.cpu().detach().numpy()
                mask = np.zeros((n, max_num_units, audio_output.size(-1))) 
                for i_b in range(n):
                    starts = np.nonzero(boundaries[i_b, 0])[0] // pooling_ratio
                    ends = np.nonzero(boundaries[i_b, 1])[0] // pooling_ratio
                    if len(starts) == 0:
                        continue
                    for i_seg, (start, end) in enumerate(zip(starts, ends)): 
                        mask[i_b, i_seg, start:end] = 1. / max(end - start, 1)
                mask = torch.FloatTensor(mask).transpose(1, 2).to('cpu').detach()
                audio_output = torch.matmul(audio_output, mask)
                pooling_ratio = 1
                n_segments = torch.sum(nphones[:, 0], dim=-1).cpu().detach().numpy()
            else:
                n_segments = nphones.cpu().detach().numpy()

            for i_b in range(n):
              M = computeMatchmap(image_output[i_b], audio_output[i_b, :, :max(int(n_segments[i_b]), 1)])
              alignment = np.argmax(M.squeeze(1).numpy(), axis=1).tolist()
              cur_idx = selected_indices[i*B+i_b]
              align_info = {
                'index': cur_idx,
                'alignment': alignment,
                'image_concepts': [0]*len(alignment),
                'align_probs': M.squeeze(1).numpy().tolist()
                }
              alignments.append(align_info)
            print('Process {} batches after {}s'.format(i, time.time()-end))
    with open('{}/{}'.format(args.exp_dir, args.alignment_scores), 'w') as f:
      json.dump(alignments, f, indent=4, sort_keys=True)

def evaluation_attention(audio_model, image_model, attention_model, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    progress_pkl = "%s/progress.pkl" % exp_dir
    progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)

    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    
    recalls = validate_attention(audio_model, image_model, attention_model, test_loader, args)
    
    avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
    
    info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  Audio: R1: {R1_:.4f} R5: {R5_:.4f}  R10: {R10_:.4f}  Image: R1: {IR1_:.4f} R5: {IR5_:.4f}  R10: {IR10_:.4f}  \n \
            '.format(epoch,loss_meter=loss_meter,R1_=recalls['A_r1'],R5_=recalls['A_r5'],R10_=recalls['A_r10'],IR1_=recalls['I_r1'],IR5_=recalls['I_r5'],IR10_=recalls['I_r10'])
    print(info)


def evaluation_vector(audio_model, image_model, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    progress_pkl = "%s/progress.pkl" % exp_dir
    progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)

    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    
    recalls = validate_vector(audio_model, image_model, test_loader, args)
    
    avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
    
    info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  Audio: R1: {R1_:.4f} R5: {R5_:.4f}  R10: {R10_:.4f}  Image: R1: {IR1_:.4f} R5: {IR5_:.4f}  R10: {IR10_:.4f}  \n \
            '.format(epoch,loss_meter=loss_meter,R1_=recalls['A_r1'],R5_=recalls['A_r5'],R10_=recalls['A_r10'],IR1_=recalls['I_r1'],IR5_=recalls['I_r5'],IR10_=recalls['I_r10'])
    print(info)
"""

def main(argv):
    parser = argparse.ArgumentParser(description='Deep Macro Token F1 for Keyword Spotting')
    parser.add_argument('CONFIG', type=str)
    args = parser.parse_args(argv)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    config = ConfigFactory.parse_file(args.CONFIG)
    
    avg_f1 = []
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
      if config.mode == 'train':
        net.train()
      elif config.mode == 'test':
        net.validate() 
      else:
        return 0

    avg_f1.append(net.history['f1'])
    print(f'Average Macro F1: {np.mean(avg_f1)}+/-{np.std(avg_f1)}')

if __name__ == '__main__':
  argv = sys.argv[1:]
  main(argv) 


