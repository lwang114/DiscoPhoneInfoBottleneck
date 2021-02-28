import numpy as np
import torch
import argparse
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils import cuda, Weight_EMA_Update
from datasets.datasets import return_data
from model import GumbelBLSTM
from pathlib import Path
import json
from evaluate import evaluate

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
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0
        self.ds_ratio = args.ds_ratio
        
        # Network & Optimizer
        self.toynet = cuda(GumbelBLSTM(self.K, ds_ratio=self.ds_ratio), self.cuda)
        self.toynet.weight_init()
        self.toynet_ema = Weight_EMA_Update(cuda(GumbelBLSTM(self.K, ds_ratio=self.ds_ratio), self.cuda),\
                self.toynet.state_dict(), decay=0.999)

        self.optim = optim.Adam(self.toynet.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        # History
        self.history = dict()
        self.history['avg_acc']=0.
        self.history['info_loss']=0.
        self.history['class_loss']=0.
        self.history['total_loss']=0.
        self.history['epoch']=0
        self.history['iter']=0

        # Tensorboard
        self.tensorboard = args.tensorboard
        if self.tensorboard :
            self.env_name = args.env_name
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists() : self.summary_dir.mkdir(parents=True,exist_ok=True)
            self.tf = SummaryWriter(log_dir=self.summary_dir)
            self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

        # Dataset
        self.data_loader = return_data(args)

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.toynet.train()
            self.toynet_ema.model.train()
        elif mode == 'eval' :
            self.toynet.eval()
            self.toynet_ema.model.eval()
        else : raise('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        temp_min = 0.1
        anneal_rate = 3e-6
        temp = 1.
        
        for e in range(self.epoch) :
            self.global_epoch += 1

            for idx, (audios,labels,masks) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = Variable(cuda(audios, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                masks = Variable(cuda(masks, self.cuda))
                in_logit, logit = self.toynet(x, masks=masks, temp=temp)

                class_loss = F.cross_entropy(logit,y).div(math.log(2))
                info_loss = (F.softmax(in_logit,dim=-1) * F.log_softmax(in_logit,dim=-1)).sum(1).mean().div(math.log(2))
                total_loss = class_loss + self.beta*info_loss
                
                izy_bound = math.log(65,2) - class_loss
                izx_bound = info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.toynet_ema.update(self.toynet.state_dict())

                prediction = F.softmax(logit,dim=1).max(1)[1]
                accuracy = torch.eq(prediction,y).float().mean()
                
                if self.num_avg != 0 :
                    _, avg_soft_logit = self.toynet(x,self.num_avg,masks=masks)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction,y).float().mean()
                else : avg_accuracy = Variable(cuda(torch.zeros(accuracy.size()), self.cuda))

                if self.global_iter % 100 == 0 :
                    temp = np.maximum(temp * np.exp(-anneal_rate * idx), temp_min)
                    print('i:{} Total Loss:{:.2f} IZY:{:.2f} IZX:{:.2f}'
                            .format(idx+1, total_loss.item(), izy_bound.item(), izx_bound.item()), end=' ')
                    print('acc:{:.4f} avg_acc:{:.4f}'
                            .format(accuracy.item(), avg_accuracy.item()), end=' ')
                    print('err:{:.4f} avg_err:{:.4f}'
                            .format(1-accuracy.item(), 1-avg_accuracy.item()))

                if self.global_iter % 10 == 0 :
                    if self.tensorboard :
                        self.tf.add_scalars(main_tag='performance/accuracy',
                                            tag_scalar_dict={
                                                'train_one-shot':accuracy.data[0],
                                                'train_multi-shot':avg_accuracy.data[0]},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={
                                                'train_one-shot':1-accuracy.data[0],
                                                'train_multi-shot':1-avg_accuracy.data[0]},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={
                                                'train_one-shot_class':class_loss.data[0],
                                                'train_one-shot_info':info_loss.data[0],
                                                'train_one-shot_total':total_loss.data[0]},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='mutual_information/train',
                                            tag_scalar_dict={
                                                'I(Z;Y)':izy_bound.data[0],
                                                'I(Z;X)':izx_bound.data[0]},
                                            global_step=self.global_iter)

            if (self.global_epoch % 2) == 0 : self.scheduler.step()
            self.test()
            
        print(" [*] Training Finished!")

    def test(self, save_ckpt=True):
        self.set_mode('eval')

        class_loss = 0
        info_loss = 0
        total_loss = 0
        izy_bound = 0
        izx_bound = 0
        correct = 0
        avg_correct = 0
        total_num = 0
        pred_dicts = []
        with torch.no_grad():
            B = 0
            for b_idx, (audios,labels,masks) in enumerate(self.data_loader['test']):
              if b_idx == 0:
                  B = audios.size(0)
              x = Variable(cuda(audios, self.cuda))
              y = Variable(cuda(labels, self.cuda))
              masks = Variable(cuda(masks, self.cuda))
              in_logit, logit, encoding = self.toynet_ema.model(x, masks=masks, return_encoding=True)

              cur_class_loss = F.cross_entropy(logit,y,size_average=False).div(math.log(2))
              cur_info_loss = (F.softmax(in_logit,dim=-1) * F.log_softmax(in_logit,dim=-1)).sum(1).mean().div(math.log(2))
              class_loss = class_loss + cur_class_loss
              info_loss = info_loss + cur_info_loss
              total_loss = total_loss + class_loss + self.beta*info_loss
              total_num += y.size(0)

              izy_bound = izy_bound + y.size(0) * math.log(65,2) - cur_class_loss 
              izx_bound = izx_bound + cur_info_loss

              prediction = F.softmax(logit,dim=1).max(1)[1]
              correct += torch.eq(prediction,y).float().sum()
              for idx in range(audios.size(0)):
                  global_idx = b_idx * B + idx
                  example_id = self.data_loader['test'].dataset.dataset[global_idx][0]
                  text = self.data_loader['test'].dataset.dataset[global_idx][1]
                  ds_ratio = self.ds_ratio
                  L = ds_ratio * (encoding.size(1) // ds_ratio)
                  encoding_ds = encoding[idx, :L].view(ds_ratio, int(L // ds_ratio), -1).sum(1)
                  units = encoding_ds.max(-1)[1]
                  pred_dicts.append({'sent_id': example_id,
                                     'units': units.cpu().detach().numpy().tolist(),  
                                     'text': text})
              
              if self.num_avg != 0 :
                  _, avg_soft_logit = self.toynet_ema.model(x,self.num_avg)
                  avg_prediction = avg_soft_logit.max(1)[1]
                  avg_correct += torch.eq(avg_prediction,y).float().sum()
              else :
                  avg_correct = Variable(cuda(torch.zeros(correct.size()), self.cuda))

        accuracy = correct/total_num
        avg_accuracy = avg_correct/total_num

        izy_bound /= total_num
        izx_bound /= total_num
        class_loss /= total_num
        info_loss /= total_num
        total_loss /= total_num

        token_f1, _, token_prec, token_recall = evaluate(pred_dicts, self.data_loader['test'].dataset.gold_dicts, ds_rate=8)
        print('[TEST RESULT]')
        print('e:{} IZY:{:.2f} IZX:{:.4f}'
                .format(self.global_epoch, izy_bound.item(), izx_bound.item()), end=' ')
        print('token precision:{:.4f} token recall:{:.4f} token f1:{:.4f} acc:{:.4f} avg_acc:{:.4f}'
                .format(token_prec, token_recall, token_f1, accuracy.item(), avg_accuracy.item()), end=' ')
        print('err:{:.4f} avg_erra:{:.4f}'
                .format(1-accuracy.item(), 1-avg_accuracy.item()))
        print()

        if self.history['avg_acc'] < avg_accuracy.item() :
            self.history['avg_acc'] = avg_accuracy.item()
            self.history['class_loss'] = class_loss.item()
            self.history['info_loss'] = info_loss.item()
            self.history['total_loss'] = total_loss.item()
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            if save_ckpt : self.save_checkpoint('best_acc.tar')
            json.dump(pred_dicts, open(self.ckpt_dir.joinpath('best_predicted_units.json'), 'w'), indent=2)
            
        if self.tensorboard :
            self.tf.add_scalars(main_tag='performance/token_prec',
                                tag_scalar_dict={
                                    'test_one-shot':token_prec})
            self.tf.add_scalars(main_tag='performance/token_recall',
                                tag_scalar_dict={
                                    'test_one-shot':token_recall})
            self.tf.add_scalars(main_tag='performance/token_f1',
                                tag_scalar_dict={
                                    'test_one-shot':token_f1})
            self.tf.add_scalars(main_tag='performance/accuracy',
                                tag_scalar_dict={
                                    'test_one-shot':accuracy.item(),
                                    'test_multi-shot':avg_accuracy.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/error',
                                tag_scalar_dict={
                                    'test_one-shot':1-accuracy.item(),
                                    'test_multi-shot':1-avg_accuracy.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='performance/cost',
                                tag_scalar_dict={
                                    'test_one-shot_class':class_loss.item(),
                                    'test_one-shot_info':info_loss.item(),
                                    'test_one-shot_total':total_loss.item()},
                                global_step=self.global_iter)
            self.tf.add_scalars(main_tag='mutual_information/test',
                                tag_scalar_dict={
                                    'I(Z;Y)':izy_bound.item(),
                                    'I(Z;X)':izx_bound.item()},
                                global_step=self.global_iter)
        self.set_mode('train')

    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.toynet.state_dict(),
                'net_ema':self.toynet_ema.model.state_dict(),
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

            self.toynet.load_state_dict(checkpoint['model_states']['net'])
            self.toynet_ema.model.load_state_dict(checkpoint['model_states']['net_ema'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
