import numpy as np
import torch
import argparse
import math
import json
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from image_model import Resnet34 # TODO 


class Solver(object):
  

  def __init__(self, args):
    self.args = args

    self.cuda = torch.cuda.is_available()
    self.epoch = args.epoch
    self.batch_size = args.batch_size

    class_freqs = json.load(open(os.path.join(args.data_path, "phrase_classes.json"), "r"))
    class_to_idx = {c:i for i, c in enumerate(sorted(class_freqs, key=lambda x:class_freqs[x], reverse=True)) if class_freqs[c] > 0}
    self.class_names = sorted(class_to_idx, key=lambda x:class_to_idx[x])
    self.n_class = len(class_to_idx)
    self.image_model = Resnet34(pretrained=True, n_class=self.n_class)  
    trainables = [p for p in self.image_model.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainables, lr=0.0001)
    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
    self.criterion = nn.CrossEntropyLoss()

    self.history = dict()
    self.history['acc'] = 0.
    self.history['epoch'] = 0
    self.history['best_epoch'] = 0

    self.exp_dir = args.exp_dir
    if not os.path.exists(self.exp_dir):
      os.makedirs(self.exp_dir)

    if args.mode == 'test':
      self.image_model.load_state_dict(torch.load( 
                           os.path.join(
                             self.exp_dir, 
                             'image_model.pth')))

  def set_mode(self, mode='train'):
    if mode == 'train':
      self.image_model.train()
    elif mode == 'eval':
      self.image_model.eval()
    else: raise('mode error. It should be either train or eval')

  def train(self, train_loader, test_loader):
    self.set_mode('train')
    for epoch in range(self.epoch):
      self.image_model.train()     
      for batch_idx, (regions, label) in enumerate(train_loader):
        # if batch_idx > 2: # XXX
        #   break
        score, feat = self.image_model(regions, return_score=True)
        loss = self.criterion(score, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if batch_idx % 100 == 0:
          print(f"Iter: {batch_idx}, loss: {loss.item()}")

      if (epoch % 2) == 0: 
        self.scheduler.step()
      self.history['epoch'] = epoch
      acc = self.test(test_loader)
      if acc > self.history['acc']:
        self.history['acc'] = acc
        self.history['best_epoch'] = epoch
        torch.save(self.image_model.state_dict(), f"{self.exp_dir}/image_model.{epoch}.pth")
  
  def test(self, test_loader, out_prefix='predictions'):
    with torch.no_grad():
      self.image_model.eval()
      correct = 0
      total = 0
      class_acc = torch.zeros(test_loader.dataset.n_class)
      class_count = torch.zeros(test_loader.dataset.n_class)

      out_file = os.path.join(self.exp_dir, f'{out_prefix}.{self.history["epoch"]}.readable')
      f = open(out_file, 'w')
      f.write('Image ID\tGold label\tPredicted label\n')
      for batch_idx, (regions, label) in enumerate(test_loader):
        # if batch_idx > 2: # XXX
        #   break
        score, feat = self.image_model(regions, return_score=True)
        pred = torch.max(score, dim=-1)[1]
        correct += torch.sum(pred == label).float().cpu()
        total += float(score.size(0))
        for idx in range(regions.size(0)):
          box_idx = batch_idx * self.batch_size + idx
          image_id = test_loader.dataset.dataset[box_idx][0].split("/")[-1].split(".")[0]
          gold_name = test_loader.dataset.class_names[label[idx]]
          pred_name = test_loader.dataset.class_names[pred[idx]] 
          f.write(f'{image_id} {gold_name} {pred_name}\n') 
    acc = (correct / total).item()
    for c in range(test_loader.dataset.n_class):
      if class_count[c] > 0:
        class_acc[c] = class_acc[c] / class_count[c]
    print(f'Epoch {self.history["epoch"]}: overall accuracy: {acc}')
    print(f'Most frequent 10 class average accuracy: {class_acc[:10].mean().item()}')
    return acc
