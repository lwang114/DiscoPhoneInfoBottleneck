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
from image_model import Resnet34 

class Solver(object):
  

  def __init__(self, args):
    self.args = args

    self.cuda = (args.cuda and torch.cuda.is_available())
    self.epoch = args.epoch
    self.batch_size = args.batch_size

    self.image_model = Resnet34(pretrained=True, n_class=trainset.n_class)  
    trainables = [p for p in image_model.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainables, lr=0.0001)
    self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    self.criterion = nn.CrossEntropyLoss()

    self.history = dict()
    self.history['acc'] = 0.
    self.history['epoch'] = 0
    self.history['best_epoch'] = 0

    self.exp_dir = args.exp_dir
    self.evaluate_only = False 
    if args.mode == 'test':
      self.load_checkpoint(self.image_model, 
                           os.path.join(
                             self.exp_dir, 
                             'image_model.pth'))

  def set_mode(self, mode='train'):
    if mode == 'train':
      self.image_model.train()
    elif mode == 'eval':
      self.image_model.eval()
    else: raise('mode error. It should be either train or eval')

  def train(self, train_loader, test_loader):
    self.set_mode('train')
    for epoch in range(10):
      self.image_model.train()     
      for batch_idx, (regions, label) in enumerate(train_loader):
        if batch_idx > 2: # XXX
          break
        score, feat = self.image_model(regions, return_score=True)
        loss = self.criterion(score, label)

        self.optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
          print(f"Iter: {batch_idx}, loss: {loss.item()}")

      if (epoch % 2) == 0: 
        scheduler.step()
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

      out_file = os.path.join(args.exp_dir, f'{out_prefix}.{self.history["epoch"]}.readable')
      f = open(out_file, 'w')
      f.write('Image ID\tGold label\tPredicted label\n')
      for batch_idx, (regions, label) in enumerate(test_loader):
        if batch_idx > 2: # XXX
          break
        score, feat = image_model(regions, return_score=True)
        pred = torch.max(score, dim=-1)[1]
        correct += torch.sum(pred == label).float().cpu()
        total += float(score.size(0))
        for idx in range(regions.size(0)):
          box_idx = batch_idx * batch_size + idx
          image_id = testset.dataset[box_idx][0].split("/")[-1].split(".")[0]
          gold_name = testset.class_names[label[idx]]
          pred_name = testset.class_names[pred[idx]] 
          f.write(f'{image_id} {gold_name} {pred_name}\n') 
    acc = (correct / total).item()
    for c in range(test_loader.dataset.n_class):
      if class_count[c] > 0:
        class_acc[c] = class_acc[c] / class_count[c]
    print(f"Epoch {epoch}, overall accuracy: {acc}")
    print(f'Most frequent 10 class average accuracy: {class_acc[:10].mean().item()}')
    return acc
