import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

class FlickrImageDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, split
    ):
    self.data_path = data_path
 
    data = []
    class_freqs = json.load(open(os.path.join(data_path, "phrase_classes.json"), "r"))
    class_to_idx = {c:i for i, c in enumerate(sorted(class_freqs, key=lambda x:class_freqs[x], reverse=True)) if class_freqs[c] > 0} # XXX
    self.class_names = sorted(class_to_idx, key=lambda x:class_to_idx[x])
    self.n_class = len(class_to_idx)

    # Load data paths to audio and visual features
    data = load_data_split(data_path, split, class_to_idx)
  
    # Set up transforms
    self.transform = transforms.Compose(
                [transforms.Scale(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                )

    # Load each image metadata
    image = [example["image"] for example in data]
    boxes = [example["box"] for example in data]
    labels = [example["label"] for example in data]
    self.dataset = list(zip(image, boxes, labels))
     
  def __getitem__(self, idx):
    image_file, box, label = self.dataset[idx]
    image = Image.open(image_file).convert("RGB")
    if len(np.asarray(image).shape) == 2:
      print(f"Gray scale image {image_file}, convert to RGB".format(image_file))
      image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
    region = image.crop(box=box)
    region = self.transform(region)
    
    return region, label

  def __len__(self):
    return len(self.dataset)

def load_data_split(data_path, split, class_to_idx):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "text" : a list of tokenized words for the class name,
            "full_text" : a list of tokenized words for the whole phrase, 
            "duration" : float,
            "interval": [begin of the word in ms, end of the word in ms],
            "image_id": str,
            "feat_idx": int, image feature idx
          }
  """
  if not split:
    filenames = []
    for split in ['train', 'val', 'test']:
      with open(os.path.join(data_path, "splits/flickr40k_{}.txt".format(split)), "r") as f:
        filenames.extend(['_'.join(line.rstrip("\n").split("/")[-1].split('_')[:-1]) for line in f])
  else:
    with open(os.path.join(data_path, "splits/flickr40k_{}.txt".format(split)), "r") as f:
      filenames = ['_'.join(line.rstrip("\n").split("/")[-1].split('_')[:-1]) for line in f]
  print(f'Number of audio files: {len(filenames)}')

  examples = []
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  idx = 0
  for line in phrase_f:
    # if idx > 800: # XXX
    #   break
    idx += 1
    phrase = json.loads(line.rstrip("\n"))
    image_id = "_".join(phrase["utterance_id"].split("_")[:-1])
    box = phrase["bbox"]
    label = phrase["label"]
    if not label in class_to_idx:
      continue

    if image_id in filenames:
      filename = os.path.join(data_path, "Flicker8k_Dataset", image_id + ".jpg")
      example = {"image": filename,
                 "box": box,
                 "label": class_to_idx[label]}
      examples.append(example)

  print(f"Number of bounding boxes = {len(examples)}")
  phrase_f.close()
  return examples

class Resnet34(imagemodels.ResNet):
    def __init__(self, pretrained=True, n_class=1):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
          self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
        
          for child in self.conv1.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer1.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer2.children():
            for p in child.parameters():
              p.requires_grad = False

          for child in self.layer3.children():
            for p in child.parameters():
              p.requires_grad = True # XXX

          for child in self.layer4.children():
            for p in child.parameters():
              p.requires_grad = True # XXX
          
          for child in list(self.avgpool.children()):
            for p in child.parameters():
              p.requires_grad = True # XXX
        self.classifier = nn.Linear(512, n_class)
              
    def forward(self, x, return_score=False):
        if x.dim() == 3:
          x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        emb = x.view(x.size(0), -1)
        score = self.classifier(emb)

        if return_score:
          return score, emb
        else:
          return emb

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--task', type=int, required=True)
  parser.add_argument('--pretrain_model', default=None)
  parser.add_argument('--exp_dir', default=None)
  args = parser.parse_args()

  task = args.task
  data_path = "/home/lwang114/data/flickr/" #/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/"
  batch_size = 128
  
  if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

  if task == 0:
    trainset = FlickrImageDataset(data_path=data_path, split="train")
    testset = FlickrImageDataset(data_path=data_path, split="test")
   
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False if args.task == 1 else True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    image_model = Resnet34(pretrained=True, n_class=trainset.n_class) 
    trainables = [p for p in image_model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainables, lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    
    for epoch in range(10):
      image_model.train()
      out_file = os.path.join(args.exp_dir, f'predictions.{epoch}.readable')
      f = open(out_file, 'w')
      f.write('Image ID\tGold label\tPredicted label\n')
     
      for batch_idx, (regions, label) in enumerate(train_loader):
        score, feat = image_model(regions, return_score=True)
        loss = criterion(score, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
          print(f"Iter:{batch_idx}, loss:{loss.item()}")

      if (epoch % 2) == 0 : scheduler.step()
      
      with torch.no_grad():
        image_model.eval()
        correct = 0
        total = 0
        class_acc = torch.zeros(trainset.n_class,)
        class_count = torch.zeros(trainset.n_class,)
        for batch_idx, (regions, label) in enumerate(test_loader):
          score, feat = image_model(regions, return_score=True)
          pred = torch.max(score, dim=-1)[1]
          correct += torch.sum(pred == label).float().cpu()
          total += float(score.size(0))
          for idx in range(regions.size(0)):
            box_idx = batch_idx * batch_size + idx
            image_id = trainset.dataset[box_idx][0].split("/")[-1].split(".")[0] 
            gold_name = trainset.class_names[label[idx]]
            pred_name = trainset.class_names[pred[idx]] 
            f.write(f'{image_id} {gold_name} {pred_name}\n')
            class_acc[label[idx]] += (pred[idx] == label[idx]).float().cpu()
            class_count[label[idx]] += 1.

        acc = (correct / total).item()
        for c in range(trainset.n_class):
          if class_count[c] > 0:
            class_acc[c] = class_acc[c] / class_count[c]

        print(f"Epoch {epoch}, overall accuracy: {acc}")
        print(f"Most frequent 10 class average accuracy: {class_acc[:10].mean().item()}") 
        if acc > best_acc:
          best_acc = acc
          torch.save(image_model.state_dict(), f"{args.exp_dir}/image_model.{epoch}.pth")
      f.close()
  elif task == 1:
    trainset = FlickrImageDataset(data_path=data_path, split=None)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)

    image_model = Resnet34(pretrained=True, n_class=455) 
    if args.pretrain_model:
      image_model.load_state_dict(torch.load(args.pretrain_model))
    feats = {}
    cur_image_id = ''
    feat_id = ''
    image_idx = 0
    for batch_idx, (regions, label) in enumerate(train_loader):    
      # if batch_idx > 2: # XXX
      #   break
      feat = image_model(regions).cpu().detach()
      for i in range(feat.size(0)):
        box_idx = batch_idx * batch_size + i
        image_id = trainset.dataset[box_idx][0].split("/")[-1].split(".")[0]

        if cur_image_id != image_id:
          if len(cur_image_id):
            feats[feat_id] = torch.stack(feats[feat_id])
            print(feat_id, feats[feat_id].size())
          feat_id = f"{image_id}_{image_idx}"
          feats[feat_id] = [feat[i]] 
          cur_image_id = image_id
          image_idx += 1  
        else:
          feats[feat_id].append(feat[i])
    feats[feat_id] = torch.stack(feats[feat_id])
    print(feat_id, feats[feat_id].size())
    np.savez("flickr8k_res34.npz", **feats)
