import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
import numpy as np
import re
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

class FlickrImageDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, split,
      splits = {
        "train": ["train"],
        "test": ["test"],           
      }
  ):
    self.splits = splits
    self.data_path = data_path
 
    data = []
    for sp in self.splits[split]:
      # Load data paths to audio and visual features
      examples = load_data_split(data_path, split)
      data.extend(examples)    
  
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
    self.dataset = list(zip(image, boxes))
     

  def __getitem__(self, idx):
    image_file, box = self.dataset[idx]
    image = Image.open(image_file).convert("RGB")
    if len(np.asarray(image).shape) == 2:
      print(f"Gray scale image {image_file}, convert to RGB".format(image_file))
      image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
    region = image.crop(box=box)
    region = self.transform(region)
    return region

  def __len__(self):
    return len(self.dataset)

def load_data_split(data_path, split):
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
  examples = []
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  for line in phrase_f:
    phrase = json.loads(line.rstrip("\n"))
    image_id = "_".join(phrase["utterance_id"].split("_")[:-1])
    box = phrase["bbox"]
    
    filename = os.path.join(data_path, "Flicker8k_Dataset", image_id + ".jpg")
    example = {"image": filename,
               "box": box}
    examples.append(example)

  print(f"Number of bounding boxes = {len(examples)}")
  phrase_f.close()
  return examples

class Resnet34(imagemodels.ResNet):
    def __init__(self, pretrained=True):
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
              p.requires_grad = False

          for child in self.layer4.children():
            for p in child.parameters():
              p.requires_grad = False
          
          for child in list(self.avgpool.children()):
            for p in child.parameters():
              p.requires_grad = False

    def forward(self, x):
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
        return emb

if __name__ == "__main__":
  data_path = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/"
  trainset = FlickrImageDataset(data_path=data_path, split="train")
  batch_size = 128
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)   
  image_model = Resnet34(pretrained=True) 
  
  feats = {}
  cur_image_id = ''
  feat_id = ''
  image_idx = 0
  for batch_idx, regions in enumerate(train_loader):    
    # if batch_idx > 2: XXX
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
