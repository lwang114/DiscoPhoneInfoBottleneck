import torch
import torchvision
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
import numpy as np
import re
import os
import json
from tqdm import tqdm
from PIL import Image

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
    self.image_model = Resnet34(pretrained=True) 
    self.transforms = transforms.Compose(
                [transforms.Scale(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                )

    # Load each image metadata
    image = [example["image"] for example in data]
    boxes = [example["boxes"] for example in data]
    self.dataset = list(zip(image, boxes))
     

  def __getitem__(self, idx):
    image_file, boxes = self.dataset[idx]
    image = Image.open(image_file + self.image_keys[idx]).convert("RGB")
    if len(np.asarray(image).shape) == 2:
      print(f"Gray scale image {image_file}, convert to RGB".format(image_file)
      image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
    outputs = []
    for box in boxes:
      region = image.crop(box=box)

      if self.transform:
        region = self.transform(region)
      image_inputs = torch.FloatTensor(region)
      emb = self.image_model(image_inputs)
      outputs.append(emb) 
    outputs = torch.cat(outputs)

    return outputs

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
  cur_image_id = ''
  phrase_f = open(os.path.join(data_path, "flickr8k_phrases.json"), "r")
  for line in phrase_f:
    phrase = json.loads(line.rstrip("\n"))
    image_id = "_".join(phrase["utterance_id"].split("_")[:-1])
    box = phrase["bbox"]
    filename = os.path.join(data_path, "Flicker8k_Dataset", image_id + ".jpg")
    
    if image_id != cur_image_id:
      example = {"image": filename,
                 "boxes": [box]}
      examples.append(example)
    else:
      examples[-1]["boxes"].append(box)
  print(f"Number of images = {len(examples)}")
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
        if x.ndim == 3:
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
        print(emb.size()) # XXX
        return emb

if __name__ == "__main__":
  data_path = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/"
  trainset = FlickrImageDataset(data_path=data_path, split="train")
  testset = FlickrImageDataset(data_path=data_path, split="test")

  feats = {}
  for idx, feat in enumerate(trainset):
    image_id = trainset.dataset[idx][0].split("/")[-1].split(".")[0]
    feat_id = f"{image_id}_{idx}"
    print(feat_id, feat.size())
    feats[feat_id] = feat.numpy()

  np.savez(os.path.join(data_path, "flickr8k_res34.npz"), **feats)
