import torch
import json
import os
import numpy as np
from torchvision import transforms
from PIL import Image

class SpeechCOCOImageDataset(torch.utils.data.Dataset):


  def __init__(
      self,
      preprocessor,
      data_path,
      split
    ):
    self.preprocessor = preprocessor
    self.data_path = data_path
    self.transform = transforms.Compose(
                [transforms.Scale(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                )

    if split == "train":
      self.max_keep_size = 500
    else:
      self.max_keep_size = 250

    data = load_data_split(
               data_path, split,
               min_class_size=min_class_size,
               max_keep_size=self.max_keep_size 
           )
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
    label = self.preprocessor.to_index(label)
    return region, label

  def __len__(self):
    return len(self.dataset)

def load_data_split(data_path, split,
                    max_keep_size):
  examples = []
  keep_counts = dict()
  sent_f = open(os.path.join(data_path, f"{split}2014/speechcoco_{split}.json"), "r")
  idx = 0
  for line in sent_f:
    sent = json.loads(line.rstrip("\n"))
    image_id = sent["image_id"]
    boxes = sent["boxes"]
    labels = sent["labels"]
    for box, label in zip(boxes, labels):
      if not label in keep_counts:
        keep_counts[label] = 1
      elif keep_counts[label] > max_keep_size:
        continue

      if image_id in filenames:
        filename = os.path.join(data_path, "{split}2014/imgs/{split}2014", image_id+".jpg")
        example = {"image": filename,
                   "box": box,
                   "label": label}
        examples.append(example)

  print(f"Number of bounding boxes = {len(examples)}")
  sent_f.close()
  return examples   
