import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo

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
              p.requires_grad = False

          for child in self.layer4.children():
            for p in child.parameters():
              p.requires_grad = True 
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

class FFNN(nn.Module):
  def __init__(self, n_class):
    super(FFNN, self).__init__()
    self.feature_extractor = nn.Sequential(
                                 nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.3)
                             )
    self.classifier = nn.Linear(1024, n_class)
  
  def forward(self, x, return_score=False):
    emb = self.feature_extractor(x)
    score = self.classifier(emb)
    if return_score:
      return score, emb
    else:
      return emb
