import torch
import torch.nn as nn

class PositionPredictor(nn.Module):
  def __init__(self, 
               input_size,
               vocab_size,
               embedding_size):
    super(PositionPredictor, self).__init__()
    self.embeddings = nn.Embedding(
                          vocab_size, 
                          embedding_size
                        )
    self.classifier = nn.Sequential(
                        nn.Linear(input_size+embedding_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 1)
                      ) 

  def forward(self, x, label):
    embed = self.embeddings(label)
    out = self.classifier(torch.cat([x, embed], dim=-1))
    return out
