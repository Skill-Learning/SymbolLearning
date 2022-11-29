import torch 
import torch.nn as nn
import torch.functional as F

class TripletHardLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletHardLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        
