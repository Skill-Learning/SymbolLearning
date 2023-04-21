import torch 
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet18
from torchvision.models import resnet18, ResNet18_Weights
from utils import *

class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        self.cfg = cfg

        self.embedding_dim = 1024
        self.mlp = nn.Sequential(nn.Linear(self.embedding_dim, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    nn.Linear(512, 256),
                                    nn.ReLU()
                                )
        self.position_mlp = nn.Sequential(nn.Linear(256, 3))
        self.quaternion_mlp = nn.Sequential(nn.Linear(256, 4), 
                                            # nn.Softmax(dim=1)
                                            )
    def forward(self,embedding):
        # embedding: [B, 1024]
        # output: [B, 7]
        output = self.mlp(embedding)
        position = self.position_mlp(output)
        quaternion = self.quaternion_mlp(output)
        output = torch.cat((position, quaternion), dim=1)
        return output

