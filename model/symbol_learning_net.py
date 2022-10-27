import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from autolab_core import YamlConfig

from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention

class SymbolLearningNet(nn.Module):
    def __init__(self, cfg):
        super(SymbolLearningNet, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.attention = Attention(cfg)

    def forward(self, initial_obs, action_vec):
        # initial_obs: [batch_size, 3, 255, 255, 255, 255]
        # action_vec: [batch_size, 4]
        # output: [batch_size, 3, 255, 255, 255, 255]
        batch_size = initial_obs.shape[0]
        initial_obs = initial_obs.view(batch_size, -1)
        action_vec = action_vec.view(batch_size, -1)
        # initial_obs: [batch_size, 3 * 255 * 255 * 255 * 255]
        # action_vec: [batch_size, 4]
        z = self.encoder(initial_obs)
        a = self.attention(z, action_vec)
        output = self.decoder(a)
        output = output.view(batch_size, 3, 255, 255, 255, 255)
        return output
    
    
