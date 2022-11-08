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

    def forward(self, initial_obs, action_vec):
        pass

