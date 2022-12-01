'''This is a barebones implementation of the encoder, which
will hopefully be a PN++ type encoder.'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet18
from torchvision.models import resnet18, ResNet18_Weights
from utils import *

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        self.image_encoder = None

        if self.cfg['image_encoder']['type'] == 'ResNet18':
            self.image_encoder_type = 'ResNet18'
            self.image_encoder = ResNet18(return_feats=True)
            img_embedding_dim = 512
            if cfg['image_encoder']['use_pretrained']:
                self.pre_trained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                init_weights_and_freeze(self.image_encoder, self.pre_trained_model)
            else:
                # TODO: add function to initialize weights
                raise NotImplementedError


        if self.image_encoder is None:
            raise ValueError('Invalid image encoder type')

        self.metadata_size = self.cfg['coarse_training']['metadata_size']
        # size = action_one_hot_size + initial_pose_size 
        self.metadata_encoder = nn.Sequential(nn.Linear(self.metadata_size, 256),
                                              nn.ReLU(),
                                                nn.Linear(256, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, img_embedding_dim))

    def forward(self, initial_image, action_vector, initial_pose):
        # initial_image: [B, 3, 255, 255, 255]
        # action_vector: [B, 5]
        # initial_pose: [B, 7]
        # output: [B, 512*2]
        img_embedding = self.image_encoder(initial_image)
        metadata = torch.cat((action_vector, initial_pose), dim=1)
        metadata_embedding = self.metadata_encoder(metadata)

        output = torch.cat((img_embedding, metadata_embedding), dim=1)
        return output

