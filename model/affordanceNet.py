import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointNet import PointNet
from .heads import QualityHead, ActionHead
from .attention import PairwiseAttentionBlock

from utils import *

class AffordanceNet(nn.Module):
    def __init__(self, cfg):
        super(AffordanceNet, self).__init__()
        self.cfg = cfg
        self.pointnet = PointNet(dropout=cfg['anet_training']['dropout'], 
                                return_point_feats= True)
        self.attention = PairwiseAttentionBlock()
        # TODO: Add the value of the in_feats for the heads
        in_feats = None
        self.quality_head = QualityHead(in_feats)
        self.action_head = ActionHead(in_feats)

    def forward(self, point_cloud, grasping_point):
        """
        point_cloud: (B, N, 3)
        grasping_point: (B, 3)
        """

        # point cloud feature extraction 
        grasping_point_input = grasping_point.unsqueeze(1)
        point_cloud_accum = torch.cat((point_cloud, grasping_point_input), dim=1)
        point_accum_feats = self.pointnet(point_cloud_accum)

        point_feats = point_accum_feats[:, :-1, :]
        grasping_point_feats = point_accum_feats[:, -1, :]
        
         


