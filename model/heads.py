import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityHead(nn.Module):
    """
        To predict the distance of the current grasp pose vs the desired grasp pose
    """
            
    def __init__(self, in_feats, out_feats = 1):
        super(QualityHead, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.quality_head = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: (batch_size, in_feats)
        return grasp_quality: (batch_size, out_feats) 
        # ! Note: low is better
        """
        return self.quality_head(x)


class ActionHead(nn.Module):
    """
    To predict the action that was applied on the object
    """

    def __init__(self, in_feats, out_classes = 2):
        super(ActionHead, self).__init__()
        self.in_feats = in_feats
        self.out_classes = out_classes

        self.action_head = nn.Sequential(
            nn.Linear(in_feats, out_classes),
        )

    def forward(self, x):
        """
        :param x: (batch_size, in_feats)
        return action: (batch_size, out_classes)
        """
        return self.action_head(x)            
