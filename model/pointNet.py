import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedWtMLP(nn.Module):
    def __init__(self, in_feats:int, out_feats:int,
                normalize:bool=True, activation:nn.Module=nn.ReLU()):
        super(SharedWtMLP, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.normalize = normalize
        self.activation = activation

        self.fc_module = nn.Sequential(
            nn.Conv1d(in_feats, out_feats, 1),
            nn.BatchNorm1d(out_feats) if normalize else nn.Identity(),
            activation
        )

    def forward(self, x):
        return self.fc_module(x)


class LinearMLP(nn.Module):
    def __init__(self, in_feats:int, out_feats:int,
                normalize:bool=True, activation:nn.Module=nn.ReLU(), dropout:float=0.3):
        super(LinearMLP, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.normalize = normalize
        self.activation = activation
        self.dropout = dropout

        self.fc_module = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(out_feats) if normalize else nn.Identity(),
            activation, 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc_module(x)
    

class PointNet(nn.Module):
    def __init__(self, num_classes=3, task="cls", num_seg_classes=6, dropout=0.3, return_point_feats=True):
        super(PointNet, self).__init__()

        self.task = task
        self.num_classes = num_classes
        self.return_point_feats = return_point_feats
        # per point feature extraction 
        self.conv1 = SharedWtMLP(3, 64)
        self.conv2 = SharedWtMLP(64, 64)
        self.conv3 = SharedWtMLP(64, 64)
        self.conv4 = SharedWtMLP(64, 128)
        self.conv5 = SharedWtMLP(128, 1024)

        if not self.return_point_feats:

            if self.task == "cls":
                self.classification_head = nn.Sequential(
                    LinearMLP(1024, 512, dropout=dropout),
                    LinearMLP(512, 256, dropout=dropout),
                    LinearMLP(256, num_classes, activation=nn.Identity(), normalize=False, dropout=0.0)
                )
                self.segmentation_head = None
            elif self.task == "seg":
                self.segmentation_head = nn.Sequential(
                SharedWtMLP(1088, 512),
                SharedWtMLP(512, 256),
                SharedWtMLP(256, num_seg_classes, activation=nn.Identity(), normalize=False)
                )

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, points):

        """
        points = (B, N, 3)
        output = (B, num_classes) if task == "cls"
        output = (B, N, num_seg_classes) if task == "seg"
        output = (B, N, 1024) if return_point_feats == True

        """
        points = points.permute(0, 2, 1)
        # points shape = (B, 3, N)
        # per point feature extraction 
        fc1 = self.conv1(points) # (B, 64, N)
        fc2 = self.conv2(fc1) # (B, 64, N)
        fc3 = self.conv3(fc2) # (B, 64, N)
        fc4 = self.conv4(fc3) # (B, 128, N)
        fc5 = self.conv5(fc4) # (B, 1024, N)

        if self.return_point_feats:
            return fc5.permute(0, 2, 1) # (B, N, 1024)

        # maxpool
        maxpool = F.max_pool1d(fc5, fc5.shape[2])
        # maxpool shape = (B, 1024, 1)
        global_feats = maxpool.view(-1, 1024) # (B, 1024)


        if self.task == "cls":
            # classification head
            output = self.classification_head(global_feats)
        elif self.task == "seg":
            # segmentation head
            maxpool = maxpool.repeat(1, 1, fc3.shape[2]) # (B, 1024, N)
            concat_feats = torch.cat([fc3, maxpool], dim=1) # (B, 1088, N)
            output = self.segmentation_head(concat_feats)
            output = output.permute(0, 2, 1) # (B, N, num_seg_classes)
        
        return output
