import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseAttentionBlock(nn.Module):
    def __init__(self, num_points, in_feats = 1024, gamma_feats = 512, out_feats = 256):
        super(PairwiseAttentionBlock, self).__init__()
        
        self.in_feats = in_feats
        self. gamma_feats = gamma_feats
        self.num_points = num_points
        self.out_feats = out_feats

        self.gamma_fn = nn.Sequential(
            nn.Linear(1, 256), 
            nn.ReLU(),
            nn.Linear(256, gamma_feats),
        )

        # TODO: Add positional encoding later

        self.beta_fn = nn.Sequential(
            nn.Linear(in_feats, in_feats), 
            nn.ReLU(),
        )

        self._norm_and_activate_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_points),
            nn.ReLU()
        )


        self.aggregation_mlp = nn.Sequential(
            nn.Linear(self.num_points, self.out_feats)
        )

        self._norm_and_activate_2 = nn.Sequential(
            nn.BatchNorm1d(self.out_feats),
            nn.ReLU()
        )

        
    def forward(self, point_features, grasping_point_features):
        """
        :param point_features: (B, num_points, in_feats)
        :param grasping_point_features: (B, in_feats)
        :return: (B, num_points ,out_feats)
        """        

        grasping_point_features = grasping_point_features.unsqueeze(1)

        delta = torch.bmm(grasping_point_features, point_features.transpose(1, 2)) # (B, 1, num_points)
        
        delta = delta.transpose(1, 2) # (B, num_points, 1)

        gamma = self.gamma_fn(delta) # (B, num_points, gamma_feats)

        beta = self.beta_fn(point_features) # (B, num_points, in_feats)

        aggregated_features = torch.bmm(gamma, beta.transpose(1, 2)) # (B, num_points, num_points)

        aggregated_features = self._norm_and_activate_1(aggregated_features.transpose(2,1)) # (B, num_points, num_points)
        aggregated_features = self.aggregation_mlp(aggregated_features.transpose(2,1)) # (B, num_points, out_feats)


        aggregated_features = self._norm_and_activate_2(aggregated_features.transpose(2,1)) # (B, num_points, out_feats)

        return aggregated_features.transpose(1, 2) # (B, out_feats, num_points)
