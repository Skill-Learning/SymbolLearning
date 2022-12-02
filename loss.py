"""
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
"""

import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.2):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, embeddings, pose_change, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        Args:
            embeddings: img+ data [B, embed_size]
            labels: [B,] : label_class [action_vector + object type one hot vector -> one single class]
            pose_change: [B, delta_pose]
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if embeddings.is_cuda
                  else torch.device('cpu'))

        batch_size = embeddings.shape[0]
        embed_norm = torch.norm(embeddings, dim=1, p=2).detach()
        embeddings = embeddings.div(embed_norm.unsqueeze(1))
        embeddings = embeddings.view(batch_size, 1,-1)
        embed_size = embeddings.shape[-1]

        if pose_change.shape[0]!= batch_size:
            raise ValueError('Number of poses do not match the batch size')

        if labels.shape[0]!= batch_size:
            raise ValueError('Number of labels do not match the batch size')


        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)


        contrast_count = embeddings.shape[1]
        contrast_feature = torch.cat(torch.unbind(embeddings, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = embeddings[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # TODO: multiply the exp_logits with the pose change
        # convert to rotattion matrix R
        # del_r = acos(tr(Ri, R_n)-1/2)
        #  rot_contrast_dot = rot@rot.T
        # mask: all positives are 1, negatives are 0
        # logits_mask: all are 1 except anchor
        # mask_neg = ones - mask
        # rot_contrast_dot = rot_contrast_dot * mask_neg
        # del_r = torch.acos(tr(rot_contrast_dot) - 1)/2
        # del_r = del_r/torch.pi
        # exp_logits = exp_logits * del_r

        log_prob = mask* logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # import ipdb; ipdb.set_trace()
        non_zero_inds = torch.where(mask.sum(1)!=0)
        mean_log_prob_pos = (log_prob[non_zero_inds]).sum(1) / mask[non_zero_inds].sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, -1).mean()

        return loss
