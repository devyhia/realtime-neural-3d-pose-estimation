import torch
from torch import nn

class TripletsLoss(nn.Module):
    def __init__(self, margin=0.01):
        super(TripletsLoss,self).__init__()

        self.margin = margin
        
    def forward(self,anchor_features, puller_features, pusher_features):
        diff_pos = (anchor_features - puller_features)
        diff_neg = (anchor_features - pusher_features)

        diff_pos = torch.mul(diff_pos, diff_pos)
        diff_neg = torch.mul(diff_neg, diff_neg)

        diff_pos = diff_pos.sum(1)
        diff_neg = diff_neg.sum(1)

        loss_pairs = diff_pos
        loss_triplets_ratio = 1 - diff_neg / (diff_pos + self.margin)
        
        loss_triplets = torch.max(
            torch.zeros_like(loss_triplets_ratio),
            loss_triplets_ratio
        )

        total_loss = torch.mean(loss_triplets + loss_pairs)

        return total_loss