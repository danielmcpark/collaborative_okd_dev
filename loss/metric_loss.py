import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from metric import pdist, DistanceWeighted

__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss']

class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                    margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)

class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()

if __name__=='__main__':
    loss = L2Triplet(sampler=DistanceWeighted)
    embeds = torch.randn([256, 128])
    labels = torch.randn([256, 128])
    a, b, c = loss(embeds, labels)
    print(a, b, c)
