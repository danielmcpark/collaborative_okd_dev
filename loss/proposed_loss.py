from __future__ import absolute_import
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from scipy import stats

def PearsonCorrelation(g_w, ce, recip=False):
    assert len(g_w.size())==2 and len(ce.size())==1, 'Necessitate checking out dimensions'

    g_w = g_w.mean(0).squeeze(0)
    if recip:
        g_w = torch.reciprocal(g_w)
    # True recip, desire to p -> +1
    # False recip, desire to p -> -1

    # We can get pearson coefficient per a batch
    g_w = (g_w - torch.mean(g_w)) + 1e-6
    ce = (ce - torch.mean(ce)) + 1e-6

    pc = torch.sum(g_w*ce) / (torch.sqrt(torch.sum(g_w**2)) * torch.sqrt(torch.sum(ce**2)))
    '''
    assert len(g_w.size())==2 and len(ce.size())==2, 'Necessitate checking out dimensions'
    print(g_w.size(), ce.size())
    #pc = list()
    pc = 0
    for i in range(g_w.size(0)):
        g_w[i,:] = (g_w[i,:] - torch.mean(g_w[i,:], dim=-1))+1e-6
        ce[i,:] = (ce[i,:] - torch.mean(ce[i,:], dim=-1))+1e-6

        #pc.append((torch.sum(g_w[i,:]*ce[i,:]) / (torch.sqrt(torch.sum(g_w[i,:]**2)) * torch.sqrt(torch.sum(ce[i,:]**2)))).item())
        pc += torch.sum(g_w[i,:]*ce[i,:]) / (torch.sqrt(torch.sum(g_w[i,:]**2)) * torch.sqrt(torch.sum(ce[i,:]**2)))
    print(pc)
    pc /= g_w.size(0)
    '''
    return pc

def SpearmanCorrelation(g_w, ce, recip=False):
    assert len(g_w.size())==2 and len(ce.size())==1, 'Necessitate checking out dimensions'

    g_w = g_w.mean(0).squeeze(0)
    if recip:
        g_w = torch.reciprocal(g_w)

    g_w = torch.FloatTensor(pd.DataFrame(g_w).rank().to_numpy()).t()
    ce = torch.FloatTensor(pd.DataFrame(ce).rank().to_numpy()).t()

    g_w = (g_w - torch.mean(g_w)) + 1e-6
    ce = (ce - torch.mean(ce)) + 1e-6

    rho = torch.sum(g_w*ce) / (torch.sqrt(torch.sum(g_w**2)) * torch.sqrt(torch.sum(ce**2)))

    return rho

def smoothing_onehot(labels):
    """Revisiting Knowledge Distillation via Label Smoothing Regularization, CVPR 2020
    Teacher free labeling 100 % accuracy
    """
    one_hot = F.one_hot(labels, NUM_CLASSES)
    target_ = ALPHA * one_hot
    noise_ = ((1-ALPHA)/(NUM_CLASSES-1)) * (1-one_hot)
    tf_targets = target_ + noise_
    return tf_targets

def covariance_loss(logits, labels, T, device):
    bsz, n_cats, n_heads = logits.size()
    if n_heads < 2:
        return 0
    all_probs = torch.softmax(logits/T, dim=1)
    label_inds = torch.ones(bsz, n_cats).cuda(device)
    label_inds[range(bsz), labels] = 0

    # removing the ground truth prob
    probs = all_probs * label_inds.unsqueeze(-1).detach()

    # re-normalize such that probs sum to 1
    #probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = (torch.softmax(logits/T, dim=1) + 1e-8)

    # cosine regularization
    #### I added under 2-line
    probs -= probs.mean(dim=1, keepdim=True)
    probs = probs / torch.sqrt(((probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
    ####
    #probs = probs / torch.sqrt(((all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
    cov_mat = torch.einsum('ijk,ijl->ikl', probs, probs)
    pairwise_inds = 1 - torch.eye(n_heads).cuda(device)
    den = bsz * (n_heads -1) * n_heads
    loss = ((cov_mat * pairwise_inds).abs().sum() / den)
    return loss

def similarity_loss(h_f, g_w, n_branches, types='mean'):
    f_stu = h_f[:,:,:,:,-1]
    features = f_stu.view(f_stu.size(0), -1)
    features_t = torch.t(features)

    stu_sim = torch.mm(features, features_t)

    features = h_f[:,:,:,:,0].view(h_f[:,:,:,:,0].size(0), -1)
    features_t = torch.t(features)
    en_sim = torch.mm(features, features_t)
    en_sim = en_sim.unsqueeze(-1)

    for i in range(1, n_branches-1):
        features = h_f[:,:,:,:,i].view(h_f[:,:,:,:,i].size(0), -1) # bsz x 1-d features
        features_t = torch.t(features) # 1-d feataures x bsz
        sim_tmp = torch.mm(features, features_t)
        en_sim = torch.cat([en_sim, sim_tmp.unsqueeze(-1)], dim=-1) # bsz x bsz x index

    if types=='mean':
        en_sim = en_sim.mean(-1)
    elif types=='weight':
        raise NotImplementedError
    else:
        raise RuntimeError('Type definition is essential!')

    diff = F.normalize(stu_sim) - F.normalize(en_sim) # <-- here version2
    loss = (diff * diff).view(-1, 1).sum(0)
    return loss

def batch_correlation_matrix(feat_samples, n_heads):
    fig = plt.figure()

    for i in range(n_heads):
        batch = feat_samples[:,:,:,:,i]
        batch = batch.view(batch.size(0), -1)
        batch_t = torch.t(batch)

        corr_matrix = torch.mm(batch, batch_t).cpu().detach().numpy()
        print(corr_matrix.shape)

        ax = fig.add_subplot(1,4,i+1)
        ax.plot(corr_matrix)

    plt.show()

if __name__ == '__main__':
    a = torch.rand(1, 3)
    b = torch.rand(3)
    SpearmanCorrelation(a, b)
