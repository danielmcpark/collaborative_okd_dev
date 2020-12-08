from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class KLLoss(nn.Module):
    def __init__(self, device):
        super(KLLoss, self).__init__()
        self.device = device

    def forward(self, pred, label, en=True):
        T=3

        predict = F.log_softmax(pred/T, dim=1)
        target_data = F.softmax(label/T, dim=1)
        target_data = target_data+10**(-7)
        target = Variable(target_data.data.cuda(self.device), requires_grad=True)

        # from implementation
        if en == True:
            loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size(0))
        else:
            loss=(target*(target.log()-predict)).sum(1).sum()/target.size(0)

        # from pytorch function
        #loss = T*T * nn.KLDivLoss(reduction='batchmean')(predict, target_data)
        return loss

class aJSLoss(nn.Module):
    def __init__(self, device, n_branches, temp):
        super(aJSLoss, self).__init__()
        self.device = device
        self.n_branches = n_branches
        self.T = temp

    def forward(self, pred, labels, coeff, label_idx, ty=None, en=True):
        assert ty is not None
        _, _, length = labels.size()
        predict = F.log_softmax(pred/self.T, dim=1) if ty=='JSD' or ty=='wJSD' else F.softmax(pred/self.T, dim=1)

        if ty == 'JSD':
            target_data = F.softmax(labels[:,:,0]/self.T, dim=1).unsqueeze(-1)
            for i in range(1, length):
                tmp = F.softmax(labels[:,:,i]/self.T, dim=1).unsqueeze(-1)
                target_data = torch.cat([target_data, tmp], dim=-1)
            target_data = target_data.mean(-1) + 10**(-7)
        elif ty == 'JD':
            predict = predict + 10**(-7)
            target_data = F.softmax(labels[:,:,label_idx+1]/self.T, dim=1) \
                    if label_idx < self.n_branches-2 else F.softmax(labels[:,:,0]/self.T, dim=1)
            target_data = target_data + 10**(-7)
        elif ty == 'wJSD':
            smoothing = F.softmax(labels[:,:,0]/self.T, dim=1)
            target_data = coeff[:,0].repeat(smoothing.size(1), 1).transpose(0, 1) * smoothing
            for i in range(1, length):
                smoothing = F.softmax(labels[:,:,i]/self.T, dim=1)
                target_data += coeff[:,i].repeat(smoothing.size(1), 1).transpose(0, 1) * smoothing
        else:
            raise ValueError('You have to choose cooperation type in [JSD, JD, wJSD]')

        target = Variable(target_data.data.cuda(self.device), requires_grad=True)
        # from implementation
        if ty == 'JD':
            loss = self.T*self.T*((((target*(target.log()-predict.log())) + (predict*(predict.log()-target.log()))).sum(1)).sum() / target.size(0)) if en == True else (((target*(target.log()-predict.log())) + (predict*(predict.log()-target.log()))).sum(1).sum() / target.size(0))
        else:
            loss = self.T*self.T*((target*(target.log()-predict)).sum(1).sum()/target.size(0)) \
                if en == True else (target*(target.log()-predict)).sum(1).sum()/target.size(0)
        if not torch.isfinite(loss):
            print('Warning: non-finite loss, ending training')
            exit(1)
        return loss
