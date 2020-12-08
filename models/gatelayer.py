from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class GateLayer_F(torch.autograd.Function):
    def forward(ctx, input, alpha, beta):
        mean = beta.data.mean()

        positive = beta.ge(mean)
        negative = beta.lt(mean)
        alpha[positive] = 1.0
        alpha[negative] = 0.0

        #print(alpha)
        #print(beta)
        # save alpha after we modify it
        ctx.save_for_backward(input, alpha)
        if not (alpha.eq(1.0).sum() + alpha.eq(0.0).sum() == alpha.nelement()):
            raise ValueError('Error: Please set the weight decay and lr of alpha to 0.0')
        if len(input.shape) ==4:
            input = input.mul(alpha.unsqueeze(2).unsqueeze(3))
        else:
            input = input.mul(alpha)
        return input

    def backward(ctx, grad_output):
        input, alpha = ctx.saved_variables
        grad_input = grad_output.clone()
        if len(input.shape) == 4:
            grad_input = grad_input.mul(alpha.data.unsqueeze(2).unsqueeze(3))
        else:
            grad_input = grad_input.mul(alpha.data)

        grad_beta = grad_output.clone()
        grad_beta = grad_beta.mul(input.data).sum(0, keepdim=True)
        if len(grad_beta.shape) == 4:
            grad_beta = grad_beta.sum(3).sum(2)
        return grad_input, None, grad_beta

class GateLayer(nn.Module):
    def __init__(self, size=1, beta_initial=0.8003, beta_limit=1.0):
        assert size>0
        super(GateLayer, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1, size).zero_().add(1.0), requires_grad=False)
        self.beta = nn.Parameter(torch.FloatTensor(1, size).zero_().add(beta_initial))
        self.beta_limit = beta_limit
        return

    def forward(self, x):
        self.beta.data.clamp_(0.0, self.beta_limit)
        x = GateLayer_F()(x, self.alpha, self.beta)
        return x


