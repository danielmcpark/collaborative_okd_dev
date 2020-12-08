import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .blocks import *

__all__ = ['My_WideResNet']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.droprate = dropout
        self.equalInOut = (in_planes==out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                            stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, droprate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, droprate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, droprate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, droprate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class My_WideResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, widen_factor=1, droprate=0.0, NL=False, embedding=False):
        super(My_WideResNet, self).__init__()
        self.num_branches = num_branches
        self.NL = NL
        self.embedding = embedding
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth-4) % 6 == 0)
        n = (depth-4) / 6

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("No valid dataset is given.")

        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, droprate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, droprate)

        for i in range(self.num_branches):
            setattr(self, 'block3_'+str(i), NetworkBlock(n, nChannels[2], nChannels[3], block, 2, droprate))
            setattr(self, 'bn1_'+str(i), nn.BatchNorm2d(nChannels[3]))
            setattr(self, 'relu_'+str(i), nn.ReLU(inplace=True))
            setattr(self, 'fc_'+str(i), nn.Linear(nChannels[3], num_classes))

        self.avgpool16x16 = nn.AvgPool2d(16)
        if self.NL:
            self.phi = nn.Conv2d(in_channels=nChannels[2], out_channels=1, kernel_size=1)
            if self.embedding:
                self.theta = nn.Conv2d(in_channels=nChannels[2], out_channels=nChannels[2], kernel_size=1)

            for i in range(self.num_branches-1):
                setattr(self, 'phi_'+str(i), nn.Conv2d(in_channels=nChannels[3], out_channels=1, kernel_size=1))
                if self.embedding:
                    setattr(self, 'theta_'+str(i), nn.Conv2d(in_channels=nChannels[3], out_channels=nChannels[3], kernel_size=1))

        self.avgpool8x8 = nn.AvgPool2d(8)
        self.gatelayer = GateBlock(nChannels[2]+(self.num_branches-1)*nChannels[3], self.num_branches-1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)

        globals()['out_3_0'] = getattr(self, 'block3_0')(out)
        globals()['out_3_0'] = getattr(self, 'relu_0')(getattr(self, 'bn1_0')(globals()['out_3_0']))

        ## High-level Context modeling
        b, c, _, _ = globals()['out_3_0'].size()
        if self.NL:
            phi = getattr(self, 'phi_0')(globals()['out_3_0'])
            phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
            phi = F.softmax(phi, dim=-1).unsqueeze(-1)
            if self.embedding:
                theta = getattr(self, 'theta_0')(globals()['out_3_0'])
                theta = theta.view(theta.size(0), theta.size(1), -1)
            else:
                theta = globals()['out_3_0'].view(globals()['out_3_0'].size(0), globals()['out_3_0'].size(1), -1)
            context = torch.bmm(theta, phi).squeeze(-1)
        else:
            context = self.avgpool8x8(globals()['out_3_0']).view(b, c)

        globals()['out_3_0'] = self.avgpool8x8(globals()['out_3_0'])
        globals()['out_3_0'] = globals()['out_3_0'].view(globals()['out_3_0'].size(0), -1)
        globals()['out_3_0'] = getattr(self, 'fc_0')(globals()['out_3_0'])
        ind = globals()['out_3_0'].unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            globals()['out_3_{}'.format(i)] = getattr(self, 'block3_'+str(i))(out)
            globals()['out_3_{}'.format(i)] = getattr(self, 'relu_'+str(i))(getattr(self, 'bn1_'+str(i))(globals()['out_3_{}'.format(i)]))

            ## Context modeling
            if self.NL:
                phi = getattr(self, 'phi_'+str(i))(globals()['out_3_{}'.format(i)])
                phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
                phi = F.softmax(phi, dim=-1).unsqueeze(-1)
                if self.embedding:
                    theta = getattr(self, 'theta_'+str(i))(globals()['out_3_{}'.format(i)])
                    theta = theta.view(theta.size(0), theta.size(1), -1)
                else:
                    theta = globals()['out_3_{}'.format(i)].view(globals()['out_3_{}'.format(i)].size(0), globals()['out_3_{}'.format(i)].size(1), -1)
                temp_s = torch.bmm(theta, phi).squeeze(-1)
            else:
                temp_s = self.avgpool8x8(globals()['out_3_{}'.format(i)]).view(b, c)
            context = torch.cat([context, temp_s], dim=1)

            globals()['out_3_{}'.format(i)] = self.avgpool8x8(globals()['out_3_{}'.format(i)])
            globals()['out_3_{}'.format(i)] = globals()['out_3_{}'.format(i)].view(globals()['out_3_{}'.format(i)].size(0), -1)
            globals()['out_3_{}'.format(i)] = getattr(self, 'fc_'+str(i))(globals()['out_3_{}'.format(i)])
            globals()['out_3_{}'.format(i)] = globals()['out_3_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['out_3_{}'.format(i)]], -1)

        ## Approximation
        #x_en = torch.mean(ind, dim=2)
        b1, c1, _, _ = out.size()
        if self.NL:
            phi = self.phi(out)
            phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
            phi = F.softmax(phi, dim=-1).unsqueeze(-1)
            if self.embedding:
                theta = self.theta(out)
                theta = theta.view(theta.size(0), theta.size(1), -1)
            else:
                theta = out.view(out.size(0), out.size(1), -1)
            x_c = torch.bmm(theta, phi).squeeze(-1)
        else:
            # Lower-level Context modeling
            x_c = self.avgpool16x16(out).view(b1, c1)
        x_c = torch.cat([x_c, context], dim=1)
        x_c = self.gatelayer(x_c)
        x_en = x_c[:,0].repeat(ind[:,:,0].size(1), 1).transpose(0, 1) * ind[:,:,0]
        for i in range(1, self.num_branches-1):
            x_en += x_c[:,i].repeat(ind[:,:,i].size(1), 1).transpose(0, 1) * ind[:,:,i]

        # Student
        x_s = getattr(self, 'block3_'+str(self.num_branches-1))(out)
        x_s = getattr(self, 'relu_'+str(self.num_branches-1))(getattr(self, 'bn1_'+str(self.num_branches-1))(x_s))
        x_s = self.avgpool8x8(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'fc_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1)

        return ind, x_en, x_c

if __name__=='__main__':
    model = My_WideResNet(dataset='cifar100', depth=16, widen_factor=8, num_branches=4, NL=True)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    peer, y, _ = model(x)
    print(peer.size())
