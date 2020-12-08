import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import *
#from gatelayer import *

__all__ = ['My_ResNetO']

class My_ResNetO(nn.Module):
    def __init__(self, dataset, depth, num_branches,
            bottleneck=False, NL=False, embedding=False, se=False):
        super(My_ResNetO, self).__init__()
        self.inplanes = 16
        self.num_branches = num_branches
        self.NL = NL
        self.embedding = embedding

        if bottleneck is True:
            n = (depth - 2) // 9
            if se:
                block = SEBottleneck
            else:
                block = Bottleneck
        else:
            n = (depth - 2) // 6
            if se:
                block = SEBasicBlock
            else:
                block = BasicBlock
        self.block = block

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("No valid dataset is given.")

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)

        fix_inplanes = self.inplanes
        self.avgpool8x8 = nn.AvgPool2d(8)

        ## for coefficient
        self.avgpool16x16 = nn.AvgPool2d(16)
        self.gatelayer = GateBlock(fix_inplanes, self.num_branches-1, bias=True)

        if self.NL:
            self.phi = nn.Conv2d(in_channels=32 * block.expansion, out_channels=1, kernel_size=1)
            if self.embedding:
                self.theta = nn.Conv2d(in_channels=32 * block.expansion, out_channels=32 * block.expansion, kernel_size=1)

        for i in range(self.num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(self.block, 64, n, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_'+str(i), nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x) # B x 32 x 16 x 16

        globals()['x_3_0'] = getattr(self, 'layer3_0')(x) # B x 64 x 8 x 8
        globals()['x_3_0'] = self.avgpool8x8(globals()['x_3_0']).view(globals()['x_3_0'].size(0), -1)
        globals()['x_3_0'] = getattr(self, 'classifier3_0')(globals()['x_3_0'])
        ind = globals()['x_3_0'].unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            globals()['x_3_{}'.format(i)] = getattr(self, 'layer3_'+str(i))(x) # B x 64 x 8 x 8
            globals()['x_3_{}'.format(i)] = self.avgpool8x8(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)].view(globals()['x_3_{}'.format(i)].size(0), -1)
            globals()['x_3_{}'.format(i)] = getattr(self, 'classifier3_'+str(i))(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['x_3_{}'.format(i)]], -1) # B x categories x num_branches-1

        ## Peer attention
        #x_en = torch.mean(ind, dim=2)
        b1, c1, _, _ = x.size()
        if self.NL:
            phi = self.phi(x)
            phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
            phi = F.softmax(phi, dim=-1).unsqueeze(-1)
            if self.embedding:
                theta = self.theta(x)
                theta = theta.view(theta.size(0), theta.size(1), -1)
            else:
                theta = x.view(x.size(0), x.size(1), -1)
            x_c = torch.bmm(theta, phi).squeeze(-1)
        else:
            x_c = self.avgpool16x16(x).view(b1, c1)
        x_c = self.gatelayer(x_c)
        x_en = x_c[:,0].repeat(ind[:,:,0].size(1), 1).transpose(0, 1) * ind[:,:,0]
        for i in range(1, self.num_branches-1):
            x_en += x_c[:,i].repeat(ind[:,:,i].size(1), 1).transpose(0, 1) * ind[:,:,i]

        ## Student
        x_s = getattr(self, 'layer3_'+str(self.num_branches-1))(x) # B x 64 x 8 x 8
        x_s = self.avgpool8x8(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'classifier3_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1) # B x categories x branches

        return ind, x_en, x_c

if __name__ == '__main__':
    model = My_ResNetO('cifar100', depth=20, num_branches=4, NL=False, embedding=False, se=False)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    y, x_m, x_c = model(x)
    print(y.size(), x_m.size(), x_c.size())
