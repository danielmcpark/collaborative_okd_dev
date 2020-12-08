import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import *

__all__ = ['My_ResNetV3']

class GradientControl(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, num_branches):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output)
        return grad_output, None

class My_ResNetV3(nn.Module):
    def __init__(self, dataset, depth, num_branches, device,
        bottleneck=False, se=False):
        super(My_ResNetV3, self).__init__()
        self.inplanes = 16
        self.num_branches = num_branches
        self.device = device

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

        self.Gradient_Control = GradientControl.apply

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

        x = self.Gradient_Control(x, self.num_branches)
        with torch.no_grad():
            globals()['x_3_0'] = getattr(self, 'layer3_0')(x) # B x 64 x 8 x 8
            globals()['x_3_0'] = self.avgpool8x8(globals()['x_3_0'])
            globals()['x_3_0'] = globals()['x_3_0'].view(globals()['x_3_0'].size(0), -1)
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
        x_en = torch.mean(ind, dim=2)
        x_c = torch.zeros([x.size(0), self.num_branches-1]).cuda(self.device)
        x_c[:] = 1 / (self.num_branches-1)
        
        ## Student
        x_s = getattr(self, 'layer3_'+str(self.num_branches-1))(x) # B x 64 x 8 x 8
        x_s = self.avgpool8x8(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'classifier3_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1) # B x categories x branches

        return ind, x_en, x_c

if __name__ == '__main__':
    model = My_ResNetV3('cifar100', depth=20, num_branches=4, device='cuda:0', se=False)
    #print(model)
    x = torch.randn(2, 3, 32, 32)
    y, x_m, x_c = model(x)
    print(y.size(), x_m.size(), x_c.size())
