import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = ['My_VGG']

class My_VGG(nn.Module):
    def __init__(self, dataset, depth, num_branches):
        super(My_VGG, self).__init__()
        self.inplanes = 64
        self.num_branches = num_branches
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError('No valid dataset!!..')

        if depth is 16:
            num_layer = 3
        elif depth is 19:
            num_layer = 4
        else:
            raise ValueError('Check out depth!!..')
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layers(128, 2)
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)

        for i in range(self.num_branches):
            setattr(self, 'layer4_'+str(i), self._make_layers(512, num_layer))
            setattr(self, 'classifier_'+str(i), nn.Linear(512, num_classes))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.gatelayer = GateBlock(512+(self.num_branches-1)*512, self.num_branches-1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplanes, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplanes=input
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x_ = self.layer3(x)
        x = self.maxpool(x_)

        x_3 = getattr(self, 'layer4_0')(x)
        
        # Context
        b, c, _, _ = x_3.size()
        context = self.avgpool(x_3).view(b, c)

        x_3 = self.maxpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier_0')(x_3)
        ind = x_3.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'layer4_'+str(i))(x)

            # Context
            temp_s = self.avgpool(temp).view(b, c)
            context = torch.cat([temp_s, context], -1)

            temp = self.maxpool(temp)
            temp = temp.view(temp.size(0), -1)
            temp = getattr(self, 'classifier_'+str(i))(temp)
            temp = temp.unsqueeze(-1)
            ind = torch.cat([ind, temp], -1)

        b1, c1, _, _ = x.size()
        x_c = self.avgpool(x_).view(b1, c1)
        x_c = torch.cat([x_c, context], dim=1)
        x_c = self.gatelayer(x_c)
        x_en = x_c[:,0].repeat(ind[:,:,0].size(1), 1).transpose(0, 1) * ind[:,:,i]
        for i in range(1, self.num_branches-1):
            x_en += x_c[:,i].repeat(ind[:,:,i].size(1), 1).transpose(0, 1) * ind[:,:,i]

        ## Student
        x_s = getattr(self, 'layer4_'+str(self.num_branches-1))(x)
        x_s = self.maxpool(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'classifier_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1)

        return ind, x_en, x_c
    
if __name__=='__main__':
    model = My_VGG(dataset='cifar100', depth=16, num_branches=4)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    ind, y = model(x)
    print(ind.size(), y.size())


