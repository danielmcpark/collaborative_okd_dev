import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = ['My_MobileNetV1']

class Block(nn.Module):
    '''depthwise conv + pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class My_MobileNetV1(nn.Module):
    def __init__(self, dataset, num_branches, shallow=None, dw_block_setting_front=None, alpha=1.0, dw_block_setting_end=None):
        super(My_MobileNetV1, self).__init__()
        self.num_branches = num_branches
        if dataset == 'cifar10':
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.num_classes = 100
        else:
            raise ValueError("Unvalid datasets!")

        def ConvBNReLU(in_planes, out_planes, stride):
            return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(inplace=True)
            )

        def ConvDWReLU(in_planes, out_planes, stride):
            return nn.Sequential(
                    nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False),
                    nn.BatchNorm2d(in_planes),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(inplace=True),
            )

        self.input_channel = 32
        self.last_channel = 1024
        self.shallow = shallow
        self.alpha = alpha

        if dw_block_setting_front is None:
            dw_block_setting_front = [
                    # i, o, s
                    [int(32*self.alpha),  int(64*self.alpha),  1],
                    [int(64*self.alpha),  int(128*self.alpha), 2],
                    [int(128*self.alpha), int(128*self.alpha), 1],
                    [int(128*self.alpha), int(256*self.alpha), 2],
                    [int(256*self.alpha), int(256*self.alpha), 1],
                    [int(256*self.alpha), int(512*self.alpha), 2]
            ]

        if len(dw_block_setting_front) == 0 or len(dw_block_setting_front[0]) != 3:
            raise ValueError("dw_block_setting_front should be non-empty "
                             "or a 3-element list, got {}".format(dw_block_setting_front))

        features = [ConvBNReLU(3, int(self.input_channel*self.alpha), stride=1)]
        for i, o, s in dw_block_setting_front:
            features.append(ConvDWReLU(in_planes=i, out_planes=o, stride=s))
        if self.shallow is None:
            for i in range(5):
                features.append(ConvDWReLU(in_planes=int(512*self.alpha),
                                            out_planes=int(512*self.alpha),
                                            stride=1))
        self.features = nn.Sequential(*features)

        ## For GAP
        self.avgpool4x4 = nn.AvgPool2d(4) # first context
        self.avgpool2x2 = nn.AvgPool2d(2) # second context

        ## Gatelayer
        self.gatelayer = GateBlock(512+(self.num_branches-1)*1024, self.num_branches-1, bias=False)

        ## For Ensemble
        for i in range(self.num_branches):
            setattr(self, 'peer0_'+str(i), ConvDWReLU(in_planes=int(512*self.alpha), out_planes=int(1024*self.alpha), stride=2))
            setattr(self, 'peer1_'+str(i), ConvDWReLU(in_planes=int(1024*self.alpha), out_planes=int(1024*self.alpha), stride=1))
            setattr(self, 'classifier_'+str(i), nn.Linear(int(self.last_channel*self.alpha), self.num_classes))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        globals()['x_0'] = getattr(self, 'peer0_0')(x)
        globals()['x_0'] = getattr(self, 'peer1_0')(globals()['x_0'])
        
        # Context extraction
        b, c, _, _ = globals()['x_0'].size()
        context = self.avgpool2x2(globals()['x_0']).view(b, c) # B x 1024

        globals()['x_0'] = self.avgpool2x2(globals()['x_0'])
        globals()['x_0'] = globals()['x_0'].view(globals()['x_0'].size(0), -1)
        globals()['x_0'] = getattr(self, 'classifier_0')(globals()['x_0'])
        ind = globals()['x_0'].unsqueeze(-1)
        for i in range(1, self.num_branches-1):
            globals()['x_{}'.format(i)] = getattr(self, 'peer0_'+str(i))(x)
            globals()['x_{}'.format(i)] = getattr(self, 'peer1_'+str(i))(globals()['x_{}'.format(i)])

            # Context Extraction
            temp_s = self.avgpool2x2(globals()['x_{}'.format(i)]).view(b, c) # B x 1024
            context = torch.cat([context, temp_s], dim=1)

            globals()['x_{}'.format(i)] = self.avgpool2x2(globals()['x_{}'.format(i)])
            globals()['x_{}'.format(i)] = globals()['x_{}'.format(i)].view(globals()['x_{}'.format(i)].size(0), -1)
            globals()['x_{}'.format(i)] = getattr(self, 'classifier_'+str(i))(globals()['x_{}'.format(i)])
            globals()['x_{}'.format(i)] = globals()['x_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['x_{}'.format(i)]], -1)

        ## Context aggregation and ensemble modeling
        b1, c1, _, _ = x.size()
        x_c = self.avgpool4x4(x).view(b1, c1)
        x_c = torch.cat([x_c, context], dim=1)
        x_c = self.gatelayer(x_c)
        x_en = x_c[:,0].repeat(ind[:,:,0].size(1), 1).transpose(0, 1) * ind[:,:,0]
        for i in range(1, self.num_branches-1):
            x_en += x_c[:,i].repeat(ind[:,:,i].size(1), 1).transpose(0, 1) * ind[:,:,i]

        # Student
        x_s = getattr(self, 'peer0_'+str(self.num_branches-1))(x)
        x_s = getattr(self, 'peer1_'+str(self.num_branches-1))(x_s)
        x_s = self.avgpool2x2(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'classifier_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1)

        return ind, x_en, x_c

if __name__ == '__main__':
    net = My_MobileNetV1(dataset='cifar100', num_branches=4, alpha=1.0)
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y, en, c = net(x)
    print(y.size(), en.size(), c.size())

