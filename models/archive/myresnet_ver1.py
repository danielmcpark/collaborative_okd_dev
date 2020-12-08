import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#from gatelayer import *

__all__ = ['My_ResNet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                      bias=False)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def ConvBNReLU(inp, oup, kernel_size=3, stride=1, padding=0, groups=1):
    return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
            )

def ConvDWReLU(inp, oup, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
            )

class OriginalMobile(nn.Module):
    def __init__(self, inp, oup, stride):
        super(OriginalMobile, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        layers = []
        layers.append(ConvDWReLU(inp, oup, 3, 1, 1, inp))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        layers.append(ConvBNReLU(inp, inp*expand_ratio, 1, 1, 0))
        layers.extend([
            ConvBNReLU(inp*expand_ratio, inp*expand_ratio, 3, stride=stride, padding=1, groups=inp*expand_ratio),
            nn.Conv2d(inp*expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #x = self.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #x = self.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

class My_ResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, b_types, bottleneck=False):
        super(My_ResNet, self).__init__()
        self.inplanes = 16
        self.num_branches = num_branches
        self.b_types = b_types

        if bottleneck is True:
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            n = (depth - 2) // 6
            block = BasicBlock
        self.block = block
        self.block2 = InvertedResidual

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
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool = nn.AvgPool2d(8)

        ## for coefficient
        #self.globalavgpool = nn.AvgPool2d(16)
        #self.gatelinear = nn.Linear(fix_inplanes, self.num_branches-1)
        #self.bn_gate = nn.BatchNorm1d(self.num_branches-1)
        #self.gate = GateLayer(self.num_branches-1)

        # For SEBlock
        self.globalavgpool = nn.AvgPool1d(num_classes)
        self.squeeze = nn.Linear(self.num_branches, self.num_branches // 2)
        self.excitate = nn.Linear(self.num_branches // 2, self.num_branches)

        for i in range(self.num_branches):
            if self.b_types == 'invmobile':
                setattr(self, 'layer3_'+str(i), self._make_layer(block, 64, n, stride=2)) if i == self.num_branches-1 else \
                setattr(self, 'layer3_'+str(i), self._make_invmobile_block(InvertedResidual, 32))
            elif self.b_types =='mobile':
                setattr(self, 'layer3_'+str(i), self._make_layer(block, 64, n, stride=2)) if i == self.num_branches-1 else \
                setattr(self, 'layer3_'+str(i), self._make_mobile_block(OriginalMobile, 32))
            elif self.b_types == 'residual':
                setattr(self, 'layer3_'+str(i), self._make_layer(block, 64, n, stride=2))
            else:
                raise ValueError('You must set block types!!')
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_'+str(i), nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.layer_ILR = ILR.apply

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

    def _make_invmobile_block(self, block, planes):
        round_nearest = 8
        width_mult=1.0
        inv_mobile_config = [6, 64, 4, 2] # t, c, n, s
        output_channel = _make_divisible(inv_mobile_config[1] * width_mult, round_nearest)

        layers = []
        for i in range(inv_mobile_config[2]):
            stride=inv_mobile_config[3] if i == 0 else 1
            layers.append(block(planes, output_channel, stride, expand_ratio=inv_mobile_config[0]))
            planes = output_channel

        return nn.Sequential(*layers)

    def _make_inception_block(self, block, planes):
        raise NotImplementedError

    def _make_mobile_block(self, block, planes):
        mobile_config = [planes, 64, 1] # i, o, s
        layers = []
        layers.append(block(inp=mobile_config[0], oup=mobile_config[1], stride=mobile_config[2]))
        return nn.Sequential(*layers)

    def _make_shuffle_block(self, block, planes):
        raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x) # B x 32 x 16 x 16

        x = self.layer_ILR(x, self.num_branches)

        ## Global Average Pooling for Squeeze-and-Excitation
        #x_c = self.globalavgpool(x)
        #x_c = x_c.view(x_c.size(0), -1)

        ## Gate
        #x_c = self.gatelinear(x_c)
        #x_c = self.bn_gate(x_c)
        #x_c = self.gate(x_c)
        #x_c = F.softmax(x_c, dim=1)

        ## Peers
        x_3 = getattr(self, 'layer3_0')(x) # B x 64 x 8 x 8

        ## Batch similarity
        #feats = x_3.view(x_3.size(0),-1)
        #feats_t = torch.t(feats)
        #s_map = F.normalize(torch.mm(feats, feats_t)).unsqueeze(-1)
        #s_map = x_c[:,0] * torch.mm(feats, feats_t)

        x_3 = self.avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier3_0')(x_3)
        #x_en = x_c[:,0].repeat(x_3.size(1), 1).transpose(0, 1) * x_3
        ind = x_3.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'layer3_'+str(i))(x) # B x 64 x 8 x 8

            ## Batch similarity
            #feats = temp.view(temp.size(0), -1)
            #feats_t = torch.t(feats)
            #tmp_map = F.normalize(torch.mm(feats, feats_t)).unsqueeze(-1)
            #s_map = torch.cat([s_map, tmp_map], -1)
            #s_map += x_c[:,i] * torch.mm(feats, feats_t)

            temp = self.avgpool(temp)
            temp = temp.view(temp.size(0), -1)
            temp = getattr(self, 'classifier3_'+str(i))(temp)
            #x_en += x_c[:,i].repeat(temp.size(1), 1).transpose(0, 1) * temp
            temp = temp.unsqueeze(-1)
            ind = torch.cat([ind, temp], -1)

        ## Student
        #s_map = s_map.unsqueeze(-1)
        x_s = getattr(self, 'layer3_'+str(self.num_branches-1))(x) # B x 64 x 8 x 8

        ## Batch similarity
        #feats = x_3.view(x_3.size(0),-1)
        #feats_t = torch.t(feats)
        #s_map =  torch.cat([F.normalize(s_map), F.normalize(torch.mm(feats, feats_t)).unsqueeze(-1)], -1)

        x_s = self.avgpool(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'classifier3_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1) # B x categories x branches

        # Into the squeeze-and-excitation
        ind_ = ind.permute(0,2,1) # B x branches x categories
        scores = self.globalavgpool(ind_) # B x branches x 1
        scores = scores.view(scores.size(0), -1) # B x branches
        scores = self.squeeze(scores)
        scores = self.excitate(F.relu(scores))
        scores = F.sigmoid(scores).unsqueeze(-1) # B x branches x 1
        scores = scores.permute(0,2,1) # B x 1 x branches

        x_en = torch.sum((ind * scores), 2)

        return ind, x_en

if __name__ == '__main__':
    model = My_ResNet('cifar100', 20, 4, 'residual')
    print(model)
    x = torch.randn(2, 3, 32, 32)
    y, x_m = model(x)
    print(y.size(), x_m.size())
