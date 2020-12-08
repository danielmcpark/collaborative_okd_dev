import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import *

__all__ = ['ONE_ResNet', 'OKDDip_ResNet', 'CLILR_ResNet', 'ResNet',
           'resnet20', 'resnet32', 'resnet56', 'resnet110']

class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None


class ResNet(nn.Module):
    def __init__(self, dataset, depth, bottleneck=False, se=False):
        super(ResNet, self).__init__()
        self.inplanes = 16

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
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x

        #x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, f1, f2

class ONE_ResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, bottleneck=False, se=False):
        super(ONE_ResNet, self).__init__()
        self.inplanes = 16
        self.num_branches = num_branches

        if bottleneck is True:
            n = (depth - 2) // 9
            if se:
                block = SEBottleneck
            else:
                block = Bottleneck
        else:
            n = (depth - 2) // 6
            if se:
                block =SEBasicBlock
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

        for i in range(self.num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(block, 64, n, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_'+str(i), nn.Linear(64 * block.expansion, num_classes))

        self.control_v1 = nn.Linear(fix_inplanes, self.num_branches)
        self.bn_v1 = nn.BatchNorm1d(self.num_branches)
        self.avgpool = nn.AvgPool2d(8)
        self.avgpool_c = nn.AvgPool2d(16)

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
        x = self.layer2(x)

        x_c = self.avgpool_c(x)
        x_c = x_c.view(x_c.size(0), -1)

        x_c = self.control_v1(x_c)
        x_c = self.bn_v1(x_c)
        x_c = F.relu(x_c)
        x_c = F.softmax(x_c, dim=1)

        x_3 = getattr(self, 'layer3_0')(x)
        x_3 = self.avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier3_0')(x_3)
        x_m = x_c[:,0].repeat(x_3.size(1), 1).transpose(0, 1) * x_3
        pro = x_3.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'layer3_'+str(i))(x)
            en = self.avgpool(en)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'classifier3_'+str(i))(en)
            x_m += x_c[:,i].repeat(en.size(1), 1).transpose(0, 1) * en
            en = en.unsqueeze(-1)
            pro = torch.cat([pro, en], -1)

        return pro, x_m

class OKDDip_ResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches=4, input_channel=64, factor=4, bottleneck=False, en=False, zero_init_residual=False, se=False):
        super(OKDDip_ResNet, self).__init__()
        self.en = en
        self.num_branches = num_branches
        self.inplanes = 16
        
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
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        
        fix_inplanes = self.inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        for i in range(self.num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(block, 64, n, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_'+str(i), nn.Linear(64 * block.expansion, num_classes))

        self.query_weight = nn.Linear(input_channel*block.expansion, input_channel//factor, bias=False)
        self.key_weight = nn.Linear(input_channel*block.expansion, input_channel//factor, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample =None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        # B x 16 x 32 x 32

        x = self.layer1(x)      # B x 16 x 32 x 32
        x = self.layer2(x)      # B x 32 x 16 x 16

        x_3 = getattr(self, 'layer3_0')(x)  # B x 64 x 8 x 8
        x_3 = self.avgpool(x_3)             # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        proj_q = self.query_weight(x_3)     # B x 8
        proj_q = proj_q[:, None, :]         # B x 1 x 8
        proj_k = self.key_weight(x_3)       # B x 8
        proj_k = proj_k[:, None, :]         # B x 1 x 8
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)                       # B x num_classes x 1

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'layer3_'+str(i))(x)
            temp = self.avgpool(temp)           # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            temp_q = self.query_weight(temp)
            temp_k = self.key_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = temp_k[:, None, :]
            temp_1 = getattr(self, 'classifier3_'+str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro, temp_1], -1)      # B x num_classes x num_branches
            proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
            proj_k = torch.cat([proj_k, temp_k], 1)

        energy = torch.bmm(proj_q, proj_k.permute(0,2,1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0,2,1)) # Teacher
        if self.en:
            return pro, x_m
        else:
            temp = getattr(self, 'layer3_'+str(self.num_branches - 1))(x)
            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            temp_out = getattr(self, 'classifier3_'+str(self.num_branches - 1))(temp) # Student
            return pro, x_m, temp_out

class CLILR_ResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, bottleneck=False, bpscale=True, se=False):
        super(CLILR_ResNet, self).__init__()
        self.inplanes = 16
        self.num_branches = num_branches
        self.bpscale = bpscale

        if bottleneck is True:
            n = (depth - 2) // 9
            if se:
                block =SEBottleneck
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        for i in range(self.num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(block, 64, n, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_'+str(i), nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.bpscale:
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches) # Backprop rescaling

        x_3 = getattr(self, 'layer3_0')(x)
        x_3 = self.avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier3_0')(x_3)
        pro = x_3.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'layer3_'+str(i))(x)
            en = self.avgpool(en)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'classifier3_'+str(i))(en)
            en = en.unsqueeze(-1)
            pro = torch.cat([pro, en], -1)

        x_m = 0
        for i in range(1, self.num_branches):
            x_m += 1/(self.num_branches-1) * pro[:,:,i]
        x_m = x_m.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp = 0
            for j in range(0, self.num_branches):
                if j != i:
                    temp += 1/(self.num_branches-1) * pro[:,:,j]
            temp = temp.unsqueeze(-1)
            x_m = torch.cat([x_m, temp], -1)

        return pro, x_m

def resnet20(dataset='cifar10', depth=20, bottleneck=False, **kwargs):
    model = ResNet(dataset=dataset, depth=depth, bottleneck=bottleneck, **kwargs)
    return model

def resnet32(dataset='cifar10', depth=32, bottleneck=False, **kwargs):
    model = ResNet(dataset=dataset, depth=depth, bottleneck=bottleneck, **kwargs)
    return model

def resnet56(dataset='cifar10', depth=56, bottleneck=False, **kwargs):
    model = ResNet(dataset=dataset, depth=depth, bottleneck=bottleneck, **kwargs)
    return model

def resnet110(dataset='cifar10', depth=110, bottleneck=True, **kwargs):
    model = ResNet(dataset=dataset, depth=depth, bottleneck=bottleneck, **kwargs)
    return model

if __name__ == '__main__':
    model = ResNet('cifar100', 20)
    print(model.block.expansion)
    x = torch.randn(2, 3, 32, 32)
    _, y = model(x)
    print(y.size())
