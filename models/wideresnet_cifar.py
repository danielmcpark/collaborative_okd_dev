import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['WideResNet', 'ONE_WideResNet', 'CLILR_WideResNet', 'OKDDip_WideResNet', 'wrn16_4']

class ILR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

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

class WideResNet(nn.Module):
    def __init__(self, dataset, depth, widen_factor=1, droprate=0.0):
        super(WideResNet, self).__init__()
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
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, droprate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        return self.fc(out)

class ONE_WideResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, widen_factor=1, droprate=0.0):
        super(ONE_WideResNet, self).__init__()
        self.num_branches = num_branches
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

        self.control_v1 = nn.Linear(nChannels[2], self.num_branches)
        self.bn_v1 = nn.BatchNorm1d(self.num_branches)
        self.avgpool_c = nn.AvgPool2d(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)

        x_c = self.avgpool_c(out)
        x_c = x_c.view(x_c.size(0), -1)
        x_c = self.control_v1(x_c)
        x_c = self.bn_v1(x_c)
        x_c = F.relu(x_c)
        X_c = F.softmax(x_c, dim=1)

        out_3 = getattr(self, 'block3_0')(out)
        out_3 = getattr(self, 'relu_0')(getattr(self, 'bn1_0')(out_3))
        out_3 = F.avg_pool2d(out_3, 8)
        out_3 = out_3.view(out_3.size(0), -1)
        out_3 = getattr(self, 'fc_0')(out_3)
        x_m = x_c[:,0].repeat(out_3.size(1), 1).transpose(0, 1) * out_3
        pro = out_3.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'block3_'+str(i))(out)
            en = getattr(self, 'relu_'+str(i))(getattr(self, 'bn1_'+str(i))(en))
            en = F.avg_pool2d(en, 8)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'fc_'+str(i))(en)
            x_m += x_c[:, i].repeat(en.size(1), 1).transpose(0, 1) * en
            en = en.unsqueeze(-1)
            pro = torch.cat([pro, en], -1)

        return pro, x_m

class OKDDip_WideResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, widen_factor=1, droprate=0.0, factor=4):
        super(OKDDip_WideResNet, self).__init__()
        self.num_branches = num_branches
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

        self.query_weight = nn.Linear(nChannels[3], nChannels[3]//factor, bias=False)
        self.key_weight = nn.Linear(nChannels[3], nChannels[3]//factor, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)

        out_3 = getattr(self, 'block3_0')(out)
        out_3 = getattr(self, 'relu_0')(getattr(self, 'bn1_0')(out_3))
        out_3 = F.avg_pool2d(out_3, 8)
        out_3 = out_3.view(out_3.size(0), -1)
        proj_q = self.query_weight(out_3)
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(out_3)
        proj_k = proj_k[:, None, :]
        out_3 = getattr(self, 'fc_0')(out_3)
        pro = out_3.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'block3_'+str(i))(out)
            temp = getattr(self, 'relu_'+str(i))(getattr(self, 'bn1_'+str(i))(temp))
            temp = F.avg_pool2d(temp, 8)
            temp = temp.view(temp.size(0), -1)
            temp_q = self.query_weight(temp)
            temp_k = self.key_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = temp_k[:, None, :]
            temp = getattr(self, 'fc_'+str(i))(temp)
            temp = temp.unsqueeze(-1)
            pro = torch.cat([pro, temp], -1)
            proj_q = torch.cat([proj_q, temp_q], 1)
            proj_k = torch.cat([proj_k, temp_k], 1)

        energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0, 2, 1))

        temp = getattr(self, 'block3_'+str(self.num_branches-1))(out)
        temp = getattr(self, 'relu_'+str(self.num_branches-1))(getattr(self, 'bn1_'+str(self.num_branches-1))(temp))
        temp = F.avg_pool2d(temp, 8)
        temp = temp.view(temp.size(0), -1)
        temp = getattr(self, 'fc_'+str(self.num_branches-1))(temp)
        return pro, x_m, temp

class CLILR_WideResNet(nn.Module):
    def __init__(self, dataset, depth, num_branches, widen_factor=1, bpscale=True, droprate=0.0):
        super(CLILR_WideResNet, self).__init__()
        self.num_branches = num_branches
        self.bpscale = bpscale
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.bpscale:
            self.layer_ILR = ILR.apply

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        if self.bpscale:
            out = self.layer_ILR(out, self.num_branches)

        out_3 = getattr(self, 'block3_0')(out)
        out_3 = getattr(self, 'relu_0')(getattr(self, 'bn1_0')(out_3))
        out_3 = F.avg_pool2d(out_3, 8)
        out_3 = out_3.view(out_3.size(0), -1)
        out_3 = getattr(self, 'fc_0')(out_3)
        pro = out_3.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'block3_'+str(i))(out)
            en = getattr(self, 'relu_'+str(i))(getattr(self, 'bn1_'+str(i))(en))
            en = F.avg_pool2d(en, 8)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'fc_'+str(i))(en)
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

def wrn16_4(dataset='cifar10', depth=16, widen_factor=4, **kwargs):
    model = WideResNet(dataset=dataset, depth=depth, widen_factor=widen_factor, **kwargs)
    return model

if __name__=='__main__':
    model = CLILR_WideResNet(dataset='cifar100', depth=16, widen_factor=8, num_branches=4)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    peer, y = model(x)
    print(peer.size())
