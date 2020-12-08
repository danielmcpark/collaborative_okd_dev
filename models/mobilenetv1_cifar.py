import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

__all__ = ['MobileNetV1_Cifar', 'ONE_MobileNetV1', 'CLILR_MobileNetV1', 'OKDDip_MobileNetV1', 'mbn_v1']

class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

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

class MobileNetV1_Cifar(nn.Module):
    def __init__(self, dataset, shallow=None, dw_block_setting_front=None, alpha=1.0, dw_block_setting_end=None):
        super(MobileNetV1_Cifar, self).__init__()
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
        self.avgpool = nn.AvgPool2d(2)

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
        if dw_block_setting_end is None:
            dw_block_setting_end = [
                    # i, o, s
                    [int(512*self.alpha), int(1024*self.alpha), 2],
                    [int(1024*self.alpha), int(1024*self.alpha), 1]
            ]
        if len(dw_block_setting_front) == 0 or len(dw_block_setting_front[0]) != 3:
            raise ValueError("dw_block_setting_front should be non-empty "
                             "or a 3-element list, got {}".format(dw_block_setting_front))
        if len(dw_block_setting_end) == 0 or len(dw_block_setting_end[0]) != 3:
            raise ValueError("dw_block_setting_front should be non-empty "
                             "or a 3-element list, got {}".format(dw_block_setting_end))

        features = [ConvBNReLU(3, int(self.input_channel*self.alpha), stride=1)]
        for i, o, s in dw_block_setting_front:
            features.append(ConvDWReLU(in_planes=i, out_planes=o, stride=s))
        if self.shallow is None:
            for i in range(5):
                features.append(ConvDWReLU(in_planes=int(512*self.alpha),
                                            out_planes=int(512*self.alpha),
                                            stride=1))
        for i, o, s in dw_block_setting_end:
            features.append(ConvDWReLU(in_planes=i, out_planes=o, stride=s))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(int(self.last_channel*self.alpha), self.num_classes)

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
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ONE_MobileNetV1(nn.Module):
    def __init__(self, dataset, num_branches, shallow=None, dw_block_setting_front=None, alpha=1.0):
        super(ONE_MobileNetV1, self).__init__()
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
        self.avgpool = nn.AvgPool2d(2)

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

        self.avgpool_c = nn.AvgPool2d(4)
        self.control_v1 = nn.Linear(512, self.num_branches)
        self.bn_v1 = nn.BatchNorm1d(self.num_branches)

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
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        
        x_c = self.avgpool_c(x)
        x_c = x_c.view(x_c.size(0), -1)

        x_c = self.control_v1(x_c)
        x_c = self.bn_v1(x_c)
        x_c = F.relu(x_c)
        x_c = F.softmax(x_c, dim=1)

        x_4 = getattr(self, 'peer0_0')(x)
        x_4 = getattr(self, 'peer1_0')(x_4)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)
        x_4 = getattr(self, 'classifier_0')(x_4)
        x_m = x_c[:,0].repeat(x_4.size(1), 1).transpose(0, 1) * x_4
        pro = x_4.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'peer0_'+str(i))(x)
            en = getattr(self, 'peer1_'+str(i))(en)
            en = self.avgpool(en)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'classifier_'+str(i))(en)
            x_m += x_c[:,i].repeat(en.size(1), 1).transpose(0, 1) * en
            en = en.unsqueeze(-1)
            pro = torch.cat([pro, en], -1)

        return pro, x_m

class CLILR_MobileNetV1(nn.Module):
    def __init__(self, dataset, num_branches, shallow=None, dw_block_setting_front=None, alpha=1.0, bpscale=True):
        super(CLILR_MobileNetV1, self).__init__()
        self.num_branches = num_branches
        self.bpscale = bpscale

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
        self.avgpool = nn.AvgPool2d(2)

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
                nn.init.zeros_(m.bias)

        if self.bpscale:
            self.layer_ILR = ILR.apply

    def forward(self, x):
        x = self.features(x)
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches)

        x_4 = getattr(self, 'peer0_0')(x)
        x_4 = getattr(self, 'peer1_0')(x_4)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)
        x_4 = getattr(self, 'classifier_0')(x_4)
        pro = x_4.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'peer0_'+str(i))(x)
            en = getattr(self, 'peer1_'+str(i))(en)
            en = self.avgpool(en)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'classifier_'+str(i))(en)
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

class OKDDip_MobileNetV1(nn.Module):
    def __init__(self, dataset, num_branches, factor=8, shallow=None, dw_block_setting_front=None, alpha=1.0):
        super(OKDDip_MobileNetV1, self).__init__()
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
        self.avgpool = nn.AvgPool2d(2)

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

        self.query_weight = nn.Linear(1024, 1024//factor, bias=False)
        self.key_weight = nn.Linear(1024, 1024//factor, bias=False)

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

        x_4 = getattr(self, 'peer0_0')(x)
        x_4 = getattr(self, 'peer1_0')(x_4)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)
        proj_q = self.query_weight(x_4)
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_4)
        proj_k = proj_k[:, None, :]
        x_4 = getattr(self, 'classifier_0')(x_4)
        pro = x_4.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'peer0_'+str(i))(x)
            temp = getattr(self, 'peer1_'+str(i))(temp)
            temp = self.avgpool(temp)
            temp = temp.view(temp.size(0), -1)
            temp_q = self.query_weight(temp)
            temp_k = self.key_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = temp_k[:, None, :]
            temp = getattr(self, 'classifier_'+str(i))(temp)
            temp = temp.unsqueeze(-1)
            pro = torch.cat([pro, temp], -1)
            proj_q = torch.cat([proj_q, temp_q], 1)
            proj_k = torch.cat([proj_k, temp_k], 1)

        energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0, 2, 1))

        temp = getattr(self, 'peer0_'+str(self.num_branches-1))(x)
        temp = getattr(self, 'peer1_'+str(self.num_branches-1))(temp)
        temp = self.avgpool(temp)
        temp = temp.view(temp.size(0), -1)
        temp_out = getattr(self, 'classifier_'+str(self.num_branches-1))(temp)
        return pro, x_m, temp_out

def mbn_v1(dataset='cifar100', shallow=None, alpha=1.0, **kwargs):
    model = MobileNetV1_Cifar(dataset=dataset, shallow=shallow, alpha=alpha, **kwargs)
    return model

if __name__ == '__main__':
    net = OKDDip_MobileNetV1(dataset='cifar10', num_branches=4, shallow=False, alpha=1.0)
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y, en, stu = net(x)
    print(y.shape, en.shape, stu.shape)

