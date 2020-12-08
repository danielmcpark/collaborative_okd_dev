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

class SELayer(nn.Module):
    "Squeeze-and-Excitation Layer"
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SALayer(nn.Module):
    "Spatial Attention Layer"
    def __init__(self, channel, reduction=4, dilation_val=4):
        super(SALayer, self).__init__()
        self.sa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=1),
                nn.BatchNorm2d(channel // reduction),
                nn.ReLU(inplace=True),

                nn.Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=dilation_val, dilation=dilation_val),
                nn.BatchNorm2d(channel // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=dilation_val, dilation=dilation_val),
                nn.BatchNorm2d(channel // reduction),
                nn.ReLU(inplace=True),

                nn.Conv2d(channel // reduction, 1, kernel_size=1)
        )

    def forward(self, x):
        y = self.sa(x)
        return y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, 4)

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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
    def __init__(self, dataset, depth, num_branches, reduction, n_dilconv=2, bottleneck=False):
        super(My_ResNet, self).__init__()
        self.inplanes = 16
        self.num_branches = num_branches
        self.reduction = reduction
        self.n_dilconv = n_dilconv

        if bottleneck is True:
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            n = (depth - 2) // 6
            block = BasicBlock
        self.block = block
        self.block2 = SEBasicBlock

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

        self.globalavgpool = nn.AdaptiveAvgPool2d(1)

        for i in range(self.num_branches-1):
            "Channel attention "
            setattr(self, 'squeeze_'+str(i), nn.Linear(64, 64 // self.reduction, bias=False))
            setattr(self, 'bn_sq_'+str(i), nn.BatchNorm1d(64 // self.reduction))
            setattr(self, 'excitation_'+str(i), nn.Linear(64 // self.reduction, 64, bias=False))
            setattr(self, 'bn_ex_'+str(i), nn.BatchNorm1d(64))

            "Spatial attention"
            setattr(self, 'gateconv_'+str(i), nn.Conv2d(64, 64 // self.reduction, kernel_size=1))
            setattr(self, 'gatebn_'+str(i), nn.BatchNorm2d(64 // self.reduction))
            for j in range(n_dilconv):
                setattr(self, 'gatedilconv_'+str(i)+'_'+str(j), nn.Conv2d(64 // self.reduction, 64 // self.reduction, kernel_size=3, \
                        padding=4, dilation=4))
                setattr(self, 'gatedilbn_'+str(i)+'_'+str(j), nn.BatchNorm2d(64 // self.reduction))
            setattr(self, 'gatelastconv_'+str(i), nn.Conv2d(64 // self.reduction, 1, kernel_size=1))

        for i in range(self.num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(self.block, 64, n, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_'+str(i), nn.Linear(64 * block.expansion, num_classes))

        "Pear attention"
        self.theta = nn.Linear(num_classes, num_classes // 4, bias=False)
        self.phi = nn.Linear(num_classes, num_classes // 4, bias=False)

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

    def _make_mobile_block(self, block, planes):
        mobile_config = [planes, 64, 1] # i, o, s
        layers = []
        layers.append(block(inp=mobile_config[0], oup=mobile_config[1], stride=mobile_config[2]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x) # B x 32 x 16 x 16

        x = self.layer_ILR(x, self.num_branches)
        '''
        ## Peers
        x_3 = getattr(self, 'layer3_0')(x) # B x 64 x 8 x 8
        x_3 = self.avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier3_0')(x_3)
        ind = x_3.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'layer3_'+str(i))(x) # B x 64 x 8 x 8
            temp = self.avgpool(temp)
            temp = temp.view(temp.size(0), -1)
            temp = getattr(self, 'classifier3_'+str(i))(temp)
            temp = temp.unsqueeze(-1)
            ind = torch.cat([ind, temp], -1)

        x_en = torch.mean(ind, dim=2)
        '''
        globals()['x_3_0'] = getattr(self, 'layer3_0')(x) # B x 64 x 8 x 8

        ## Channel attention
        b, c, _, _ = globals()['x_3_0'].size()
        scores = self.globalavgpool(globals()['x_3_0']).view(b, c) # B x 64
        scores = F.relu(getattr(self, 'bn_sq_0')((getattr(self, 'squeeze_0')(scores))))
        scores = F.sigmoid(getattr(self, 'bn_ex_0')(getattr(self, 'excitation_0')(scores)))
        #scores = getattr(self, 'bn_ex_0')(getattr(self, 'excitation_0')(scores))
        #scores = scores.view(b, c, 1, 1).expand_as(globals()['x_3_0']) ## channel attention

        ## Spatial attention
        #s_spatial = getattr(self, 'gateconv_0')(globals()['x_3_0'])
        #s_spatial = F.relu(getattr(self, 'gatebn_0')(s_spatial))
        #for i in range(self.n_dilconv):
        #    s_spatial = getattr(self, 'gatedilconv_0_'+str(i))(s_spatial)
        #    s_spatial = F.relu(getattr(self, 'gatedilbn_0_'+str(i))(s_spatial))
        #s_spatial = getattr(self, 'gatelastconv_'+str(i))(s_spatial)
        #s_spatial = s_spatial.expand_as(globals()['x_3_0'])

        #att = 1 + F.sigmoid(scores * s_spatial)

        globals()['x_3_0'] = globals()['x_3_0'] * scores.view(b, c, 1, 1).expand_as(globals()['x_3_0']) ## Only channel attention
        #globals()['x_3_0'] = globals()['x_3_0'] * att
        globals()['x_3_0'] = self.avgpool(globals()['x_3_0'])
        globals()['x_3_0'] = globals()['x_3_0'].view(globals()['x_3_0'].size(0), -1)
        globals()['x_3_0'] = getattr(self, 'classifier3_0')(globals()['x_3_0'])
        ind = globals()['x_3_0'].unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            globals()['x_3_{}'.format(i)] = getattr(self, 'layer3_'+str(i))(x) # B x 64 x 8 x 8

            ## Channel attention
            temp_s = self.globalavgpool(globals()['x_3_{}'.format(i)]).view(b, c) # B x 64
            #scores = torch.cat([scores, temp_s], -1) # B x (64 x self.num_branches-1)
            temp_s = F.relu(getattr(self, 'bn_sq_'+str(i))(getattr(self, 'squeeze_'+str(i))(temp_s)))
            temp_s = F.sigmoid(getattr(self, 'bn_ex_'+str(i))(getattr(self, 'excitation_'+str(i))(temp_s)))
            #temp_s = getattr(self, 'bn_ex_'+str(i))(getattr(self, 'excitation_'+str(i))(temp_s))
            #temp_s = temp_s.view(b, c, 1, 1).expand_as(globals()['x_3_{}'.format(i)])

            ## Spatial attention
            #temp_spa = getattr(self, 'gateconv_{}'.format(i))(globals()['x_3_{}'.format(i)])
            #temp_spa = F.relu(getattr(self, 'gatebn_{}'.format(i))(temp_spa))
            #for j in range(self.n_dilconv):
            #    temp_spa = getattr(self, 'gatedilconv_{}_{}'.format(i, j))(temp_spa)
            #    temp_spa = F.relu(getattr(self, 'gatedilbn_{}_{}'.format(i, j))(temp_spa))
            #temp_spa = getattr(self, 'gatelastconv_{}'.format(i))(temp_spa)
            #temp_spa = temp_spa.expand_as(globals()['x_3_{}'.format(i)])

            #temp_att = 1 + F.sigmoid(temp_s * temp_spa)

            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)] * temp_s.view(b, c, 1, 1).expand_as(globals()['x_3_{}'.format(i)]) ## Only channel attention
            #globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)] * temp_att
            globals()['x_3_{}'.format(i)] = self.avgpool(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)].view(globals()['x_3_{}'.format(i)].size(0), -1)
            globals()['x_3_{}'.format(i)] = getattr(self, 'classifier3_'+str(i))(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['x_3_{}'.format(i)]], -1) # B x categories x num_branches-1

        '''
        scores = F.relu(self.squeeze(scores))
        scores = F.sigmoid(self.excitation(scores)) # B x (64 x self.num_branches-1)
        scores = scores.view(b, c, -1) # B x 64 x self.num_branches-1
        '''
        '''
        globals()['x_3_0'] = globals()['x_3_0'] * scores[:,:,0].view(b, c, 1, 1).expand_as(globals()['x_3_0'])
        globals()['x_3_0'] = self.avgpool(globals()['x_3_0'])
        globals()['x_3_0'] = globals()['x_3_0'].view(globals()['x_3_0'].size(0), -1)
        globals()['x_3_0'] = getattr(self, 'classifier3_0')(globals()['x_3_0'])
        ind = globals()['x_3_0'].unsqueeze(-1)
        
        for i in range(1, self.num_branches-1):
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)] * scores[:,:,i].view(b, c, 1, 1).expand_as(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = self.avgpool(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)].view(globals()['x_3_{}'.format(i)].size(0), -1)
            globals()['x_3_{}'.format(i)] = getattr(self, 'classifier3_'+str(i))(globals()['x_3_{}'.format(i)])
            globals()['x_3_{}'.format(i)] = globals()['x_3_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['x_3_{}'.format(i)]], -1)
        '''
        ## Peer attention
        x_en = torch.mean(ind, dim=2)
        '''
        t_ind = ind[:,:,0].view(ind[:,:,0].size(0), -1)
        #theta = self.theta(t_ind) # (B, embedding)
        #phi = self.phi(t_ind) # (B, embedding)
        theta = t_ind
        phi = t_ind
        theta = theta[:, None, :] # (B, 1, embedding)
        phi = phi[:, None, :] # (B, 1, embedding)

        for i in range(1, self.num_branches-1):
            t_ind = ind[:,:,i].view(ind[:,:,i].size(0), -1)
            #temp_theta = self.theta(t_ind)
            #temp_phi = self.phi(t_ind)
            temp_theta = t_ind
            temp_phi = t_ind
            temp_theta = temp_theta[:, None, :]
            temp_phi = temp_phi[:, None, :]
            theta = torch.cat([theta, temp_theta], 1) # (B, num_branches-1, embedding)
            phi = torch.cat([phi, temp_phi], 1) # (B, num_branches-1, embedding)

        similarity = torch.bmm(theta, phi.permute(0, 2, 1)) # (B, num_branches-1, num_branches-1)
        att = F.softmax(similarity, dim=-1)
        scale = torch.bmm(ind, att.permute(0, 2, 1)) # (B, categories, num_branches-1)
        x_en = torch.mean((ind + scale), dim=2)

        #x_en = torch.bmm(ind, att.permute(0, 2, 1)).sum(-1) / (self.num_branches-1) # (B, categories, num_branches-1)
        '''
        ## Student
        x_s = getattr(self, 'layer3_'+str(self.num_branches-1))(x) # B x 64 x 8 x 8
        x_s = self.avgpool(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'classifier3_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1) # B x categories x branches

        return ind, x_en

if __name__ == '__main__':
    model = My_ResNet('cifar100', 20, 4, 16)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    y, x_m = model(x)
    print(y.size(), x_m.size())
