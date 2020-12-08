import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet_IMG', 'My_ResNet18', 'My_ResNet34', 'My_ResNet50', 'My_ResNet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #x = F.relu(x)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #x = F.relu(x)
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
    def backward(ctx, gard_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

class ResNet_IMG(nn.Module):
    def __init__(self, block, layers, reduction, num_branches, dataset):
        self.inplanes = 64
        self.num_branches = num_branches
        self.reduction = reduction

        if dataset == 'imagenet':
            num_classes = 1000
        elif dataset == 'cub200':
            num_classes = 200
        else:
            raise ValueError("No valid dataset...")

        super(ResNet_IMG, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        fix_inplanes = self.inplanes
        self.avgpool28x28 = nn.AvgPool2d(28)

        ## for coefficient
        self.avgpool14x14 = nn.AvgPool2d(14)
        self.gatelinear = nn.Linear(fix_inplanes + (self.num_branches-1)*(256+512)*block.expansion, self.num_branches-1)
        self.bn_gate = nn.BatchNorm1d(self.num_branches-1)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        for i in range(num_branches-1):
            "Channel attention "
            setattr(self, 'squeeze_'+str(i), nn.Linear(512*block.expansion, 512*block.expansion // self.reduction, bias=False))
            setattr(self, 'bn_sq_'+str(i), nn.BatchNorm1d(512*block.expansion // self.reduction))
            setattr(self, 'excitation_'+str(i), nn.Linear(512*block.expansion // self.reduction, 512*block.expansion, bias=False))
            setattr(self, 'bn_ex_'+str(i), nn.BatchNorm1d(512*block.expansion))
        
        for i in range(num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(block, 256, layers[2], stride=2))
            setattr(self, 'layer4_'+str(i), self._make_layer(block, 512, layers[3], stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'fc_'+str(i), nn.Linear(512*block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #self.layer_ILR = ILR.apply

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
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
        x = self.maxpool(x)

        x = self.layer1(x) # B x 64 x 56 x 56
        x = self.layer2(x) # B x 128 x 28 x 28

        #x = self.layer_ILR(x, self.num_branches)

        globals()['x_3_0'] = getattr(self, 'layer3_0')(x) # B x 256 x 14 x 14
        b, c, _, _ = globals()['x_3_0'].size()
        context = self.avgpool14x14(globals()['x_3_0']).view(b, c) # B x 256

        for i in range(1, self.num_branches-1):
            globals()['x_3_{}'.format(i)] = getattr(self, 'layer3_'+str(i))(x) # B x 256 x 14 x 14
            temp_c = self.avgpool14x14(globals()['x_3_{}'.format(i)]).view(b, c) # B x 256
            context = torch.cat([context, temp_c], dim=1)

        globals()['x_4_0'] = getattr(self, 'layer4_0')(globals()['x_3_0']) # B x 512 x 7 x 7

        b, c, _, _ = globals()['x_4_0'].size()
        scores = self.avgpool(globals()['x_4_0']).view(b, c) # B x 512
        context = torch.cat([context, scores], dim=1)
        scores = F.relu(getattr(self, 'bn_sq_0')(getattr(self, 'squeeze_0')(scores)))
        scores = F.sigmoid(getattr(self, 'bn_ex_0')(getattr(self, 'excitation_0')(scores)))

        globals()['x_4_0'] = globals()['x_4_0'] * scores.view(b, c, 1, 1).expand_as(globals()['x_4_0'])
        globals()['x_4_0'] = self.avgpool(globals()['x_4_0']).view(globals()['x_4_0'].size(0), -1)
        globals()['x_4_0'] = getattr(self, 'fc_0')(globals()['x_4_0'])
        ind = globals()['x_4_0'].unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            globals()['x_4_{}'.format(i)] = getattr(self, 'layer4_'+str(i))(globals()['x_3_{}'.format(i)]) # B x 512 x 7 x 7
            ## Channel attention
            temp_s = self.avgpool(globals()['x_4_{}'.format(i)]).view(b, c) # B x 512
            context = torch.cat([context, temp_s], dim=1)
            temp_s = F.relu(getattr(self, 'bn_sq_'+str(i))(getattr(self, 'squeeze_'+str(i))(temp_s)))
            temp_s = F.sigmoid(getattr(self, 'bn_ex_'+str(i))(getattr(self, 'excitation_'+str(i))(temp_s)))

            globals()['x_4_{}'.format(i)] = globals()['x_4_{}'.format(i)] * temp_s.view(b, c, 1, 1).expand_as(globals()['x_4_{}'.format(i)])
            globals()['x_4_{}'.format(i)] = self.avgpool(globals()['x_4_{}'.format(i)]).view(globals()['x_4_{}'.format(i)].size(0), -1)
            globals()['x_4_{}'.format(i)] = getattr(self, 'fc_'+str(i))(globals()['x_4_{}'.format(i)])
            globals()['x_4_{}'.format(i)] = globals()['x_4_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['x_4_{}'.format(i)]], -1) # B x categories x num_branches-1

        ## Peer attention
        #x_en = torch.mean(ind, dim=2)
        b1, c1, _, _ = x.size()
        x_c = self.avgpool28x28(x).view(b1, c1)
        x_c = torch.cat([x_c, context], dim=1)
        x_c = self.gatelinear(x_c)
        x_c = self.bn_gate(x_c)
        x_c = F.relu(x_c)
        x_c = F.softmax(x_c, dim=1)
        x_en = x_c[:,0].repeat(ind[:,:,0].size(1), 1).transpose(0, 1) * ind[:,:,0]
        for i in range(1, self.num_branches-1):
            x_en += x_c[:,i].repeat(ind[:,:,i].size(1), 1).transpose(0, 1) * ind[:,:,i]
        
        ## Student
        x_s = getattr(self, 'layer3_'+str(self.num_branches-1))(x) # B x 256 x 14 x 14
        x_s = getattr(self, 'layer4_'+str(self.num_branches-1))(x_s) # B x 512 x 7 x 7
        x_s = self.avgpool(x_s)
        x_s = x_s.view(x_s.size(0), -1)
        x_s = getattr(self, 'fc_'+str(self.num_branches-1))(x_s)
        x_s = x_s.unsqueeze(-1)
        ind = torch.cat([ind, x_s], -1) # B x categories x branches

        return ind, x_en

def My_ResNet18(reduction, num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(BasicBlock, [2, 2, 2, 2], reduction, num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    print('ResNet-18 Use pretrained model for initialization')
    return model

def My_ResNet34(reduction, num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(BasicBlock, [3, 4, 6, 3], reduction, num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    print('ResNet-34 Use pretrained model for initialization')
    return model

def My_ResNet50(reduction, num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(Bottleneck, [3, 4, 6, 3], reduction, num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    print('ResNet-50 Use pretrained model for initialization')
    return model

def My_ResNet101(reduction, num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(Bottleneck, [3, 4, 23, 3], reduction, num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    print('ResNet-101 Use pretrained model for initialization')
    return model


if __name__ == '__main__':
    model = My_ResNet18(pretrained=False, reduction=4, num_branches=4, dataset='imagenet')
    print(model)
    x = torch.FloatTensor(2, 3, 224, 224)
    y, en = model(x)
    print(y.shape, en.shape)
