import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .blocks import *

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

class ResNet_IMG(nn.Module):
    def __init__(self, block, layers, num_branches, dataset, NL=False, embedding=False):
        self.inplanes = 64
        self.num_branches = num_branches
        self.NL = NL
        self.embedding = embedding

        if dataset == 'imagenet':
            num_classes = 1000
        elif dataset == 'cub200':
            num_classes = 200
        elif dataset == 'cars196':
            num_classes = 196
        elif dataset == 'dogs120':
            num_classes = 120
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
        self.avgpool = nn.AvgPool2d(7, stride=1)

        ## for coefficient
        self.avgpool14x14 = nn.AvgPool2d(14)
        self.gatelayer = GateBlock(fix_inplanes+(self.num_branches-1)*(256+512)*block.expansion, self.num_branches-1)

        if self.NL:
            self.phi = nn.Conv2d(in_channels=128*block.expansion, out_channels=1, kernel_size=1)
            if self.embedding:
                self.theta = nn.Conv2d(in_channels=128*block.expansion, out_channels=128*block.expansion, kernel_size=1)

            for i in range(self.num_branches-1):
                setattr(self, 'phi3_'+str(i), nn.Conv2d(in_channels=256*block.expansion, out_channels=1, kernel_size=1))
                setattr(self, 'phi4_'+str(i), nn.Conv2d(in_channels=512*block.expansion, out_channels=1, kernel_size=1))
                if self.embedding:
                    setattr(self, 'theta3_'+str(i), nn.Conv2d(in_channels=256*block.expansion,
                                                              out_channels=256*block.expansion, kernel_size=1))
                    setattr(self, 'theta4_'+str(i), nn.Conv2d(in_channels=512*block.expansion,
                                                              out_channels=512*block.expansion, kernel_size=1))

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
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

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

        globals()['x_3_0'] = getattr(self, 'layer3_0')(x) # B x 256 x 14 x 14
        b, c, _, _ = globals()['x_3_0'].size()
        if self.NL:
            phi = getattr(self, 'phi3_0')(globals()['x_3_0'])
            phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
            phi = F.softmax(phi, dim=-1).unsqueeze(-1)
            if self.embedding:
                theta = getattr(self, 'theta3_0')(globals()['x_3_0'])
                theta = theta.view(theta.size(0), theta.size(1), -1)
            else:
                theta = globals()['x_3_0'].view(globals()['x_3_0'].size(0), globals()['x_3_0'].size(1), -1)
            context = torch.bmm(theta, phi).squeeze(-1)
        else:
            context = self.avgpool14x14(globals()['x_3_0']).view(b, c) # B x 256

        for i in range(1, self.num_branches-1):
            globals()['x_3_{}'.format(i)] = getattr(self, 'layer3_'+str(i))(x) # B x 256 x 14 x 14
            if self.NL:
                phi = getattr(self, 'phi3_'+str(i))(globals()['x_3_{}'.format(i)])
                phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
                phi = F.softmax(phi, dim=-1).unsqueeze(-1)
                if self.embedding:
                    theta = getattr(self, 'theta3_'+str(i))(globals()['x_3_{}'.format(i)])
                    theta = theta.view(theta.size(0), theta.size(1), -1)
                else:
                    theta = globals()['x_3_{}'.format(i)].view(globals()['x_3_{}'.format(i)].size(0), globals()['x_3_{}'.format(i)].size(1), -1)
                temp_c = torch.bmm(theta, phi).squeeze(-1)
            else:
                temp_c = self.avgpool14x14(globals()['x_3_{}'.format(i)]).view(b, c) # B x 256
            context = torch.cat([context, temp_c], dim=1)

        globals()['x_4_0'] = getattr(self, 'layer4_0')(globals()['x_3_0']) # B x 512 x 7 x 7

        b, c, _, _ = globals()['x_4_0'].size()
        if self.NL:
            phi = getattr(self, 'phi4_0')(globals()['x_4_0'])
            phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
            phi = F.softmax(phi, dim=-1).unsqueeze(-1)
            if self.embedding:
                theta = getattr(self, 'theta4_0')(globals()['x_4_0'])
                theta = theta.view(theta.size(0), theta.size(1), -1)
            else:
                theta = globals()['x_4_0'].view(globals()['x_4_0'].size(0), globals()['x_4_0'].size(1), -1)
            scores = torch.bmm(theta, phi).squeeze(-1)
        else:
            scores = self.avgpool(globals()['x_4_0']).view(b, c) # B x 512
        context = torch.cat([context, scores], dim=1)

        globals()['x_4_0'] = self.avgpool(globals()['x_4_0']).view(globals()['x_4_0'].size(0), -1)
        globals()['x_4_0'] = getattr(self, 'fc_0')(globals()['x_4_0'])
        ind = globals()['x_4_0'].unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            globals()['x_4_{}'.format(i)] = getattr(self, 'layer4_'+str(i))(globals()['x_3_{}'.format(i)]) # B x 512 x 7 x 7
            ## Channel attention
            if self.NL:
                phi = getattr(self, 'phi4_'+str(i))(globals()['x_4_{}'.format(i)])
                phi = phi.permute(0, 2, 3, 1).squeeze(-1).view(phi.size(0), -1)
                phi = F.softmax(phi, dim=-1).unsqueeze(-1)
                if self.embedding:
                    theta = getattr(self, 'theta4_'+str(i))(globals()['x_4_{}'.format(i)])
                    theta = theta.view(theta.size(0), theta.size(1), -1)
                else:
                    theta = globals()['x_4_{}'.format(i)].view(globals()['x_4_{}'.format(i)].size(0), globals()['x_4_{}'.format(i)].size(1), -1)
                temp_s = torch.bmm(theta, phi).squeeze(-1)
            else:
                temp_s = self.avgpool(globals()['x_4_{}'.format(i)]).view(b, c) # B x 512
            context = torch.cat([context, temp_s], dim=1)

            globals()['x_4_{}'.format(i)] = self.avgpool(globals()['x_4_{}'.format(i)]).view(globals()['x_4_{}'.format(i)].size(0), -1)
            globals()['x_4_{}'.format(i)] = getattr(self, 'fc_'+str(i))(globals()['x_4_{}'.format(i)])
            globals()['x_4_{}'.format(i)] = globals()['x_4_{}'.format(i)].unsqueeze(-1)
            ind = torch.cat([ind, globals()['x_4_{}'.format(i)]], -1) # B x categories x num_branches-1

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
            x_c = self.avgpool28x28(x).view(b1, c1)
        x_c = torch.cat([x_c, context], dim=1)
        x_c = self.gatelayer(x_c)
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

        return ind, x_en, x_c

def My_ResNet18(num_branches, NL=False, embedding=False, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(BasicBlock, [2, 2, 2, 2], num_branches, dataset, NL, embedding, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    print('ResNet-18 Use pretrained model for initialization')
    return model

def My_ResNet34(num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(BasicBlock, [3, 4, 6, 3], num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    print('ResNet-34 Use pretrained model for initialization')
    return model

def My_ResNet50(num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(Bottleneck, [3, 4, 6, 3], num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    print('ResNet-50 Use pretrained model for initialization')
    return model

def My_ResNet101(num_branches, pretrained=False, dataset='imagenet', **kwargs):
    model = ResNet_IMG(Bottleneck, [3, 4, 23, 3], num_branches, dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    print('ResNet-101 Use pretrained model for initialization')
    return model


if __name__ == '__main__':
    model = My_ResNet50(pretrained=False, num_branches=4, dataset='cub200', NL=False, embedding=False)
    print(model)
    x = torch.FloatTensor(2, 3, 224, 224)
    y, en, c = model(x)
    print(y.shape, en.shape, c.shape)
