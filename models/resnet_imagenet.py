import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ONE_ResNet_IMG', 'OKDDip_ResNet_IMG', 'CLILR_ResNet_IMG', 'resnet',
           'resnet18_img', 'resnet34_img', 'resnet50_img']

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
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

class ONE_ResNet_IMG(nn.Module):
    def __init__(self, depth, reduction, num_branches, dataset):
        self.inplanes = 64
        self.num_branches = num_branches
        self.reduction = reduction

        if dataset == 'imagenet':
            num_classes = 1000
        elif dataset == 'cub200':
            num_classes = 200
        else:
            raise ValueError("No valid dataset...")

        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block = Bottleneck

        self.block = block

        super(ONE_ResNet_IMG, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        fix_inplanes = self.inplanes

        ## for coefficient
        self.control_v1 = nn.Linear(fix_inplanes, self.num_branches)
        self.bn_v1 = nn.BatchNorm1d(self.num_branches)
        self.avgpool_c = nn.AvgPool2d(28)
        self.avgpool = nn.AvgPool2d(7, stride=1)
 
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

        x_c = self.avgpool_c(x)
        x_c = x_c.view(x_c.size(0), -1)

        x_c = self.control_v1(x_c)
        x_c = self.bn_v1(x_c)
        x_c = F.relu(x_c)
        x_c = F.softmax(x_c, dim=1)

        x_ = getattr(self, 'layer3_0')(x)
        x_ = getattr(self, 'layer4_0')(x_)
        x_ = self.avgpool(x_)
        x_ = x_.view(x_.size(0), -1)
        x_ = getattr(self, 'fc_0')(x_)
        x_m = x_c[:,0].repeat(x_.size(1), 1).transpose(0, 1) * x_
        pro = x_.unsqueeze(-1)

        for i in range(1, self.num_branches):
            en = getattr(self, 'layer3_'+str(i))(x)
            en = getattr(self, 'layer4_'+str(i))(en)
            en = self.avgpool(en)
            en = en.view(en.size(0), -1)
            en = getattr(self, 'fc_'+str(i))(en)
            x_m += x_c[:,i].repeat(en.size(1), 1).transpose(0, 1) * en
            en = en.unsqueeze(-1)
            pro = torch.cat([pro, en], -1)
        
        return pro, x_m

class OKDDip_ResNet_IMG(nn.Module):
    def __init__(self, depth, reduction, num_branches, dataset):
        self.inplanes = 64
        self.num_branches = num_branches
        self.reduction = reduction

        if dataset == 'imagenet':
            num_classes = 1000
        elif dataset == 'cub200':
            num_classes = 200
        else:
            raise ValueError("No valid dataset...")

        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block = Bottleneck

        self.block = block

        super(OKDDip_ResNet_IMG, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        fix_inplanes = self.inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for i in range(num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layer(block, 256, layers[2], stride=2))
            setattr(self, 'layer4_'+str(i), self._make_layer(block, 512, layers[3], stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'fc_'+str(i), nn.Linear(512*block.expansion, num_classes))

        self.query_weight = nn.Linear(512*block.expansion, 512 // reduction, bias=False)
        self.key_weight = nn.Linear(512*block.expansion, 512 // reduction, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        x_4 = getattr(self, 'layer3_0')(x)
        x_4 = getattr(self, 'layer4_0')(x_4)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)
        proj_q = self.query_weight(x_4)
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_4)
        proj_k = proj_k[:, None, :]
        x_4_0 = getattr(self, 'fc_0')(x_4)
        pro = x_4_0.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'layer3_'+str(i))(x)
            temp = getattr(self, 'layer4_'+str(i))(temp)
            temp = self.avgpool(temp)
            temp = temp.view(temp.size(0), -1)
            temp_q = self.query_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = self.key_weight(temp)
            temp_k = temp_k[:, None, :]
            temp_1 = getattr(self, 'fc_'+str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro, temp_1], -1)
            proj_q = torch.cat([proj_q, temp_q], 1)
            proj_k = torch.cat([proj_k, temp_k], 1)

        energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0, 2, 1))

        temp = getattr(self, 'layer3_'+str(self.num_branches-1))(x)
        temp = getattr(self, 'layer4_'+str(self.num_branches-1))(temp)
        temp = self.avgpool(temp)
        temp = temp.view(temp.size(0), -1)
        temp_out = getattr(self, 'fc_'+str(self.num_branches-1))(temp)
        
        return pro, x_m, temp_out

class CLILR_ResNet_IMG(nn.Module):
    def __init__(self, depth, reduction, num_branches, dataset, bpscale=True):
        self.inplanes = 64
        self.num_branches = num_branches
        self.reduction = reduction
        self.bpscale = bpscale

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

        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block = Bottleneck

        self.block = block

        super(CLILR_ResNet_IMG, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        fix_inplanes = self.inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for i in range(self.num_branches // 2):
            setattr(self, 'layer3_'+str(i), self._make_layer(block, 256, layers[2], stride=2))
            if i == self.num_branches // 2 - 1:
                pass
            else:
                self.inplanes = fix_inplanes

        fix_inplanes = self.inplanes
        for i in range(self.num_branches):
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

        if self.bpscale:
            self.layer_ILR = ILR.apply

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
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches//2) # Backprop rescaling

        x_3_0 = getattr(self, 'layer3_0')(x)
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches//2) # Backprop rescaling

        x_4_0 = getattr(self, 'layer4_0')(x_3_0)
        x_4_0 = self.avgpool(x_4_0)
        x_4_0 = x_4_0.view(x_4_0.size(0), -1)
        x_4_0 = getattr(self, 'fc_0')(x_4_0)
        pro = x_4_0.unsqueeze(-1)

        x_4_1 = getattr(self, 'layer4_1')(x_3_0)
        x_4_1 = self.avgpool(x_4_1)
        x_4_1 = x_4_1.view(x_4_1.size(0), -1)
        x_4_1 = getattr(self, 'fc_1')(x_4_1)
        x_4_1 = x_4_1.unsqueeze(-1)
        pro = torch.cat([pro, x_4_1], -1)

        x_3_1 = getattr(self, 'layer3_1')(x)
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches//2) # Backprop rescaling

        x_4_2 = getattr(self, 'layer4_2')(x_3_1)
        x_4_2 = self.avgpool(x_4_2)
        x_4_2 = x_4_2.view(x_4_2.size(0), -1)
        x_4_2 = getattr(self, 'fc_2')(x_4_2)
        x_4_2 = x_4_2.unsqueeze(-1)
        pro = torch.cat([pro, x_4_2], -1)

        x_4_3 = getattr(self, 'layer4_3')(x_3_1)
        x_4_3 = self.avgpool(x_4_3)
        x_4_3 = x_4_3.view(x_4_3.size(0), -1)
        x_4_3 = getattr(self, 'fc_3')(x_4_3)
        x_4_3 = x_4_3.unsqueeze(-1)
        pro = torch.cat([pro, x_4_3], -1)

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

class resnet(nn.Module):
    def __init__(self, block, layers, dataset):
        self.inplanes = 64
        self.dataset = dataset
        if dataset == 'imagenet':
            num_classes = 1000
        elif dataset == 'cub200':
            num_classes = 200
        elif dataset == 'cars196':
            num_classes = 196
        elif dataset == 'stanford':
            num_classes = 11318
        elif dataset == 'dogs120':
            num_classes = 120
        else:
            raise ValueError("No valid dataset...")

        super(resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if dataset == 'imagenet':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample= None
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.layer4(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) if self.dataset == 'imagenet' else self.fc1(x)

        return x

def resnet18_img(pretrained=False, dataset='imagenet', **kwargs):
    model = resnet(BasicBlock, [2, 2, 2, 2], dataset=dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print('ResNet-18 Use pretrained model for initialization')
    return model

def resnet34_img(pretrained=False, dataset='imagenet', **kwargs):
    model = resnet(BasicBlock, [3, 4, 6, 3], dataset=dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        print('ResNet-34 Use pretrained model for initialization')
    return model

def resnet50_img(pretrained=False, dataset='imagenet', **kwargs):
    model = resnet(Bottleneck, [3, 4, 6, 3], dataset=dataset, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        print('ResNet-50 Use pretrained model for initialization')
    return model

if __name__ == '__main__':
    model = resnet50_img(pretrained=True, dataset='stanford')
    print(model.fc1.weight.size())
    x = torch.FloatTensor(2, 3, 224, 224)
    y = model(x)
    print(y.size())
