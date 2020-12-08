import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VGG', 'OKDDip_VGG', 'ONE_VGG', 'CLILR_VGG', 'vgg16']

class ILR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

class VGG(nn.Module):
    def __init__(self, dataset, depth):
        super(VGG, self).__init__()
        self.inplanes = 64
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
        self.layer4 = self._make_layers(512, num_layer)
        self.classifier =nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplanes, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplanes=input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

class OKDDip_VGG(nn.Module):
    def __init__(self, dataset, depth, num_branches, factor=8):
        super(OKDDip_VGG, self).__init__()
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

        self.query_weight = nn.Linear(512, 512//factor, bias=False)
        self.key_weight = nn.Linear(512, 512//factor, bias=False)

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
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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
        x = self.layer2(x)
        x = self.layer3(x)

        x_3 = getattr(self, 'layer4_0')(x)
        x_3 = x_3.view(x_3.size(0), -1)
        proj_q = self.query_weight(x_3)
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)
        proj_k = proj_k[:, None, :]
        x_3 = getattr(self, 'classifier_0')(x_3)
        pro = x_3.unsqueeze(-1)

        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'layer4_'+str(i))(x)
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

        energy = torch.bmm(proj_q, proj_k.permute(0,2,1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0,2,1))

        temp = getattr(self, 'layer4_'+str(self.num_branches-1))(x)
        temp = temp.view(temp.size(0), -1)
        temp = getattr(self, 'classifier_'+str(self.num_branches-1))(temp)
        return pro, x_m, temp

class ONE_VGG(nn.Module):
    def __init__(self, dataset, depth, num_branches):
        super(ONE_VGG, self).__init__()
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

        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.control_v1 = nn.Linear(self.inplanes, self.num_branches)
        self.bn_v1 = nn.BatchNorm1d(self.num_branches)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplanes, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplanes=input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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
        x = self.layer2(x)
        x = self.layer3(x)

        x_c = self.avgpool_c(x)
        x_c = x_c.view(x_c.size(0), -1)
        x_c = self.control_v1(x_c)
        x_c = self.bn_v1(x_c)
        x_c = F.relu(x_c)
        x_c = F.softmax(x_c, dim=1)

        x_3 = getattr(self, 'layer4_0')(x)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier_0')(x_3)
        x_m = x_c[:,0].view(-1, 1).repeat(1, x_3.size(1)) * x_3
        pro = x_3.unsqueeze(-1)
        
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer4_'+str(i))(x)
            temp = temp.view(temp.size(0), -1)
            temp = getattr(self, 'classifier_'+str(i))(temp)
            x_m += x_c[:,i].view(-1, 1).repeat(1, temp.size(1)) * temp
            temp = temp.unsqueeze(-1)
            pro = torch.cat([pro, temp], -1)

        return pro, x_m

class CLILR_VGG(nn.Module):
    def __init__(self, dataset, depth, num_branches, bpscale=True):
        super(CLILR_VGG, self).__init__()
        self.inplanes = 64
        self.num_branches = num_branches
        self.bpscale = bpscale

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        if self.bpscale:
            self.layer_ILR = ILR.apply

    def _make_layers(self, input, num_layer):
        layers = []
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplanes, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplanes=input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches)

        x_3 = getattr(self, 'layer4_0')(x)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = getattr(self, 'classifier_0')(x_3)
        pro = x_3.unsqueeze(-1)
        
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer4_'+str(i))(x)
            temp = temp.view(temp.size(0), -1)
            temp = getattr(self, 'classifier_'+str(i))(temp)
            temp = temp.unsqueeze(-1)
            pro = torch.cat([pro, temp], -1)

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

def vgg16(dataset, depth, *kwargs):
    model = VGG(dataset=dataset, depth=depth)
    return model

if __name__=='__main__':
    model = CLILR_VGG(dataset='cifar100', depth=16, num_branches=4)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    _, y = model(x)
    print(y.size())


