import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

__all__ = ['DenseNet', 'DenseNet_OKDDip', 'DenseNet_ONEILR', 'densenet40k12']

class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_feature)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
            )
            self.add_module('denselayer%d' %(i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, dataset, growth_rate=12, block_config=[16, 16, 16], compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 small_inputs=True, efficient=False, KD=False):
        super(DenseNet, self).__init__()
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("Dataset conflict!!..")

        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.KD = KD

        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' %(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        # D40K12  B x 132 x 8 x 8
        # D100K12 B x 342 x 8 x 8
        # D100K40 B x 1126 x 8 x 8
        x = F.relu(features, inplace=True)
        x_f = F.avg_pool2d(x, kernel_size=self.avgpool_size).view(features.size(0), -1) # B x 132
        x = self.classifier(x_f)
        if self.KD == True:
            return x_f, x
        else:
            return x

class DenseNet_OKDDip(nn.Module):
    def __init__(self, dataset, growth_rate=12, block_config=(16, 16, 16), num_branches=4, input_channels=132,
                 factor=8, compression=0.5, num_init_features=24, bn_size=4, drop_rate=0,
                 small_inputs=True, efficient=False):
        super(DenseNet_OKDDip, self).__init__()
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("Unvalid dataset!!..")
        assert 0 < compression <=1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.num_branches = num_branches
        
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, pading=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i != len(block_config) - 1:
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' %(i + 1), block)
                num_features = num_features + num_layers * growth_rate

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' %(i + 1), trans)
                num_features = int(num_features * compression)
            else:
                block = _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        efficient=efficient,
                )
                for i in range(self.num_branches):
                    setattr(self, 'Branch' + str(i), block)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        num_features = num_features + num_layers * growth_rate
        for i in range(self.num_branches):
            setattr(self, 'norm_final_' + str(i), nn.BatchNorm2d(num_features))
            setattr(self, 'relu_final_' + str(i), nn.ReLU(inplace=True))

        # Linear layer
        for i in range(self.num_branches):
            setattr(self, 'classifier3_' + str(i), nn.Linear(num_features, num_classes))

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.query_weight = nn.Linear(input_channels, input_channels//factor, bias=False)
        self.key_weight = nn.Linear(input_channels, input_channels//factor, bias=False)

    def forward(self, x):
        x = self.features(x) # B x 60 x 8 x 8
        
        x_3 = getattr(self, 'Branch0')(x) # B x 132 x 8 x 8
        x_3 = getattr(self, 'norm_final_0')(x_3)
        x_3 = getattr(self, 'relu_final_0')(x_3)
        x_3 = self.avgpool(x_3).view(x_3.size(0), -1) # B x 132
        proj_q = self.query_weight(x_3) # B x 8
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)   # B x 8
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        pro = x_3_1.unsqueeze(-1)
        for i in range(1, self.num_branches-1):
            temp = getattr(self, 'Branch' + str(i))(x)
            temp = getattr(self, 'norm_final_'+str(i))(temp)
            temp = getattr(self, 'relu_final_'+str(i))(temp)
            temp = self.avgpool(temp).view(temp.size(0), -1) # B x 132
            temp_q = self.query_weight(temp)
            temp_k = self.key_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = temp_k[:, None, :]
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro, temp_1], -1)
            proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
            proj_k = torch.cat([proj_k, temp_k], 1)

        energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        x_m = torch.bmm(pro, attention.permute(0, 2, 1))

        temp = getattr(self, 'Branch'+str(self.num_branches - 1))(x)
        temp = self.avgpool(temp) # B x 64 x 1 x 1
        temp = temp.view(temp.size(0), -1)
        temp_out = getattr(self, 'classifier3_' + str(self.num_branches - 1))(temp)
        return pro, x_m, temp_out

class DenseNet_ONEILR(nn.Module):
    def __init__(self, dataset, growth_rate=12, block_config=(16, 16, 16), num_branches=4, bpscale=False,
                 compression=0.5, num_init_features=24, bn_size=4, drop_rate=0,
                 small_inputs=True, efficient=False, ind=False):
        super(DenseNet_ONEILR, self).__init__()
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("Unvalid dataset !!...")

        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.ind = ind
        self.bpscale = bpscale
        self.num_branches = num_branches

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i != len(block_config) - 1:
                block = _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:
                block = _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        efficient=efficient,
                )
                for i in range(self.num_branches):
                    setattr(self, 'layer3_' + str(i), block)

        if self.bpscale == False:
            self.avgpool_c = nn.AvgPool2d(8)
            self.control_v1 = nn.Linear(num_features, self.num_branches)
            self.bn_v1 = nn.BatchNorm1d(self.num_branches)

        num_features = num_features + num_layers * growth_rate
        for i in range(self.num_branches):
            setattr(self, 'norm_final_' + str(i), nn.BatchNorm2d(num_features))
            setattr(self, 'relu_final_' + str(i), nn.ReLU(inplace = True))

        # Linear layer
        for i in range(self.num_branches):
            setattr(self, 'classifier3_' + str(i), nn.Linear(num_features, num_classes))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        if self.bpscale:
            self.layer_ILR = ILR.apply

    def forward(self, x):
        x = self.features(x) # B x 60 x 8 x 8
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches)

        x_3 = getattr(self, 'layer3_0')(x)  # B x 132 x 8 x 8
        x_3 = getattr(self, 'norm_final_0')(x_3)
        x_3 = getattr(self, 'relu_final_0')(x_3)
        x_3 = self.avgpool(x_3).view(x_3.size(0), -1)   # B x 132
        x_3_1 = getattr(self, 'classifier3_0')(x_3) # B x num_classes
        pro = x_3_1.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            temp = getattr(self, 'norm_final_' + str(i))(temp)
            temp = getattr(self, 'relu_final_' + str(i))(temp)
            temp = self.avgpool(temp).view(temp.size(0), -1)
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro, temp_1], -1)

        if self.ind:
            return pro, None
        else:
            if self.bpscale:
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
            else:
                x_c = self.avgpool_c(x)
                x_c = x_c.view(x_c.size(0), -1)
                print(x_c.size())
                x_c = self.control_v1(x_c)
                x_c = self.bn_v1(x_c)
                x_c = F.relu(x_c)
                x_c = F.softmax(x_c, dim=1)
                x_m = x_c[:, 0].view(-1, 1).repeat(1, pro[:,:,0].size(1)) * pro[:,:,0]
                for i in range(1, self.num_branches):
                    x_m += x_c[:,i].view(-1, 1).repeat(1, pro[:,:,i].size(1)) * pro[:,:,i]
            return pro, x_m

def densenet40k12(dataset):
    model = DenseNet(dataset=dataset, growth_rate=12, block_config=[6, 6, 6])
    return model

if __name__=='__main__':
    model=DenseNet_ONEILR(dataset='cifar10', growth_rate=12, block_config = [6, 6, 6], bpscale=False)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    _, stu= model(x)
    print(stu.size())
