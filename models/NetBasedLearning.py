import torch
import torch.nn as nn
from .resnet_cifar import *
from .resnet_imagenet import *
from .wideresnet_cifar import *
from .blocks import *

__all__ = ['DML', 'NetBasedOurs']

class DML(nn.Module):
    def __init__(self, model, dataset, depth, num_branches, wf, bottleneck=False, se=False):
        super(DML, self).__init__()
        self.num_branches = num_branches

        for i in range(self.num_branches):
            if model == 'resnet20':
                setattr(self, 'stu'+str(i), resnet20(dataset=dataset, depth=depth, bottleneck=bottleneck, se=se))
            elif model == 'resnet32':
                setattr(self, 'stu'+str(i), resnet32(dataset=dataset, depth=depth, bottleneck=bottleneck, se=se))
            elif model == 'resnet56':
                setattr(self, 'stu'+str(i), resnet56(dataset=dataset, depth=depth, bottleneck=bottleneck, se=se))
            elif model == 'resnet110':
                setattr(self, 'stu'+str(i), resnet110(dataset=dataset, depth=depth, bottleneck=bottleneck, se=se))
            elif model == 'resnet18':
                setattr(self, 'stu'+str(i), resnet18_img(pretrained=False, dataset=dataset, depth=depth))
            elif model == 'resnet34':
                setattr(self, 'stu'+str(i), resnet34_img(pretrained=False, dataset=dataset, depth=depth))
            elif model == 'resnet50':
                setattr(self, 'stu'+str(i), resnet50_img(pretrained=False, dataset=dataset, depth=depth))
            elif model == 'wrn16_4':
                setattr(self, 'stu'+str(i), wrn16_4(dataset=dataset, depth=depth, widen_factor=wf))

    def forward(self, x):
        out = self.stu0(x)
        out = out.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp_out = getattr(self, 'stu'+str(i))(x)
            temp_out = temp_out.unsqueeze(-1)
            out = torch.cat([out, temp_out], -1)
        return out

## Temporarily Implemeted with ResNet
class NetBasedOurs(nn.Module):
    def __init__(self, dataset, depth, num_branches, models, device, arch='resnet50_img', bottleneck=False, nl=False, embedding=False, se=False, pretrained=False):
        super(NetBasedOurs, self).__init__()
        self.dataset = dataset
        self.num_branches = num_branches
        self.nl = nl
        self.embedding = embedding
        self.device = device

        for i in range(self.num_branches):
            if arch == 'ResNet':
                setattr(self, 'stu'+str(i), models.__dict__[arch](dataset=dataset, depth=depth, bottleneck=bottleneck, se=se))
            elif arch == 'resnet50_img':
                setattr(self, 'stu'+str(i), models.__dict__[arch](pretrained=pretrained, dataset=dataset, depth=depth))
            else:
                raise ValueError('The architecture {} is not implemented'.format(str(model)))
        '''
        # GAP mode
        if dataset in ['cifar10', 'cifar100']:
            self.avgpool8x8 = nn.AvgPool2d(8)
            self.avgpool16x16 = nn.AvgPool2d(16)
        else:
            self.avgpool7x7 = nn.AvgPool2d(7)
            self.avgpool14x14 = nn.AvgPool2d(14)

        # Gatelayer
        gate_node = (self.num_branches-1)*self.stu0.block.expansion*(32+64) \
                if dataset in ['cifar10', 'cifar100'] else (self.num_branches-1)*self.stu0.block.expansion*(256+512)
        self.gatelayer = GateBlock(gate_node, self.num_branches-1, bias=False)
        '''
    def forward(self, x):
        out, f1, f2 = getattr(self, 'stu0')(x)
        assert len(f1.size()) == 4 and len(f2.size()) == 4, 'Need to check out tensor size'
        out = out.unsqueeze(-1)
        '''
        b1, c1, _, _ = f1.size()
        b2, c2, _, _ = f2.size()
        if self.dataset in ['cifar10', 'cifar100']:
            assert c1 == 32*self.stu0.block.expansion and c2 == 64*self.stu0.block.expansion, 'Conflict channel!!'
        else:
            assert c1 == 256*self.stu0.block.expansion and c2 == 512*self.stu0.block.expansion, 'Conflict channel!!'

        # Extract context
        if self.dataset in ['cifar10', 'cifar100']:
            x_c = self.avgpool16x16(f1).view(b1, c1)
            ctx = self.avgpool8x8(f2).view(b2, c2)
        else:
            x_c = self.avgpool14x14(f1).view(b1, c1)
            ctx = self.avgpool7x7(f2).view(b2, c2)
        '''
        for i in range(1, self.num_branches-1):
            tmp_out, tmp_f1, tmp_f2 = getattr(self, 'stu'+str(i))(x)
            tmp_out = tmp_out.unsqueeze(-1)
            out = torch.cat([out, tmp_out], -1)
            '''
            # Extract context
            tmp_x_c = self.avgpool16x16(tmp_f1).view(b1, c1) if self.dataset in ['cifar10', 'cifar100'] else self.avgpool14x14(tmp_f1).view(b1, c1)
            tmp_ctx = self.avgpool8x8(tmp_f2).view(b2, c2) if self.dataset in ['cifar10', 'cifar100'] else self.avgpool7x7(tmp_f2).view(b2, c2)
            x_c = torch.cat([x_c, tmp_x_c], -1)
            ctx = torch.cat([ctx, tmp_ctx], -1)
            '''
        '''
        # Context merging and make coeeficient
        x_c = torch.cat([x_c, ctx], dim=1)
        x_c = self.gatelayer(x_c)
        
        x_en = x_c[:,0].repeat(out[:,:,0].size(1), 1).transpose(0, 1) * out[:,:,0]
        for i in range(1, self.num_branches-1):
            x_en += x_c[:,i].repeat(out[:,:,i].size(1), 1).transpose(0, 1) * out[:,:,i]
        '''
        x_en = torch.mean(out, dim=2)
        x_c = torch.zeros([x.size(0), self.num_branches-1]).cuda(self.device)
        x_c[:] = 1 / (self.num_branches-1)

        # Student
        out_stu, _, _ = getattr(self, 'stu'+str(self.num_branches-1))(x)
        out_stu = out_stu.unsqueeze(-1)
        out = torch.cat([out, out_stu], -1)

        return out, x_en, x_c


if __name__ == '__main__':
    #model = DML(model='resnet18', dataset='imagenet', depth=18, num_branches=4)
    model = NetBasedOurs(dataset='cub200', depth=32, num_branches=4)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    y, en, c = model(x)
    print(y.size())
