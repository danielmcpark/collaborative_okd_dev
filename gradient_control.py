from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, sys
import argparse
import shutil
import time
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from utils import Bar, AverageMeter, RunningAverage, ramps
from loss import KLLoss, aJSLoss, proposed_loss
from thop import profile

import models
import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
STIFLE = 2.0
WARM_UP = True

# Parser settings
parser = argparse.ArgumentParser(description='MCPARK AAAI Framework')
parser.add_argument('--dataset', type=str, default='cifar100', metavar='string',
                    help='insert training dataset (cifar10, cifar100, imagenet, tiny-imagenet, cub200)')
parser.add_argument('--batch-size', type=int, default=128, metavar='int',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='int',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=300, metavar='int',
                    help='# of epochs to train (default: 160)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='int',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='float',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='float',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='float',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 225],
                    help='lr annealing scheule, default=[150, 225]')

parser.add_argument('--arch', type=str, default='My_ResNetV3', metavar='string',
                    help='model to use (look up for a model folder)')
parser.add_argument('--nb_arch', type=str, default='ResNet', metavar='string',
                    help='model to use (look up for a model folder)')

parser.add_argument('--depth', type=int, default=20, metavar='int',
                    help='resnet depth')
parser.add_argument('--wf', type=int, default=1, metavar='int',
                    help='wide resnet channel variation factor')
parser.add_argument('--bottleneck', action='store_true', default=False,
                    help='resnet bottleneck usage')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ngpu', type=str, default='cuda:0', metavar='string',
                    help='CUDA device 0, 1, 2, 3')
parser.add_argument('--seed', type=int, default=1, metavar='int',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='int',
                    help='batch interval for displaying training log')
parser.add_argument('--resume', type=str, default='/mnt/disk3/logs/newkd/resnet/rm32_201124_pcsc_p8_t1.5_g0.0_corr0.0_sig_gs0_cifar100', metavar='string',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', type=str, default='', metavar='string',
                    help='path to dataset')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='float', help='consistency_rampup ratio')
parser.add_argument('--tracefile_pearson', type=str, default='', metavar='string',
                    help='path to save spearman tracefile')
parser.add_argument('--tracefile_spearman', type=str, default='', metavar='string',
                    help='path to save pearson tracefile')
parser.add_argument('--tracefile_diversity', type=str, default='', metavar='string',
                    help='path to save diversity tracefile')
parser.add_argument('--gate_coeff', type=str, default='', metavar='string',
                    help='path to save pearson tracefile')
parser.add_argument('--flops', action='store_true', default=False,
                    help='flag of computing flops')

parser.add_argument('--mine', action='store_true', default=False,
                    help='flag of running my network')
parser.add_argument('--gamma', type=float, default=0.4, metavar='float',
                    help='diversity and cooperation balancing params (default: 0.5)')
parser.add_argument('--slope', type=float, default=1, metavar='float',
                    help='magnitude of slope (default: 1)')
parser.add_argument('--fd', action='store_true', default=False,
                    help='flag of applying forced diversity')
parser.add_argument('--ajs', action='store_true', default=False,
                    help='flag of applying alpha Jensen-shannon divergence')
parser.add_argument('--coo_type', type=str, default='JSD', metavar='string',
                    help='choose one type of cooperation loss in [JSD, JD, wJSD]')
parser.add_argument('--sim', action='store_true', default=False,
                    help='flag of applying batch similarity loss')
parser.add_argument('--nl', action='store_true', default=False,
                    help='flag of applying non-local context')
parser.add_argument('--embedding', action='store_true', default=False,
                    help='flag of applying embdded context')
parser.add_argument('--se', action='store_true', default=False,
                    help='flag of applying SEBlock in ResNet')
parser.add_argument('--num_branches', type=int, default=5, metavar='int',
                    help='# of ensemble branches')
parser.add_argument('--FD_temp', type=float, default=5.0, metavar='float',
                    help='FD_temp (default: 5.0)')
parser.add_argument('--JSD_temp', type=float, default=3.0, metavar='float',
                    help='JSD_temp (default: 5.0)')

parser.add_argument('--baseline', action='store_true', default=False,
                    help='flag of baseline training')
parser.add_argument('--mobile_alpha', type=float, default=1.0, metavar='float',
                    help='mobilenet alpha (default: 1.0)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = args.ngpu if args.cuda else "cpu"

torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Python  version: {}".format(sys.version.replace('\n', ' ')))
print("PyTorch version: {}".format(torch.__version__))
print("cuDNN   version: {}".format(torch.backends.cudnn.version()))

## Dataset preprocessing and instantiating
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data, train=True, download=True, transform =transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/mnt/disk3/cifar100', train=True, download=False, transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/mnt/disk3/cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'imagenet':
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'val3')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)
elif args.dataset == 'cub200':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_train = dataset.CUB2011Classification(args.data, train=True, download=True, transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))

    train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    dataset_val = dataset.CUB2011Classification(args.data, train=False, download=True, transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)
elif args.dataset == 'cars196':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_train = dataset.Cars196Classification(args.data, train=True, download=True, transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))

    train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    dataset_val = dataset.Cars196Classification(args.data, train=False, download=True, transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)
elif args.dataset == 'dogs120':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_train = dataset.Dogs120Classification(args.data, train=True, download=True, transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))

    train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    dataset_val = dataset.Dogs120Classification(args.data, train=False, download=True, transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

## ResNet
if args.arch in ["My_ResNetV3"]:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'device': args.ngpu, 'bottleneck': args.bottleneck, 'se': args.se}

## Exception
else:
    raise ValueError('Check network again !!')
model = models.__dict__[args.arch](**kwargs)

pdist = nn.PairwiseDistance(p=2)

## Model configuration printing
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("=> Model : {}".format(model))
print("=> Model Parameters: {}".format(count_model_parameters(model)))
print("=> Parameter : {}".format(args))

## Model upload on CUDA core
model.cuda(device)
cudnn.benchmark = True

if args.flops:
    input = torch.randn(1, 3, 32, 32).cuda(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)

## Optimization definition
## Classification
criterion = nn.CrossEntropyLoss(reduction='mean').cuda(device)

## Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.1)

## For model monitoring
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'\n=> (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

def get_current_consistency_weight(epoch):
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train_warmup(train_loader, model, criterion, optimizer, epoch):
    model.train()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    losses = AverageMeter()
    aggregation_1 = AverageMeter()
    aggregation_5 = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs, out_t, gate_weight = model(inputs)
        loss_cross = 0

        for i in range(0, args.num_branches-1):
            ce = criterion(outputs[:,:,i], targets)
            loss_cross += ce

        loss_cross += criterion(out_t, targets) # Ensemble <--> Target CE
        loss_cross += criterion(outputs[:,:,-1], targets) # Student <--> Target CE

        # 1. Warm-up
        loss = loss_cross
        losses.update(loss.data, inputs.size(0))

        aggregation_1_ = 0
        aggregation_5_ = 0
        for i in range(args.num_branches):
            metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
            accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
            accTop5_avg[i].update(metrics[1].item(), inputs.size(0))
            if i != args.num_branches-1:
                aggregation_1_ += metrics[0].item()
                aggregation_5_ += metrics[1].item()

        aggregation_1.aggregate(aggregation_1_, args.num_branches-1, inputs.size(0))
        aggregation_5.aggregate(aggregation_5_, args.num_branches-1, inputs.size(0))
        
        e_metrics = accuracy(out_t.data, targets.data, topk=(1, 5))
        accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
        accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        exit()
        show_metrics = {}
        show_metrics.update({'Loss': losses.avg })

        for i in range(args.num_branches+1):
            if i == args.num_branches-1:
                show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
            elif i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})
                pass
        show_metrics.update({'Top1_agg': aggregation_1.agg})
        bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
        bar.next()
    bar.finish()
    return

## Top-1, Top-5 accuracy
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        tot_correct = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].view(-1).float().sum(0)
            #tot_correct.append(correct_k)
            tot_correct.append(correct_k.mul_(100.0 / batch_size))
        return tot_correct

for epoch in range(args.start_epoch, 1):
    print('Warup epoch:{}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    train_warmup(train_loader, model, criterion, optimizer, epoch)
    _ = test_my(test_loader, model, criterion, epoch, initial_tick)
