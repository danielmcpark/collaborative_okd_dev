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
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from utils import Bar, AverageMeter, RunningAverage, ramps
from loss import KLLoss, PSLoss, RelationLoss
from thop import profile

import models

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

# Parser settings
parser = argparse.ArgumentParser(description='MCPARK CVPR Framework')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='GIVEN DATASETS',
                    help='insert training dataset (cifar10, cifar100, imagenet)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='# of epochs to train (default: 160)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--arch', type=str, default='ResNet',
                    help='model to use (look up for a model folder)')
parser.add_argument('--reduction', type=int, default=16,
                    help='wide resnet depth')
parser.add_argument('--depth', type=int, default=20,
                    help='wide resnet depth')
parser.add_argument('--wf', type=int, default=1,
                    help='wide channel factor')
parser.add_argument('--bottleneck', action='store_true', default=False,
                    help='resnet bottleneck')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ngpu', type=str, default='cuda:3',
                    help='CUDA device 0, 1, 2, 3')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, # of gammas should be equal to schedule')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 300],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='baseline training')
parser.add_argument('--okd', action='store_true', default=False,
                    help='running okd')
parser.add_argument('--fd', action='store_true', default=False,
                    help='forced diversity')
parser.add_argument('--dml', action='store_true', default=False,
                    help='running dml')
parser.add_argument('--mine', action='store_true', default=False,
                    help='running dml')
parser.add_argument('--bpscale', action='store_true', default=False,
                    help='running dml')
parser.add_argument('--num_branches', type=int, default=5, metavar='N',
                    help='# of branches')
parser.add_argument('--mobile_alpha', type=float, default=1.0, metavar='FN',
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
kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/mnt/disk3/cifar10', train=True, download=True, transform =transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/mnt/disk3/cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/mnt/disk3/cifar100', train=True, download=True, transform=transforms.Compose([
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
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
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
elif args.dataset == 'cinic10':
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                    std=[0.24205776, 0.23828046, 0.25874835])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                                    ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                    ]))

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)
elif args.dataset == 'cub200':
    train_loader = torch.utils.data.DataLoader(
        dataset.Cub2011('/mnt/disk3/cub200', train=True, download=True, transform=transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(scale=(0.16, 1), size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255])
            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/mnt/disk3/cub200', train=False, transform=transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255])
            ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'tiny-imagenet':
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    train_dataset = dataset.TinyImageNet(traindir, split='train',
                                    transform=transforms.Compose([
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    transforms.RandomRotation(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                                    ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    test_dataset = dataset.TinyImageNet(testdir, split='val',
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

if args.arch=="ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'bottleneck': args.bottleneck}
elif args.arch=="DML":
    kwargs = {'model': 'resnet20', 'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck}
elif args.arch=="CLILR_ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'bpscale': args.bpscale}
elif args.arch=="My_ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'reduction': args.reduction, 'bottleneck': args.bottleneck}
else:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck}
model = models.__dict__[args.arch](**kwargs)

pdist = nn.PairwiseDistance(p=2)

## Model configuration printing
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("=> Model : {}".format(model))
print("=> Model Parameters: {}".format(count_model_parameters(model)))
print("=> Parameter : {}".format(args))

## Model upload on CUDA core
'''
if args.dataset == 'imagenet':
    if args.cuda:
        model = nn.DataParallel(model).cuda()
elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
    if args.cuda:
        model.cuda(device)
'''
model.cuda(device)
cudnn.benchmark = True

## Optimization definition
criterion = nn.CrossEntropyLoss(reduction='none').cuda(device)
criterion_kl = KLLoss(device)
criterion_ps = PSLoss(device, args.num_branches, args.batch_size)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

## Training logic
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print(batch_idx)
        if args.cuda:
            if args.dataset == 'imagenet':
                inputs, targets = inputs.cuda(device), targets.cuda(device)
            elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
                inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        #print(loss)
        loss = loss.mean()
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(args.start_epoch, args.epochs):
    train(train_loader, model, criterion, optimizer, epoch)

print("Finished saving training history")
