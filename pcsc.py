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

# Parser settings
parser = argparse.ArgumentParser(description='MCPARK AAAI Framework')
parser.add_argument('--dataset', type=str, default='cifar10', metavar='string',
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

parser.add_argument('--arch', type=str, default='WideResNet', metavar='string',
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
parser.add_argument('--save', type=str, default='', metavar='string',
                    help='path to save prune model (default: ./logs)')
parser.add_argument('--resume', type=str, default='', metavar='string',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', type=str, default='', metavar='string',
                    help='path to dataset')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='whether to run evaluation')
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
parser.add_argument('--stifle_ce', type=float, default=0.0, metavar='float',
                    help='JSD_temp (default: 5.0)')
parser.add_argument('--wu', action='store_true', default=False,
                    help='flag of warm-up')

parser.add_argument('--baseline', action='store_true', default=False,
                    help='flag of baseline training')
parser.add_argument('--mobile_alpha', type=float, default=1.0, metavar='float',
                    help='mobilenet alpha (default: 1.0)')
args = parser.parse_args()

if os.path.isdir(args.save) is False:
    try:
        os.mkdir(args.save)
    except:
        print("Bypassing Exception!")
        pass

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
        datasets.CIFAR100(args.data, train=True, download=True, transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(args.data, train=False, transform=transforms.Compose([
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
if args.arch=="ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'bottleneck': args.bottleneck, 'se': args.se}
elif args.arch=="DML":
    kwargs = {'model': 'wrn16_4', 'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'se': args.se, 'wf': args.wf}
elif args.arch in ["My_ResNet", "My_ResNetV2"]:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'device': args.ngpu, 'bottleneck': args.bottleneck, 'NL': args.nl, 'embedding': args.embedding, 'se': args.se}
elif args.arch=="My_ResNetO":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'NL': args.nl, 'embedding': args.embedding, 'se': args.se}
elif args.arch in ["ResNet_IMG", "My_ResNet18", "My_ResNet50"]:
    kwargs = {'dataset': args.dataset, 'num_branches': args.num_branches, 'pretrained': args.pretrained}
elif args.arch=="resnet50_img":
    kwargs = {'dataset': args.dataset, 'pretrained': args.pretrained}

##
elif args.arch=="DenseNet":
    kwargs = {'dataset': args.dataset, 'growth_rate': 12, 'block_config': [6, 6, 6]}
elif args.arch in ["My_DenseNet"]:
    kwargs = {'dataset': args.dataset, 'growth_rate': 12, 'block_config': [6, 6, 6], 'num_branches': args.num_branches}

## WRN
elif args.arch=="WideResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'widen_factor': args.wf}
elif args.arch=="My_WideResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'widen_factor': args.wf, 'NL': args.nl, 'embedding': args.embedding}

## VGG
elif args.arch=="VGG":
    kwargs = {'dataset': args.dataset, 'depth': args.depth}
elif args.arch in ["My_VGG"]:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches}

## MobileNetV1
elif args.arch in ["My_MobileNetV1"]:
    kwargs = {'dataset': args.dataset, 'num_branches': args.num_branches, 'shallow': None}

## NetBasedOurs
elif args.arch=="NetBasedOurs":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'nl': args.nl, 'embedding': args.embedding, 'se': args.se, 'models': models, 'arch': args.nb_arch, 'pretrained': args.pretrained, 'device': args.ngpu}

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
criterion_kl = KLLoss(device)
criterion_aJS = aJSLoss(device, args.num_branches, args.JSD_temp)

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
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'\n=> (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

def get_current_consistency_weight(epoch):
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train_my(train_loader, model, criterion, criterion_kl, criterion_aJS, optimizer, epoch, init_tick):
    model.train()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    losses_corr = AverageMeter()
    losses_ajs = AverageMeter()
    losses_sim = AverageMeter()
    aggregation_1 = AverageMeter()
    aggregation_5 = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)
    #pc = list()
    #sc = list()
    pc = 0
    sc = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs, out_t, gate_weight = model(inputs)
        loss_cross = 0
        loss_kd = 0
        loss_corr = 0
        loss_ajs = 0

        ce = criterion(outputs[:,:,0], targets)
        ce_pool = ce.unsqueeze(-1)
        if ce > args.stifle_ce:
            loss_cross += ce
        else:
            pass
        #loss_cross += ce
        for i in range(1, args.num_branches-1):
            ce = criterion(outputs[:,:,i], targets)
            ce_pool = torch.cat([ce_pool, ce.unsqueeze(-1)], dim=-1)
            #loss_cross += ce
            if ce > args.stifle_ce:
                loss_cross += ce
            else:
                pass

            if args.ajs:
                loss_ajs += criterion_aJS(outputs[:,:,i],
                                          outputs[:,:,:-1],
                                          gate_weight,
                                          i,
                                          ty=args.coo_type,
                                          en=True)
        #pc.append(proposed_loss.PearsonCorrelation(gate_weight, ce_pool.unsqueeze(0)).item()) # [bsz, n_branches], [n_branches]
        #sc.append(proposed_loss.SpearmanCorrelation(gate_weight, ce_pool))
        pc += proposed_loss.PearsonCorrelation(gate_weight, ce_pool).item()
        sc += proposed_loss.SpearmanCorrelation(gate_weight, ce_pool).item()
        loss_cross += criterion(out_t, targets) # Ensemble <--> Target CE
        loss_cross += criterion(outputs[:,:,-1], targets) # Student <--> Target CE

        ## Hinton KD loss of ensemble and target peer (3. Original distilation loss)
        loss_kd = criterion_kl(outputs[:,:,-1], out_t, en=True) # Student <--> Ensemble
        loss_kd = consistency_weight * loss_kd # Distillation Loss

        ## Forced diversity loss (4. Diversity loss)
        if args.fd:
            loss_corr = proposed_loss.covariance_loss(outputs[:,:,:-1], targets, T=args.FD_temp, device=device)

        ## Define total loss
        # 1. Data + 2. Batch correlation + 3. HintonKD + 4. Diversity + 5. Cooperation
        loss = loss_cross + loss_kd + (1-args.gamma)*args.slope*loss_corr + args.gamma*loss_ajs

        losses_kd.update(loss_kd.data, inputs.size(0))
        losses_corr.update(loss_corr.data, inputs.size(0)) if args.fd else losses_corr.update(0, inputs.size(0))
        losses_ajs.update(loss_ajs.data, inputs.size(0)) if args.ajs else losses_ajs.update(0, inputs.size(0))
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
        
        show_metrics = {}
        show_metrics.update({'Loss': losses.avg, 'FD': losses_corr.avg, 'Coo': losses_ajs.avg})

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
    pc = pc / len(train_loader)
    sc = sc / len(train_loader)
    return show_metrics, pc, sc, gate_weight.mean(0)

def train_warmup(train_loader, model, criterion, criterion_kl, optimizer, epoch):
    model.train()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    aggregation_1 = AverageMeter()
    aggregation_5 = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs, out_t, gate_weight = model(inputs)
        loss_cross = 0
        loss_kd = 0

        for i in range(0, args.num_branches-1):
            ce = criterion(outputs[:,:,i], targets)
            loss_cross += ce

        loss_cross += criterion(out_t, targets) # Ensemble <--> Target CE
        loss_cross += criterion(outputs[:,:,-1], targets) # Student <--> Target CE

        ## Hinton KD loss of ensemble and target peer (3. Original distilation loss)
        loss_kd = criterion_kl(outputs[:,:,-1], out_t, en=True) # Student <--> Ensemble
        loss_kd = consistency_weight * loss_kd # Distillation Loss

        # 1. Warm-up
        loss = loss_cross + loss_kd

        losses_kd.update(loss_kd.data, inputs.size(0))
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

def test_my(test_loader, model, criterion, epoch, init_tick):
    model.eval()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    dist_avg = AverageMeter()
    aggregation_1 = AverageMeter()
    aggregation_5 = AverageMeter()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            outputs, out_t, _ = model(inputs)
            '''
            if torch.argmax(outputs[:,:,-1]) == targets:
                fig = plt.figure(figsize=(20, 10))
                for i in range(args.num_branches):
                    #prob = F.softmax(outputs[:,:,i], dim=-1)
                    prob = outputs[:,:,i]
                    prob = prob.tolist()
                    prob = sum(prob, [])
                    x = range(len(prob))
                    ax = fig.add_subplot(3, 4, i+1)
                    ax.plot(x, prob)

                #prob = F.softmax(out_t, dim=-1)
                prob = out_t
                prob = prob.tolist()
                prob = sum(prob, [])
                x = range(len(prob))
                ax1 = fig.add_subplot(3, 4, 9)
                ax1.plot(x, prob)
                label_ = F.one_hot(targets, 100).tolist()
                label_ = sum(label_, [])
                ax2 = fig.add_subplot(3, 4, 10)
                ax2.plot(x, label_)
                plt.show()
                exit()
            '''
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
            '''
            outputs = F.softmax(outputs, dim=1)
            for idx in range(outputs.size(0)):
                ret = outputs[idx,:,:]
                ret = ret.t()
                sim = 0
                for j in range(args.num_branches-1):
                    for k in range(j+1, args.num_branches-1):
                        sim += pdist(ret[j:j+1,:],ret[k:k+1,:])
                sim = sim / (args.num_branches-1) # total branches
                dist_avg.update(sim.item(), outputs.size(0))
            '''
            show_metrics = {}
            show_metrics.update({'Diversity': dist_avg.avg})

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
    return show_metrics

## Storing model checkpoint
def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

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

## Just for testing
if args.evaluate:
    tick = time.time()
    test_metrics = test_my(test_loader, model, criterion, 0, tick)
    exit()

diversity = []
pearson = list()
spearman = list()

best_prec1 = 0.
best_en = 0.
initial_tick = time.time()
gate_coeff_list = torch.zeros(size=(args.epochs, args.num_branches-1))

if args.wu:
    for epoch in range(args.start_epoch, 5):
        print('Warup epoch:{}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        train_warmup(train_loader, model, criterion, criterion_kl, optimizer, epoch)
        _ = test_my(test_loader, model, criterion, epoch, initial_tick)

for epoch in range(args.start_epoch, args.epochs):
    print('Current epoch: {}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    train_metrics, pc, sc, gate_values = train_my(train_loader, model, criterion, criterion_kl, criterion_aJS, optimizer, epoch, initial_tick)
    test_metrics = test_my(test_loader, model, criterion, epoch, initial_tick)

    print(pc, sc)
    pearson.append(pc)
    spearman.append(sc)
    gate_coeff_list[epoch] = gate_values
    diversity.append(test_metrics['Diversity'])

    pd.Series(pearson).to_csv(os.path.join("./csv/pcsc", args.tracefile_pearson), index=None)
    pd.Series(spearman).to_csv(os.path.join("./csv/pcsc", args.tracefile_spearman), index=None)
    torch.save(gate_coeff_list, os.path.join("./csv/pcsc", args.gate_coeff))
    
    #pd.Series(diversity).to_csv(os.path.join("./csv", args.tracefile_diversity), index=None)
    #is_best = test_metrics['Top1_stu'] > best_prec1
    is_best = test_metrics['Top1_en'] > best_en

    best_prec1 = max(test_metrics['Top1_stu'], best_prec1)
    best_en = max(test_metrics['Top1_en'], best_en)
    
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)
     
    print("Best accuracy: "+str(best_prec1))
    print("Best ensemble: "+str(best_en))
    print("=> SE: {}, Depth: {}, g: {}, slope: {}, p: {}".format(args.se, args.depth, args.gamma, args.slope, args.num_branches))
    scheduler.step()
    print("Finished saving training history")
