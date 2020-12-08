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
from utils import Bar, AverageMeter, RunningAverage, ramps, SpeedoMeter, GPUMemHooker
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
parser.add_argument('--milestones', type=int, nargs='+', default=[60, 100],
                    help='annealing schedule default=[150, 225] of 300 epochs')

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
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='initialize with pretrained parameters')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='float', help='consistency_rampup ratio')
parser.add_argument('--tracefile_train_1', type=str, default='', metavar='string',
                    help='path to save student train accuracy tracefile')
parser.add_argument('--tracefile_train_2', type=str, default='', metavar='string',
                    help='path to save ensemble train accuracy tracefile')
parser.add_argument('--tracefile_test_1', type=str, default='', metavar='sring',
                    help='path to save student test accuracy tracefile')
parser.add_argument('--tracefile_test_2', type=str, default='', metavar='string',
                    help='path to save emsemble test accuracy tracefile')
parser.add_argument('--tracefile_tr_loss', type=str, default='', metavar='string',
                    help='path to save training loss tracefile')
parser.add_argument('--tracefile_diversity', type=str, default='', metavar='string',
                    help='path to save diversity tracefile')
parser.add_argument('--tracefile_thrp', type=str, default='', metavar='string',
                    help='path to save throughput (samples/s) tracefile')
parser.add_argument('--tracefile_mem', type=str, default='', metavar='string',
                    help='path to save GPU mem usage (MiB) tracefile')
parser.add_argument('--trace', action='store_true', default=False,
                    help='flag of saving tracefile')
parser.add_argument('--flops', action='store_true', default=False,
                    help='flag of computing flops')
parser.add_argument('--you_want_to_save', action='store_true', default=False,
                    help='flag of computing flops')

parser.add_argument('--mine', action='store_true', default=False,
                    help='flag of running my network')
parser.add_argument('--gamma', type=float, default=0.4, metavar='float',
                    help='diversity and cooperation balancing params (default: 0.5)')
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
parser.add_argument('--num_branches', type=int, default=4, metavar='int',
                    help='# of ensemble branches')
parser.add_argument('--FD_temp', type=float, default=1.2, metavar='float',
                    help='FD_temp (default: 5.0)')
parser.add_argument('--JSD_temp', type=float, default=2.0, metavar='float',
                    help='JSD_temp (default: 5.0)')

parser.add_argument('--baseline', action='store_true', default=False,
                    help='flag of baseline training')
parser.add_argument('--okd', action='store_true', default=False,
                    help='flag of running okd')
parser.add_argument('--dml', action='store_true', default=False,
                    help='flag of running dml')
parser.add_argument('--bpscale', action='store_true', default=False,
                    help='flag of applying clilr bpscale')
parser.add_argument('--mobile_alpha', type=float, default=1.0, metavar='float',
                    help='mobilenet alpha (default: 1.0)')
args = parser.parse_args()

if os.path.isdir(args.save) is False:
    os.mkdir(args.save)

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
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
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

elif args.dataset == 'tiny-imagenet':
    train_dataset = dataset.TinyImageNet(args.data, split='train', ## From Relational Knowledge Distillation
                                    transform=transforms.Compose([
                                    transforms.Lambda(lambda x: x.convert("RGB")),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                    transforms.RandomRotation(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None, **kwargs)

    test_dataset = dataset.TinyImageNet(args.data, split='val',
                                    transform=transforms.Compose([
                                    transforms.Lambda(lambda x: x.convert("RGB")),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, sampler=None, **kwargs)

## ResNet
if args.arch=="ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'bottleneck': args.bottleneck, 'se': args.se}
elif args.arch=="DML":
    kwargs = {'model': 'mbn_v1', 'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'se': args.se, 'wf': args.wf}
elif args.arch=="CLILR_ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'bpscale': args.bpscale, 'se': args.se}
elif args.arch=="My_ResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'nl': args.nl, 'embedding': args.embedding, 'se': args.se}
elif args.arch in ["ONE_ResNet", "OKDDip_ResNet"]:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck': args.bottleneck, 'se': args.se}
elif args.arch in ["ResNet_IMG", "My_ResNet18", "My_ResNet50"]:
    kwargs = {'dataset': args.dataset, 'num_branches': args.num_branches}
elif args.arch=="resnet50_img":
    kwargs = {'dataset': args.dataset, 'pretrained': args.pretrained}


## DenseNet
elif args.arch == "DenseNet":
    kwargs = {'dataset': args.dataset, 'growth_rate': 12, 'block_config': [6, 6, 6]}
elif args.arch in ["My_DenseNet", "DenseNet_OKDDip"]:
    kwargs = {'dataset': args.dataset, 'growth_rate': 12, 'block_config': [6, 6, 6], 'num_branches': args.num_branches}
elif args.arch == "DenseNet_ONEILR":
    kwargs = {'dataset': args.dataset, 'growth_rate': 12, 'block_config': [6, 6, 6], 'num_branches': args.num_branches, 'bpscale': args.bpscale}

## WRN
elif args.arch=="CLILR_WideResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bpscale': args.bpscale, 'widen_factor': args.wf}
elif args.arch in ["ONE_WideResNet", "OKDDip_WideResNet"]:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'widen_factor': args.wf}
elif args.arch=="My_WideResNet":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'widen_factor': args.wf, 'NL': args.nl, 'embedding': args.embedding}

## VGG
elif args.arch == "VGG":
    kwargs = {'dataset': args.dataset, 'depth': args.depth}
elif args.arch in ["ONE_VGG", "OKDDip_VGG", "CLILR_VGG", "My_VGG"]:
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches}

## MobileNetV1
elif args.arch in ["My_MobileNetV1", "ONE_MobileNetV1", "CLILR_MobileNetV1", "OKDDip_MobileNetV1"]:
    kwargs = {'dataset': args.dataset, 'num_branches': args.num_branches, 'shallow': None}

## MobileNetV2
elif args.arch in ["MobileNet_V2"]:
    kwargs = {'dataset': args.dataset, 'pretrained': args.pretrained}

## NetBasedOurs
elif args.arch=="NetBasedOurs":
    kwargs = {'dataset': args.dataset, 'depth': args.depth, 'num_branches': args.num_branches, 'bottleneck':args.bottleneck, 'nl':args.nl, 'embedding': args.embedding, 'se': args.se, 'models': models, 'arch': args.nb_arch}

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
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.1) # cub200 [60, 100] # cifar [150, 225]
## Metric learning
    #if epoch >= 60 and epoch <= 119:
    #    lr = lr * 0.2
    #elif epoch >= 120 and epoch <= 159:
    #    lr = lr * 0.2 * 0.2
    #elif epoch >= 160 and epoch <= 199:
    #    lr = lr * 0.2 * 0.2 * 0.2
    #elif epoch >= 200 and epoch <= 249:
    #    lr = lr * 0.2 * 0.2 * 0.2 * 0.2
    #elif epoch >=250:
    #    lr = lr * 0.2 * 0.2 * 0.2 * 0.2 * 0.2

## Throughput tracer and Memory hooker
speedometer = SpeedoMeter(args.batch_size, len(train_loader), args.log_interval)
memhooker = GPUMemHooker(len(train_loader), args.log_interval, args.ngpu)

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

## Training logic
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.data, inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.suffix = '({}/{})   || Loss: {:.4f} | top1: {:.4f}'.format(
                        batch_idx + 1,
                        len(train_loader),
                        losses.avg,
                        top1.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

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

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        #outputs, out_t = model(inputs)
        outputs, out_t, gate_weight = model(inputs)
        # tf_targets = proposed_loss.smoothing_onehot(targets)
        loss_cross = 0
        loss_kd = 0
        loss_corr = 0
        loss_ajs = 0
        loss_sim = 0

        var_list = 0
        ## Cross-entropy loss (1. Data loss)
        for i in range(args.num_branches-1):
            loss_cross += criterion(outputs[:,:,i], targets) # Peers <--> Target CE included student
            ## alpha Jensen-Shannon divergence loss (5. Cooperation loss)
            if args.ajs:
                loss_ajs += criterion_aJS(outputs[:,:,i],
                                          outputs[:,:,:-1],
                                          gate_weight,
                                          i,
                                          ty=args.coo_type,
                                          en=True)
        loss_cross += criterion(out_t, targets) # Ensemble <--> Target CE
        loss_cross += criterion(outputs[:,:,-1], targets) # Student <--> Target CE

        ## Batch correleration loss (2. Batch correlation loss)
        if args.sim:
            '''
            stu_outputs = F.normalize(outputs[:,:,-1], dim=1)
            stu_outputs_t = stu_outputs.permute(1, 0)
            corr_stu = torch.mm(stu_outputs, stu_outputs_t)

            en_outputs = F.normalize(out_t, dim=1)
            en_outputs_t = en_outputs.permute(1, 0)
            corr_en = torch.mm(en_outputs, en_outputs_t)

            diff = corr_stu - corr_en

            loss_sim = ((diff * diff).view(-1, 1).sum(0) / (args.batch_size * args.batch_size)).squeeze(0) # Student <--> Ensemble
            '''
            loss_sim = proposed_loss.similarity_loss(h_features, gate_weight, args.num_branches, types='mean')
            loss_sim = (loss_sim / (args.batch_size * args.batch_size)).squeeze(0)

        ## Hinton KD loss of ensemble and target peer (3. Original distilation loss)
        loss_kd = criterion_kl(outputs[:,:,-1], out_t, en=True) # Student <--> Ensemble
        loss_kd = consistency_weight * loss_kd # Distillation Loss

        ## Forced diversity loss (4. Diversity loss)
        if args.fd:
            loss_corr = proposed_loss.covariance_loss(outputs[:,:,:-1], targets, T=args.FD_temp, device=device)

        ## Define total loss
        # 1. Data + 2. Batch correlation + 3. HintonKD + 4. Diversity + 5. Cooperation (Perfect JSD)
        loss = loss_cross + 3000*loss_sim + loss_kd + (1-args.gamma)*loss_corr + args.gamma*loss_ajs

        losses_kd.update(loss_kd.data, inputs.size(0))
        losses_corr.update(loss_corr.data, inputs.size(0)) if args.fd else losses_corr.update(0, inputs.size(0))
        losses_ajs.update(loss_ajs.data, inputs.size(0)) if args.ajs else losses_ajs.update(0, inputs.size(0))
        losses_sim.update(loss_sim.data, 1) if args.sim else losses_sim.update(0, 1)
        losses.update(loss.data, inputs.size(0))

        for i in range(args.num_branches):
            metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
            accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
            accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

        e_metrics = accuracy(out_t.data, targets.data, topk=(1, 5))
        accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
        accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        show_metrics = {}
        show_metrics.update({'Loss': losses.avg, 'HKD': losses_kd.avg, 'FD': losses_corr.avg, 'Coo': losses_ajs.avg, 'BS': losses_sim.avg})

        for i in range(args.num_branches+1):
            if i == args.num_branches-1:
                show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
            elif i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                pass
                #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})
        bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
        bar.next()
        #throughput = speedometer(batch_idx+1, epoch, init_tick)
        #mem_alloc = memhooker(batch_idx+1)
    bar.finish()
    #return show_metrics, throughput, mem_alloc
    return show_metrics

def train_distill(train_loader, model, criterion, criterion_kl, optimizer, epoch, init_tick):
    model.train()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_corr = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs, out_t = model(inputs)
        loss_cross = 0
        loss_kl = 0
        if args.bpscale:
            for i in range(args.num_branches):
                loss_cross += criterion(outputs[:,:,i], targets)
                loss_kl += criterion_kl(outputs[:,:,i], out_t[:,:,i])
        else:
            for i in range(args.num_branches):
                loss_cross += criterion(outputs[:,:,i], targets)
                loss_kl += criterion_kl(outputs[:,:,i], out_t)
            loss_cross += criterion(out_t, targets)
        loss_kl = consistency_weight * loss_kl
        loss = loss_cross + loss_kl

        losses_kl.update(loss_kl.data, inputs.size(0))
        losses.update(loss.data, inputs.size(0))

        for i in range(args.num_branches):
            metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
            accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
            accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

        if args.bpscale:
            e_metrics = accuracy(torch.mean(outputs, dim=2), targets, topk=(1, 5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
            accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))
        else:
            e_metrics = accuracy(out_t.data, targets.data, topk=(1, 5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
            accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        show_metrics = {}
        show_metrics.update({'Loss': losses.avg, 'LossKL': losses_kl.avg})
        for i in range(args.num_branches+1):
            if i == args.num_branches-1:
                show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
            elif i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                pass
                #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})

        bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
        bar.next()
        throughput = speedometer(batch_idx+1, epoch, init_tick)
        mem_alloc = memhooker(batch_idx+1)
    bar.finish()
    return show_metrics, throughput, mem_alloc

def train_dml(train_loader, model, criterion, criterion_kl, optimizer, epoch, init_tick):
    model.train()

    accTop1_avg = list(range(args.num_branches + 1))
    accTop5_avg = list(range(args.num_branches + 1))
    for i in range(args.num_branches + 1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs = model(inputs)

        ## Compute model output loss
        loss_cross = 0
        for idx in range(args.num_branches):
            loss_cross += criterion(outputs[:,:,idx], targets)

        ## pair-wise loss
        loss_kl = 0
        for x in range(args.num_branches):
            en_output = 0
            for y in range(args.num_branches):
                if x != y:
                    en_output += outputs[:,:,y]
            loss_kl += criterion_kl(outputs[:,:,x], en_output / (args.num_branches-1))

        loss = loss_cross + loss_kl
        losses_kl.update(loss_kl.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        for i in range(args.num_branches):
            metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
            accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
            accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

        e_metrics = accuracy(torch.mean(outputs, dim=2), targets, topk=(1, 5))
        accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
        accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        show_metrics = {}
        show_metrics.update({'Loss': losses.avg, 'LossKL': losses_kl.avg})
        for i in range(args.num_branches+1):
            if i == args.num_branches-1:
                show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
            elif i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                pass
                #show_metrics.update({'Top1_stu_'+str(i): accTop1_avg[i].avg})

        bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
        bar.next()
        throughput = speedometer(batch_idx+1, epoch, init_tick)
        mem_alloc = memhooker(batch_idx+1)

    bar.finish()
    return show_metrics, throughput, mem_alloc

## Train OKDDip
def train_okd(train_loader, model, criterion, criterion_kl, optimizer, epoch, init_tick):
    model.train()

    accTop1_avg = list(range(args.num_branches + 1))
    accTop5_avg = list(range(args.num_branches + 1))
    for i in range(args.num_branches + 1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    #loss_true_avg = AverageMeter()
    losses_kl = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        outputs, x_m, x_stu = model(inputs)
        loss_true = 0
        loss_group = 0
        for i in range(args.num_branches-1):
            loss_true += criterion(outputs[:,:,i], targets)
            loss_group += criterion_kl(outputs[:,:,i], x_m[:,:,i])

        loss_cross = loss_true + criterion(x_stu, targets)
        loss_kl = consistency_weight*(loss_group + criterion_kl(x_stu, torch.mean(outputs, dim=2)))
        loss = loss_cross + loss_kl

        #loss_true_avg.update(loss_true.item(), inputs.size(0))
        losses_kl.update(loss_kl.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        for i in range(args.num_branches-1):
            metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
            accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
            accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

        ## Storing student metrics
        metrics = accuracy(x_stu, targets, topk=(1, 5))
        accTop1_avg[args.num_branches-1].update(metrics[0].item(), inputs.size(0))
        accTop5_avg[args.num_branches-1].update(metrics[1].item(), inputs.size(0))

        ## Storing ensemble metrics
        e_metrics = accuracy(torch.mean(outputs, dim=2), targets, topk=(1, 5))
        accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
        accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        show_metrics = {}
        show_metrics.update({'Loss': losses.avg, 'LossKL': losses_kl.avg})
        for i in range(args.num_branches+1):
            if i == args.num_branches-1:
                show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
            elif i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                pass
                #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})

        bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
        bar.next()
        throughput = speedometer(batch_idx+1, epoch, init_tick)
        mem_alloc = memhooker(batch_idx+1)

    bar.finish()
    return show_metrics, throughput, mem_alloc

## Testing logic
def test(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    speed = AverageMeter()

    model.eval()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

            losses.update(loss.data, inputs.size(0))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            bar.suffix = '({}/{}) | Loss: {:.4f} | top1: {:.4f}'.format(
                    batch_idx + 1,
                    len(test_loader),
                    losses.avg,
                    top1.avg,
            )
            bar.next()
        bar.finish()

    return (losses.avg, top1.avg)

def test_my(test_loader, model, criterion, epoch, init_tick):
    model.eval()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    dist_avg = AverageMeter()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            #outputs, out_t = model(inputs)
            outputs, out_t, _ = model(inputs)

            for i in range(args.num_branches):
                metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
                accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
                accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

            e_metrics = accuracy(out_t.data, targets.data, topk=(1, 5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
            accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))

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

            show_metrics = {}
            show_metrics.update({'Diversity': dist_avg.avg})

            for i in range(args.num_branches+1):
                if i == args.num_branches-1:
                    show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
                elif i == args.num_branches:
                    show_metrics.update({'Top1_en': accTop1_avg[i].avg})
                else:
                    pass
                    #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})
            bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
            bar.next()
        bar.finish()
    return show_metrics

def test_distill(test_loader, model, criterion, epoch):
    model.eval()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    dist_avg = AverageMeter()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            outputs, out_t = model(inputs)
            for i in range(args.num_branches):
                metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
                accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
                accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

            if args.bpscale:
                e_metrics = accuracy(torch.mean(outputs, dim=2), targets, topk=(1, 5))
                accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
                accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))
            else:
                e_metrics = accuracy(out_t.data, targets.data, topk=(1, 5))
                accTop1_avg[args.num_branches].update(e_metrics[0].item(), inputs.size(0))
                accTop5_avg[args.num_branches].update(e_metrics[1].item(), inputs.size(0))

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

            show_metrics = {}
            show_metrics.update({'Diversity': dist_avg.avg})
            for i in range(args.num_branches+1):
                if i == args.num_branches-1:
                    show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
                elif i == args.num_branches:
                    show_metrics.update({'Top1_en': accTop1_avg[i].avg})
                else:
                    pass
                    #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})

            bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
            bar.next()
        bar.finish()
    return show_metrics

def test_dml(test_loader, model, criterion, epoch):
    model.eval()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches + 1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    diversity = AverageMeter() # diversity

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            outputs = model(inputs)
            for i in range(args.num_branches):
                metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
                accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
                accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

            e_metrics = accuracy(torch.mean(outputs, dim=2), targets, topk=(1, 5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item())
            accTop5_avg[args.num_branches].update(e_metrics[1].item())
            
            outputs = F.softmax(outputs, dim=1)
            for idx in range(outputs.size(0)):
                ret = outputs[idx,:,:]
                ret = ret.t()
                sim = 0
                for j in range(args.num_branches-1):
                    for k in range(j+1, args.num_branches-1):
                        sim += pdist(ret[j:j+1,:],ret[k:k+1,:])
                sim = sim / (args.num_branches-1)
                diversity.update(sim.item(), outputs.size(0))

            show_metrics = {}
            show_metrics.update({'Diversity': diversity.avg})
            for i in range(args.num_branches+1):
                if i == args.num_branches-1:
                    show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
                elif i == args.num_branches:
                    show_metrics.update({'Top1_en': accTop1_avg[i].avg})
                else:
                    pass
                    #show_metrics.update({'Top1_stu'+str(i): accTop1_avg[i].avg})

            bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
            bar.next()
        bar.finish()
    return show_metrics


def test_okd(test_loader, model, criterion, epoch):
    model.eval()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches + 1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    diversity = AverageMeter() # diversity

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            outputs, x_m, x_stu = model(inputs)
            for i in range(args.num_branches-1):
                metrics = accuracy(outputs[:,:,i], targets, topk=(1, 5))
                accTop1_avg[i].update(metrics[0].item(), inputs.size(0))
                accTop5_avg[i].update(metrics[1].item(), inputs.size(0))

            metrics = accuracy(x_stu, targets, topk=(1, 5))
            accTop1_avg[args.num_branches-1].update(metrics[0].item(), inputs.size(0))
            accTop5_avg[args.num_branches-1].update(metrics[0].item(), inputs.size(0))

            e_metrics = accuracy(torch.mean(outputs, dim=2), targets, topk=(1, 5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item())
            accTop5_avg[args.num_branches].update(e_metrics[1].item())
            
            outputs = F.softmax(outputs, dim=1)
            for idx in range(outputs.size(0)):
                ret = outputs[idx,:,:]
                ret = ret.t()
                sim = 0
                for j in range(args.num_branches-1):
                    for k in range(j+1, args.num_branches-1):
                        sim += pdist(ret[j:j+1,:],ret[k:k+1,:])
                sim = sim / (args.num_branches-1) # total branches - myself
                diversity.update(sim.item(), outputs.size(0))
            
            show_metrics = {}
            show_metrics.update({'Diversity': diversity.avg})
            for i in range(args.num_branches+1):
                if i == args.num_branches-1:
                    show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
                elif i == args.num_branches:
                    show_metrics.update({'Top1_en': accTop1_avg[i].avg})
                else:
                    pass
                    #show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})

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
    if args.okd:
        test_metrics = test_okd(test_loader, model, criterion, 0, tick)
    elif args.dml:
        test_metrics = test_dml(test_loader, model, criterion, 0, tick)
    elif args.mine:
        test_metrics = test_my(test_loader, model, criterion, 0, tick)
    elif args.baseline:
        test_metrics = test(test_loader, model, criterion, 0)
    else:
        test_metrics = test_distill(test_loader, model, criterion)
    exit()

## Baseline training steps
elif args.baseline:
    train_acc = []
    test_acc = []
    tr_loss = []
    te_loss = []
    best_prec1 = 0.

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc1 = train(train_loader, model, criterion, optimizer, epoch)
        test_loss, test_acc1 = test(test_loader, model, criterion, epoch)

        train_acc.append(train_acc1)
        test_acc.append(test_acc1)
        tr_loss.append(train_loss)

        if args.trace:
            pd.Series(train_acc).to_csv(os.path.join("./csv", args.tracefile_train_1), index=None)
            pd.Series(test_acc).to_csv(os.path.join("./csv", args.tracefile_test_1), index=None)
            pd.Series(tr_loss).to_csv(os.path.join("./csv", args.tracefile_tr_loss), index=None)

        is_best = test_acc1 > best_prec1
        best_prec1 = max(test_acc1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)
        print("Best accuracy: "+str(best_prec1))
        scheduler.step()
    print("Finished saving training history")

## Online distillation steps
else:
    for i in range(2):
        globals()['train_acc_{}'.format(i)] = list()
        globals()['test_acc_{}'.format(i)] = list()
    #tr_loss = []
    diversity = []
    thrp = []
    mem_util = []

    best_prec1 = 0.
    best_en = 0.
    initial_tick = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print('Current epoch: {}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        if args.okd:
            train_metrics, throughput, mem_usage = train_okd(train_loader, model, criterion, criterion_kl, optimizer, epoch, initial_tick)
            test_metrics = test_okd(test_loader, model, criterion, epoch)
        elif args.dml:
            train_metrics, throughput, mem_usage = train_dml(train_loader, model, criterion, criterion_kl, optimizer, epoch, initial_tick)
            test_metrics = test_dml(test_loader, model, criterion, epoch)
        elif args.mine:
            train_metrics = train_my(train_loader, model, criterion, criterion_kl, criterion_aJS, optimizer, epoch, initial_tick)
            test_metrics = test_my(test_loader, model, criterion, epoch, initial_tick)
        else:
            train_metrics, throughput, mem_usage = train_distill(train_loader, model, criterion, criterion_kl, optimizer, epoch, initial_tick)
            test_metrics = test_distill(test_loader, model, criterion, epoch)

        ## Storing student or ensemble accuracy
        globals()['train_acc_0'].append(train_metrics['Top1_stu'])
        globals()['test_acc_0'].append(test_metrics['Top1_stu'])

        globals()['train_acc_1'].append(train_metrics['Top1_en'])
        globals()['test_acc_1'].append(test_metrics['Top1_en'])

        ## Storing other classifier accuracy
        #for i in range(1, args.num_branches):
        #    globals()['train_acc_{}'.format(i)].append(train_metrics['Top1_C'+str(i)])
        #    globals()['test_acc_{}'.format(i)].append(test_metrics['Top1_C'+str(i)])

        ## Storing training loss and diversity
        #tr_loss.append(train_metrics['Loss'])
        diversity.append(test_metrics['Diversity'])
        #thrp.append(throughput)
        #mem_util.append(mem_usage)

        if args.trace:
            pd.Series(globals()['train_acc_0']).to_csv(os.path.join("./csv", args.tracefile_train_1), index=None)
            pd.Series(globals()['train_acc_1']).to_csv(os.path.join("./csv", args.tracefile_train_2), index=None)

            pd.Series(globals()['test_acc_0']).to_csv(os.path.join("./csv", args.tracefile_test_1), index=None)
            pd.Series(globals()['test_acc_1']).to_csv(os.path.join("./csv", args.tracefile_test_2), index=None)

            #pd.Series(tr_loss).to_csv(os.path.join("./csv", args.tracefile_tr_loss), index=None)
            pd.Series(diversity).to_csv(os.path.join("./csv", args.tracefile_diversity), index=None)
            #pd.Series(thrp).to_csv(os.path.join("./csv", args.tracefile_thrp), index=None)
            #pd.Series(mem_util).to_csv(os.path.join("./csv", args.tracefile_mem), index=None)

        is_best = test_metrics['Top1_stu'] > best_prec1
        best_prec1 = max(test_metrics['Top1_stu'], best_prec1)
        best_en = max(test_metrics['Top1_en'], best_en)
        if args.you_want_to_save:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath=args.save)
        print("Best accuracy: "+str(best_prec1))
        print("Best ensemble: "+str(best_en))
        print("=> SE: {}, Depth: {}, g: {}, tJSD: {}, tFD: {}, p: {}".format(args.se, args.depth, args.gamma, args.JSD_temp, args.FD_temp, args.num_branches))
        scheduler.step()
    print("Finished saving training history")
