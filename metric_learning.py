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
from metric.pairsampler import DistanceWeighted
from metric.batchsampler import NPairs
from loss import KLLoss, L2Triplet
from thop import profile
from tqdm import tqdm

import metric
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
parser.add_argument('--iter-per-epoch', type=int, default=100, metavar='int',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.1, metavar='float',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='float',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='float',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--arch', type=str, default='WideResNet', metavar='string',
                    help='model to use (look up for a model folder)')
parser.add_argument('--reduction', type=int, default=16, metavar='int',
                    help='reduction ratio of SE block channel embedding')
parser.add_argument('--depth', type=int, default=20, metavar='int',
                    help='resnet depth')
parser.add_argument('--wf', type=int, default=1, metavar='int',
                    help='wide resnet channel variation factor')
parser.add_argument('--embedding_size', type=int, default=128, metavar='int',
                    help='embedding size (default: 1)')
parser.add_argument('--bottleneck', action='store_true', default=False,
                    help='resnet bottleneck usage')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ngpu', type=str, default='cuda:0', metavar='string',
                    help='CUDA device 0, 1, 2, 3')
parser.add_argument('--seed', type=int, default=1, metavar='int',
                    help='random seed (default: 1)')
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
parser.add_argument('--schedule', type=int, nargs='+', default=[25, 30, 35],
                    help='decrease learning rate at these epochs.')
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
parser.add_argument('--trace', action='store_true', default=False,
                    help='flag of saving tracefile')
parser.add_argument('--flops', action='store_true', default=False,
                    help='flag of computing flops')
parser.add_argument('--okd', action='store_true', default=False,
                    help='flag of running okd')
parser.add_argument('--fd', action='store_true', default=False,
                    help='flag of applying forced diversity')
parser.add_argument('--dml', action='store_true', default=False,
                    help='flag of running dml')
parser.add_argument('--mine', action='store_true', default=False,
                    help='flag of running my network')
parser.add_argument('--bpscale', action='store_true', default=False,
                    help='flag of applying clilr bpscale')
parser.add_argument('--num_branches', type=int, default=5, metavar='int',
                    help='# of ensemble branches')
parser.add_argument('--mobile_alpha', type=float, default=1.0, metavar='float',
                    help='mobilenet alpha (default: 1.0)')
parser.add_argument('--triplet_margin', type=float, default=0.2, metavar='float',
                    help='mobilenet alpha (default: 0.2)')

args = parser.parse_args()

Temperature = 5

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
if args.dataset == 'cub200':
    dataset_train = dataset.CUB2011Metric(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=NPairs(dataset_train, args.batch_size, 5, args.iter_per_epoch), **kwargs)

    dataset_train_eval = dataset.CUB2011Metric(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

    train_eval_loader = torch.utils.data.DataLoader(
        dataset_train_eval,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    eval_loader = torch.utils.data.DataLoader(
        dataset.CUB2011Metric(args.data, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

elif args.dataset=='cars196':
    dataset_train = dataset.Cars196Metric(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=NPairs(dataset_train, args.batch_size, 5, args.iter_per_epoch), **kwargs)

    dataset_train_eval = dataset.Cars196Metric(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

    train_eval_loader = torch.utils.data.DataLoader(
        dataset_train_eval,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    eval_loader = torch.utils.data.DataLoader(
        dataset.Cars196Metric(args.data, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

elif args.dataset=='stanford':
    dataset_train = dataset.StanfordOnlineProductsMetric(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=NPairs(dataset_train, args.batch_size, 5, args.iter_per_epoch), **kwargs)

    dataset_train_eval = dataset.StanfordOnlineProductsMetric(args.data, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

    train_eval_loader = torch.utils.data.DataLoader(
        dataset_train_eval,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    eval_loader = torch.utils.data.DataLoader(
        dataset.StanfordOnlineProductsMetric(args.data, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[104/255.0, 117/255.0, 128/255.0], std=[1.0/255, 1.0/255, 1.0/255]) # for googlenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
else:
    raise RuntimeError('Request valid dataset!!..')

if args.arch in ["resnet18_img", "resnet50_img"]:
    kwargs = {'dataset': args.dataset, 'pretrained': True}
model_base = models.__dict__[args.arch](**kwargs)

model  = models.LinearEmbedding(model_base,
                         output_size=model_base.fc1.weight.size(0),
                         embedding_size=args.embedding_size,
                         normalize=True)

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
    input = torch.randn(1, 3, 224, 224).cuda(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)

## Optimization definition
## Metric learning
triplet_criterion = L2Triplet(sampler=DistanceWeighted(), margin=args.triplet_margin)

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

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

## Training logic
def train(loader, model, criterion, optimizer, epoch):
    lr_scheduler.step()
    model.train()

    loss_all = []
    train_iter = tqdm(loader)
    for inputs, targets in train_iter:
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)

        embeds = model(inputs)
        loss = criterion(embeds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())

        train_iter.set_description("[Train][Epoch %d] Triplet: %.5f" %(epoch, loss.item()))
    print('[Epoch %d] Loss: %.5f' %(epoch, torch.Tensor(loss_all).mean()))

    return

## Testing logic
def test(loader, model, epoch):
    K = [1]
    model.eval()
    test_iter = tqdm(loader)
    embeddings_all, labels_all = [], []

    with torch.no_grad():
        for inputs, targets in test_iter:
            if args.cuda:
                inputs, targets = inputs.cuda(device), targets.cuda(device)

            outputs = model(inputs)
            embeddings_all.append(outputs.data)
            labels_all.append(targets.data)
            test_iter.set_description("[Eval][Epoch %d]" %epoch)

        embeddings_all = torch.cat(embeddings_all).cpu()
        labels_all = torch.cat(labels_all).cpu()
        rec = metric.recall(embeddings_all, labels_all, K=K)

        for k, r in zip(K, rec):
            print(' [Epoch %d] Recall@%d: [%.4f]\n' %(epoch, k, 100*r))

    return rec[0]

## Just for testing
if args.evaluate:
    print(test(eval_loader, model, 0))
    exit()

best_train_rec = test(train_eval_loader, model, 0)
best_val_rec = test(eval_loader, model, 0)

for epoch in range(1, args.epochs+1):
    train(train_loader, model, triplet_criterion, optimizer, epoch)
    train_recall = test(train_eval_loader, model, epoch)
    val_recall = test(eval_loader, model, epoch)

    if best_train_rec < train_recall:
        best_train_rec = train_recall

    if best_val_rec < val_recall:
        best_val_rec = val_recall
        if args.save is not None:
            if not os.path.isdir(args.save):
                os.mkdir(args.save)
            torch.save(model.state_dict(), "%s/%s" %(args.save, "model_best.pth.tar"))
    
    if args.save is not None:
        if not os.path.isdir(args.save):
            os.mkdir(args.save_dir)
        torch.save(model.state_dict(), "%s/%s" %(args.save, "checkpoint.pth.tar"))
        with open("%s/result.txt" % args.save, 'w') as f:
            f.write('Best Train Recall@1: %.4f\n' % (best_train_rec * 100))
            f.write('Best Test Recall@1: %.4f\n' % (best_val_rec * 100))
            f.write('Final Recall@1: %.4f\n' % (val_recall * 100))

    print("Best Train Recall: %.4f" % best_train_rec)
    print("Best Eval Recall: %.4f" % best_val_rec)
