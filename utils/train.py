import torch
from .misc import AverageMeter


def mine(train_loader, model, criterion, criterion_kl, optimizer, epoch, init_tick):
    model.train()

    accTop1_avg = list(range(args.num_branches+1))
    accTop5_avg = list(range(args.num_branches+1))
    for i in range(args.num_branches+1):
        accTop1_avg[i] = AverageMeter()
        accTop5_avg[i] = AverageMeter()
    losses = AverageMeter()
    #losses_coeff = AverageMeter()
    #losses_origin = AverageMeter()
    losses_kd = AverageMeter()
    losses_corr = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))
    consistency_weight = get_current_consistency_weight(epoch)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(device), targets.cuda(device)
        
        outputs, out_t, coeff = model(inputs) # coeff.size() = [BatchSize, n_heads]
        loss_cross = 0
        #loss_coeff = 0
        #loss_origin = 0
        loss_kd = 0
        loss_corr = 0
        #min_ce = []
        '''
        for i in range(args.num_branches-1):
            # until recent - this is sota
            loss_cross += criterion(outputs[:,:,i], targets) # Peers <--> Target CE included student

            # new trial <---- unstable!!!
            #print(criterion(outputs[:,:,i], targets), coeff[:,i])
            #print(coeff[:,i] * criterion(outputs[:,:,i], targets))
            #loss_1 = consistency_weight * (coeff[:,i] * criterion(outputs[:,:,i] , targets)).mean()
            #loss_2 = (1-consistency_weight) * criterion(outputs[:,:,i], targets).mean()
            #loss_coeff += loss_1
            #loss_origin += loss_2
            #loss_cross +=  loss_1 + loss_2 # (BatchSize, 1) per one head
            #loss_cross += coeff[:,i].mean() * criterion(outputs[:,:,i], targets)
        loss_cross += criterion(out_t, targets).mean() # Ensemble <--> Target CE
        loss_cross += criterion(outputs[:,:,-1], targets).mean() # Student <--> Target CE
        '''
        for i in range(args.num_branches):
            loss_cross += criterion(outputs[:,:,i], targets)
            loss_kd += criterion_kl(outputs[:,:,i], out_t)
            #loss_cross += (consistency_weight * coeff[:,i] * criterion(outputs[:,:,i], targets)).mean() + (1-consistency_weight * criterion(outputs[:,:,i], targets)).mean()
            #min_ce.append((coeff[:,i] * criterion(outputs[:,:,i], targets)).mean())
        loss_cross += criterion(out_t, targets) # Ensemble <--> Target CE
        #if args.rkd:
        #    loss_kd += criterion_rl(outputs[:,:,-1], out_t) # Student <--> Ensemble RKD
        loss_kd = consistency_weight * loss_kd # Distillation Loss
        '''
        min_ce_index = min_ce.index(min(min_ce))
        print(min_ce_index)
        for i in range(args.num_branches):
            if i != min_ce_index:
                loss_kd += criterion_kl(outputs[:,:,i], outputs[:,:,min_ce_index])
            else:
                pass
        '''
        ## forced diversity
        if args.fd:
            loss_corr = covariance_loss(outputs[:,:,:-1], targets, Temperature)

        loss = loss_cross + loss_kd + loss_corr

        losses_kd.update(loss_kd.data, inputs.size(0))
        #losses_kd.update(0, inputs.size(0))
        losses_corr.update(loss_corr.data, inputs.size(0)) if args.fd else losses_corr.update(0, inputs.size(0))
        #losses_coeff.update(loss_coeff.data, inputs.size(0))
        #losses_origin.update(loss_origin.data, inputs.size(0))
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
        show_metrics.update({'Loss': losses.avg, 'LossKD': losses_kd.avg, 'LossCorr': losses_corr.avg})
        #show_metrics.update({'Loss': losses.avg, 'LossCoeff': losses_coeff.avg, 'LossOrigin': losses_origin.avg, 'LossKD': losses_kd.avg})
        for i in range(args.num_branches+1):
            if i == args.num_branches-1:
                show_metrics.update({'Top1_stu': accTop1_avg[i].avg})
            elif i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})
        '''
        for i in range(args.num_branches+1):
            if i == args.num_branches:
                show_metrics.update({'Top1_en': accTop1_avg[i].avg})
            else:
                show_metrics.update({'Top1_C'+str(i): accTop1_avg[i].avg})
        '''
        bar.suffix = " | ".join("{}: {:.4f}".format(k, v) for k, v in show_metrics.items())
        bar.next()
        throughput = speedometer(batch_idx+1, epoch, init_tick)
        mem_alloc = memhooker(batch_idx+1)
    bar.finish()
    return show_metrics, throughput, mem_alloc
