import torch.nn as nn
import torch
import os
import shutil

import datetime
import torch.cuda
import models
from prune import prune_net

from tools.optim_sche import get_optim_sche, CIFAR100_MILESTONES
from tools.get_data import get_train_loader, get_test_loader
from tools.get_parameters import get_args
from tools.flops_params import get_flops_params


def UpdateBnFactor():
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.l * torch.sign(m.weight.data))


def train_epoch(fepoch, epoch):
    net.train()

    length = len(train_loader)
    total_sample = len(train_loader.dataset)
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    batch_size = 0

    for step, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        if step == 0:
            batch_size = len(y)
        optimizer.zero_grad()

        output = net(x)
        loss = loss_function(output, y)
        loss.backward()

        if args.trainspflag:
            UpdateBnFactor()

        optimizer.step()

        total_loss += (loss.item()/length)
        _, predict = output.topk(5, 1, True, True)
        # _, predict = torch.max(output, 1)
        predict = predict.t()
        correct = predict.eq(y.view(1, -1).expand_as(predict))
        correct_1 += float(correct[:1].view(-1).sum())/total_sample
        correct_5 += float(correct[:5].view(-1).sum())/total_sample
        # correct += (predict == y).sum()

        if step % 20 == 0:
            print("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}".format(
                epoch + 1, step + 1, step * batch_size + len(y), total_sample, loss.item()
            ))

    fepoch.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc1:{:.3%}\t acc5:{:.3%}\n".format(
        epoch + 1, total_loss, optimizer.param_groups[0]['lr'], correct_1, correct_5
    ))
    if args.trainspflag:
        sum_scaling = 0
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                sum_scaling += torch.sum(torch.FloatTensor.abs(m.weight.data).view(-1))
        fepoch.write("Scaling:{}\n".format(sum_scaling))
        print("Scaling:{}".format(sum_scaling))

    fepoch.flush()
    return net


def eval_epoch(tnet):
    tnet.eval()

    length = len(test_loader)
    total_sample = len(test_loader.dataset)
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    inference_time = 0

    for step, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = tnet(x)
            # _, predict = torch.max(output, 1)
            _, predict = output.topk(5, 1, True, True)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            inference_time += (start.elapsed_time(end) / len(test_loader.dataset))  # milliseconds

            loss = loss_function(output, y)
            total_loss += (loss.item() / length)

            predict = predict.t()
            correct = predict.eq(y.view(1, -1).expand_as(predict))
            correct_1 += float(correct[:1].view(-1).sum()) / total_sample
            correct_5 += float(correct[:5].view(-1).sum()) / total_sample
            # correct += (predict == y).sum()

    acc1 = correct_1
    acc5 = correct_5
    return acc1, acc5, total_loss, inference_time


def training():
    global best_acc
    total_time = 0
    is_best = False

    if not os.path.exists(train_checkpoint_path):
        os.makedirs(train_checkpoint_path)

    with open(os.path.join(train_checkpoint_path, 'EpochLog.txt'), 'w') as fepoch:
        with open(os.path.join(train_checkpoint_path, 'EvalLog.txt'), 'w') as feval:
            with open(os.path.join(train_checkpoint_path, 'Best.txt'), 'w') as fbest:
                print("start training")

                for epoch in range(start_epoch, args.e):
                    train_epoch(fepoch, epoch)
                    print("evaluating")
                    accuracy1, accuracy5, averageloss, inference_time = eval_epoch(net)
                    feval.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc1:{:.3%}\t acc5:{:.3%}\n".format(
                        epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy1, accuracy5
                    ))
                    feval.flush()
                    if scheduler is not None:
                        scheduler.step()

                    if accuracy1 > best_acc:
                        best_acc = accuracy1
                        is_best = True
                    else:
                        is_best = False

                    save_dict = {
                        'start_epoch': epoch + 1,
                        'model_state_dict': net.state_dict(),
                        'best_acc': best_acc,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    print("saving regular")
                    torch.save(save_dict, os.path.join(train_checkpoint_path, 'regularParam.pth'))
                    shutil.copyfile(os.path.join(train_checkpoint_path, 'regularParam.pth'),
                                        os.path.join(most_recent_path, 'regularParam.pth'))
                    

                    if is_best:
                        print("saving best")
                        shutil.copyfile(os.path.join(train_checkpoint_path, 'regularParam.pth'),
                                        os.path.join(train_checkpoint_path, 'bestParam.pth'))
                        shutil.copyfile(os.path.join(train_checkpoint_path, 'regularParam.pth'),
                                        os.path.join(most_recent_path, 'recent.pth'))
                        fbest.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc1:{:.3%}\t acc5:{:.3%}\n".format(
                            epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy1, accuracy5
                        ))
                        fbest.flush()

                    # print(inference_time)
                    total_time += inference_time

    print(total_time)
    print(total_time / epoch)


if __name__ == '__main__':
    # arguments from command line
    args = get_args()

    # data processing and prepare for training
    train_loader = get_train_loader(args)
    test_loader = get_test_loader(args)
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.
    start_epoch = 0

    # define checkpoint path
    time = str(datetime.date.today() + datetime.timedelta(days=1))
    checkpoint_path = os.path.join(args.save, args.net)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    folder_name = None
    if args.trainflag:
        folder_name = "train"
    elif args.retrainflag:
        folder_name = "retrain"
    elif args.pruneflag:
        folder_name = "prune"
    else:
        folder_name = "trainsp"
    train_checkpoint_path = os.path.join(checkpoint_path, folder_name, time)
    most_recent_path = os.path.join(checkpoint_path, folder_name)

    # define gpus and get net
    device_ids = [int(i) for i in list(args.gpu.split(','))]
    net = None

    if args.retrainflag:
        checkpoint = torch.load(os.path.join(os.path.join(checkpoint_path, 'prune'), 'prunedParam.pth'))
        net = models.__dict__[args.net](cfg=checkpoint['cfg'])
        net = nn.DataParallel(net, device_ids=device_ids)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net = models.__dict__[args.net](cfg=None)
        net = nn.DataParallel(net, device_ids=device_ids)
        if args.pruneflag:
            net.load_state_dict(torch.load(os.path.join(os.path.join(checkpoint_path, 'trainsp'), 'recent.pth'))
                                ['model_state_dict'])
    net = net.cuda()
    # get optimizer and scheduler
    optimizer, scheduler = get_optim_sche(args.lr, args.optim, net, args.dataset, args.momentum, args.wd)

    if args.resumeflag:
        print('load checkpoint to resume')
        checkpoint = torch.load(os.path.join(most_recent_path, 'regularParam.pth'))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['start_epoch']
        best_acc = checkpoint['beat_acc']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=CIFAR100_MILESTONES, gamma=0.1,
                                                         last_epoch=start_epoch-1)

    # prune
    if args.pruneflag:
        if not os.path.exists(train_checkpoint_path):
            os.makedirs(train_checkpoint_path)
        cfg, new_net, net, ratio = prune_net(net, args.percent, args.net)
        new_net = nn.DataParallel(new_net, device_ids=device_ids)
        new_net = new_net.cuda()
        # test the net after pruning(before real prune)
        accuracy1, accuracy5, averageloss, inference_time = eval_epoch(net)
        print("before real prune: Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t infer:{:.5f}\n".format(
            averageloss, accuracy1, accuracy5, inference_time))

        # get param, flops and acc for pruned net
        accuracy1, accuracy5, averageloss, inference_time = eval_epoch(new_net)
        print("real prune: Loss:{:.3f}\t acc1:{:.3%}\t acc5:{:.3%}\t infer:{:.5f}\n".format(
            averageloss, accuracy1, accuracy5, inference_time))
        f, p = get_flops_params(new_net.module.cpu(), args.dataset)
        new_net = new_net.cuda()
        # save(cfg, state_dict)
        with open(os.path.join(train_checkpoint_path, 'flops_and_params'), 'w') as fp:
            fp.write("flops:{}\t params:{}\t ratio:{:.3f}\n".format(f, p, ratio))
            fp.flush()
        save_prune_dict = {
            'model_state_dict': new_net.state_dict(),
            'cfg': cfg
        }
        torch.save(save_prune_dict, os.path.join(train_checkpoint_path, 'prunedParam.pth'))
        shutil.copyfile(os.path.join(train_checkpoint_path, 'prunedParam.pth'),
                        os.path.join(most_recent_path, 'prunedParam.pth'))
    else:
        f, p = get_flops_params(net.module.cpu(), args.dataset)
        net = net.cuda()
        with open(os.path.join(checkpoint_path, 'flops_and_params'), 'w') as fp:
            fp.write("flops:{}\t params:{}\n".format(f, p))
            fp.flush()
        training()
