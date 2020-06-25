import torch
import torch.optim as optim

CIFAR100_MILESTONES = [80, 120]
ImageNet_MILESTONES = [30, 60]


def get_optim_sche(lr, opt, net, dataset, momentum, weight_decay):
    if opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    MILESTONES = []
    gamma = 0
    if dataset == 'cifar100':
        MILESTONES = CIFAR100_MILESTONES
        gamma = 0.1
    elif dataset == 'imagenet':
        MILESTONES = ImageNet_MILESTONES
        gamma = 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=gamma, last_epoch=-1)
    return optimizer, scheduler
