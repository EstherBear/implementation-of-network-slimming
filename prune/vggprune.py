import torch.nn as nn
import numpy as np
import torch
import models


def vgg19prune(net, percent):
    # global sort the bn scaling
    factors = []
    count_scaling = 0

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            for w in m.weight.data:
                factors.append(abs(w))
            count_scaling += m.weight.data.shape[0]
    value, idx = torch.sort(torch.from_numpy(np.array(factors, dtype=float)))

    # calculate the threshold
    threshold = value[int(count_scaling * percent)]

    # get cfg and cfg_mask for all layers
    cfg = []
    cfg_mask = []
    pruned = 0
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            mask = torch.gt(m.weight.data.abs().cpu(), threshold.float()).float().cuda()
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            pruned += mask.shape[0] - cfg[-1]
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    # get pruned_ratio
    ratio = float(pruned) / float(count_scaling)

    # make the real prune
    # print(cfg)
    new_net = models.__dict__['vgg19'](cfg=cfg)
    # print(net)
    # print(new_net)
    cfg_mask_index = 0
    last_mask = torch.tensor([1, 1, 1])
    next_mask = cfg_mask[cfg_mask_index]
    for [m0, m1] in zip(net.module.modules(), new_net.modules()):
        # get the index to prune for each bn(weight, bias, running_mean, running_var)
        if isinstance(m0, nn.BatchNorm2d):
            idx = np.squeeze(np.argwhere(np.array(next_mask.cpu())))
            if idx.size == 1:
                idx = np.expand_dims(idx, 0)
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            last_mask = next_mask
            cfg_mask_index += 1
            if cfg_mask_index < len(cfg_mask):
                next_mask = cfg_mask[cfg_mask_index]
        # prune conv(as the output of last bn and as the input of next bn)
        elif isinstance(m0, nn.Conv2d):
            idx1 = np.squeeze(np.argwhere(np.array(last_mask.cpu())))
            if idx1.size == 1:
                idx1 = np.expand_dims(idx1, 0)
            idx0 = np.squeeze(np.argwhere(np.array(next_mask.cpu())))
            if idx0.size == 1:
                idx0 = np.expand_dims(idx0, 0)
            w = m0.weight.data[idx0.tolist(), :, :, :].clone()
            m1.weight.data = w[:, idx1.tolist(), :, :].clone()
        # prune linear
        elif isinstance(m0, nn.Linear):
            idx = np.squeeze(np.argwhere(np.array(cfg_mask[-1].cpu())))
            if idx.size == 1:
                idx = np.expand_dims(idx, 0)
            m1.weight.data = m0.weight.data[:, idx.tolist()].clone()
            m1.bias.data = m0.bias.data.clone()

    print("prune {}%".format(ratio * 100))
    print(new_net)
    return cfg, new_net, net, ratio




