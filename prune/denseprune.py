import torch.nn as nn
from models.select_channels import ChannelSelection
import numpy as np
import torch
import models


def densenet40prune(net, percent):
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

    # get pruned_ratio
    ratio = float(pruned) / float(count_scaling)

    # make the real prune
    new_net = models.__dict__['densenet40'](cfg=cfg)
    m0_list = list(net.module.modules())
    m1_list = list(new_net.modules())
    cfg_mask_index = 0
    is_first = True
    last_mask = None
    next_mask = cfg_mask[cfg_mask_index]
    for module_index in range(len(m0_list)):
        m0 = m0_list[module_index]
        m1 = m1_list[module_index]
        # get the index to prune for each bn
        if isinstance(m0, nn.BatchNorm2d):
            idx = np.squeeze(np.argwhere(np.array(next_mask.cpu())))
            if idx.size == 1:
                idx = np.expand_dims(idx, 0)
            # prune channel_selection following bn
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # modify ChannelSelection
            # print(m1_list[module_index + 1].index.data.shape, idx.shape)
            m1_list[module_index + 1].index.data.zero_()
            m1_list[module_index + 1].index.data[idx] = 1.0
            # print(m1_list[module_index + 1].index.data.shape, idx.shape)

            # update masks
            last_mask = next_mask
            cfg_mask_index += 1
            if cfg_mask_index < len(cfg_mask):
                next_mask = cfg_mask[cfg_mask_index]

        # prune conv(as the output of last bn)
        elif isinstance(m0, nn.Conv2d):
            # not prune first conv
            if is_first:
                m1.weight.data = m0.weight.data.clone()
                is_first = False
                continue

            idx1 = np.squeeze(np.argwhere(np.array(last_mask.cpu())))
            if idx1.size == 1:
                idx1 = np.expand_dims(idx1, 0)
            w = m0.weight.data[:, idx1.tolist(), :, :].clone()
            m1.weight.data = w.clone()

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
