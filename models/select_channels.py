import torch.nn as nn
import torch
import numpy as np

__all__= ['ChannelSelection']


class ChannelSelection(nn.Module):
    def __init__(self, num_channels):
        super(ChannelSelection, self).__init__()
        self.index = nn.Parameter(torch.ones(num_channels))  # to calculate params

    def forward(self, x):
        idx = np.squeeze(np.argwhere(self.index.data.cpu().numpy()))
        if idx.size == 1:
            idx = np.expand_dims(idx, axis=0)
        out = x[:, idx, :, :]
        return out
