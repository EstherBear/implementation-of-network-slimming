from models.select_channels import ChannelSelection
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
# GrowthRate, CompressionRate, DropRate

__all__ = ['densenet40']
GrowthRate = 12
cfg_list = []
StartRate = 2 * GrowthRate
for _ in range(3):
    cfg_list.append([StartRate + i * GrowthRate for i in range(12+1)])
    StartRate += 12 * GrowthRate
DefaultCfg = [item for sublist in cfg_list for item in sublist]


class BasicBlock(nn.Module):
    def __init__(self, inplanes, cfg, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.select = ChannelSelection(inplanes)
        self.conv = nn.Conv2d(in_channels=cfg, out_channels=GrowthRate, kernel_size=3, padding=1, stride=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.bn(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv(out)

        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = torch.cat((out, x), dim=1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, cfg, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.select = ChannelSelection(inplanes)
        self.conv = nn.Conv2d(in_channels=cfg, out_channels=outplanes, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv(out)
        out = F.avg_pool2d(out, kernel_size=2)

        return out


class DenseNet(nn.Module):
    def __init__(self, cfg=None, compression_rate=1.0, droprate=0.0, num_class=100):
        super(DenseNet, self).__init__()
        self.inplanes = 2 * GrowthRate
        self.compression_rate = compression_rate
        self.drop_rate = droprate

        if cfg is None:
            cfg = DefaultCfg

        self.conv = nn.Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.dense1 = self._make_denseblock(12, cfg[0: 12 * 1])
        self.trans1 = self._make_transition(cfg[12])
        self.dense2 = self._make_denseblock(12, cfg[12 + 1: 12 * 2 + 1])
        self.trans2 = self._make_transition(cfg[12 * 2 + 1])
        self.dense3 = self._make_denseblock(12, cfg[12 * 2 + 2: 12 * 3 + 2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.select = ChannelSelection(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg[-1], num_class)

        self._init_weights()

    def _make_denseblock(self, blocks, cfg):
        layers = []
        for i in range(blocks):
            layers += [BasicBlock(self.inplanes, cfg[i], self.drop_rate)]
            self.inplanes += GrowthRate
        return nn.Sequential(*layers)

    def _make_transition(self, cfg):
        inplanes = self.inplanes
        self.inplanes = int(self.inplanes // self.compression_rate)
        return Transition(inplanes, cfg, self.inplanes)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        out = self.bn(out)
        out = self.select(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out


def densenet40(cfg=None, num_class=100):
    return DenseNet(cfg=cfg, num_class=num_class)


