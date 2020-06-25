from models.select_channels import ChannelSelection
import torch.nn as nn
import torch
import math


# [conv1in, conv1out=conv2in, conv2out=conv3in] (conv3out=planes*expansion)
# cfg the channels after bn
defaultcfg = [[16, 16, 16], [64, 16, 16]*(18-1), [64, 32, 32], [128, 32, 32]*(18-1), [128, 64, 64],
              [256, 64, 64]*(18-1), [256]]

__all__ = ['resnet164']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride):
        super(Bottleneck, self).__init__()
        self.expansion = Bottleneck.expansion

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = ChannelSelection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * self.expansion, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Identity()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        return out


class ResNet(nn.Module):
    def __init__(self, num_class, cfg=None):
        super(ResNet, self).__init__()
        self.inplanes = 16
        if cfg is None:
            cfg = [item for sublist in defaultcfg for item in sublist]

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.stage1 = self.make_layers(Bottleneck, 16, 18, cfg[0: 18 * 3], 1)
        self.stage2 = self.make_layers(Bottleneck, 32, 18, cfg[18 * 3: 2 * 18 * 3], 2)
        self.stage3 = self.make_layers(Bottleneck, 64, 18, cfg[2 * 18 * 3: 3 * 18 * 3], 2)
        self.bn = nn.BatchNorm2d(64 * Bottleneck.expansion)
        self.select = ChannelSelection(64 * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg[-1], num_class)
        self._init_weights()

    def make_layers(self, Block, planes, blocks, cfg, stride):
        layers = []
        layers += [Block(self.inplanes, planes, cfg[0: 3], stride)]
        self.inplanes = planes * Block.expansion
        for i in range(1, blocks):
            layers += [Block(self.inplanes, planes, cfg[i * 3: i * 3 + 3], 1)]

        return nn.Sequential(*layers)

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
        out = self.conv1(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = self.bn(out)
        out = self.select(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet164(cfg=None, num_class=100):
    return ResNet(num_class, cfg)


# net = resnet164()
# x = torch.rand(1, 3, 32, 32)
# y = net(x)
# print(y.shape)
# for m in net.stage1[0].modules():
#     print(m)
# print(net.stage1[0])

