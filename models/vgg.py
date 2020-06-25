import torch.nn as nn
import torch
import math

__all__ = ['vgg19']

defaultCFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]


class vgg(nn.Module):
    def __init__(self, cfg=None, num_class=100):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultCFG
        self.classifier = nn.Linear(cfg[-1], num_class)
        self.features = self.make_layers(cfg)
        self._init_weights()

    def make_layers(self, cfg):
        in_channels = 3
        layers = []
        for c in cfg:
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=3, padding=1, stride=1,
                                     bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                in_channels = c
        layers += [nn.AdaptiveAvgPool2d(1)]
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
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def vgg19(cfg=None, num_class=100):
    return vgg(cfg, num_class)


if __name__ == '__main__':
    net = vgg19()
    x = torch.rand(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
