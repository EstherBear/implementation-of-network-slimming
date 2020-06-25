import models
from thop import profile
from thop import clever_format
import torch


def get_flops_params(net, dataset):
    if dataset == 'cifar100':
        x = torch.randn(1, 3, 32, 32)
    elif dataset == "imagenet":
        x = torch.randn(1, 3, 224, 224)
    else:
        print("The net is not provided.")
        exit(0)

    macs, params = profile(model=net, inputs=(x,))
    flops, params = clever_format([macs, params], "%.3f")
    print("flops and params: ", flops, params)
    return flops, params
    # summary(net.cuda(), (3, 32, 32))


# net = MyResNet34()
# f, p = get_flops_params(net, "resnet34")
