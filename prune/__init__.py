from prune.vggprune import vgg19prune
from prune.resnetprune import resnet164prune
from prune.denseprune import densenet40prune


def prune_net(net, percent, net_name):
    return eval(net_name + "prune")(net, percent)
