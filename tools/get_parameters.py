import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", default='vgg19', help='net type')
    parser.add_argument("-dataset", default='cifar100', help='dataset')
    parser.add_argument("-b", default=64, type=int, help='batch size for training')
    parser.add_argument("-tb", default=256, type=int, help='batch size for testing')
    parser.add_argument("-lr", default=0.1, help='initial learning rate', type=float)
    parser.add_argument("-e", default=160, help='EPOCH', type=int)
    parser.add_argument("-optim", default="SGD", help='optimizer')
    parser.add_argument('-momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument("-gpu", default="0,1", help='select GPU', type=str)
    parser.add_argument("-l", default=0.0001, help='lambda for sparsity', type=float)
    parser.add_argument('-percent', type=float, default=0.5, help='scale sparse rate')
    parser.add_argument('-save', default='./log', type=str, help='path to save model and training log')
    parser.add_argument("-retrainflag", action='store_true', help='retrain or not', default=False)
    parser.add_argument("-trainflag", action='store_true', help='normal train or not', default=False)
    parser.add_argument("-trainspflag", action='store_true', help='training with sparsity or not', default=False)
    parser.add_argument("-pruneflag", action='store_true', help='prune or not', default=False)
    parser.add_argument('-resumeflag', action='store_true', help='resume training or not', default=False)

    args = parser.parse_args()
    return args
