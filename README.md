# implementation-of-network-slimming
A reproduction of Learning Efﬁcient Convolutional Networks through Network Slimming

## Arguments  
- -net: net type, default='vgg19'
- -dataset: dataset, default='cifar100'
- -b: batch size for training, default=64
- -tb: batch size for testing, default=256
- -lr: initial learning rate, default=0.1
- -e: epoch, default=160
- -optim: optimizer, default="SGD"
- -momentum: SGD momentum, default=0.9
- -gpu: select GPU, default="0,1"
- -wd: weight decay, default=1e-4
- -l: lambda for sparsity, default=0.0001
- -percent: scale sparse rate, default=0.5
- -save: path to save model and training log, default='./log'  
- -trainflag: normal train or not, default=False
- -trainspflag: training with sparsity or not, default=False
- -retrainflag: retrain or not, default=False
- -resumeflag: resume training or not, default=False
- -pruneflag: prune or not, default=False

## Examples
__Tips:__ Please put your dataset in the _data_ folder or modify your path to dataset in _get_data.py_ before running the following code.  
### DenseNet-40 with growth rate 12:
#### baseline:
```
python train.py -trainflag -net densenet40
```
#### train with sparsity
```
python train.py -net densenet40 -trainspflag -l 0.00001
```

#### 40% pruned
```
python train.py -pruneflag -net densenet40 -percent 0.4
```

#### 40% pruned and fine-tune
```
python train.py -retrainflag -net densenet40
```

## Partial Results
model | params | FLOPs | best_top1 | best_top5 | inference time(ms)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------
DenseNet-40 baseline | 1.110M | 287.826M  | 74.880% | 94.080% | 0.3457976494639176
DenseNet-40 train with sparsity | 1.110M | 287.826M  | 74.760% | 94.000% | 0.28089947460222564
DenseNet-40 40% pruned  | 706.364K | 196.729M  | 72.960%| 93.110% | 0.27184
DenseNet-40 40% pruned and fine-tune | 706.364K | 196.729M  | 74.970%| 93.920% | 0.260371252737705
DenseNet-40 60% pruned  | 502.940K | 148.270M  | 42.320% | 71.390% | 0.31692
DenseNet-40 60% pruned and fine-tune  | 502.940K | 148.270M  | 74.670% | 94.170% | 0.285006638637279

## References
https://github.com/Eric-mingjie/network-slimming  
https://arxiv.org/pdf/1708.06519.pdf

