# Paper Reproduce [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

Official PyTorch implementation of **ConvNeXt**, from the following paper:

[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR 2022.\
[Zhuang Liu](https://liuzhuang13.github.io), [Hanzi Mao](https://hanzimao.me/), [Chao-Yuan Wu](https://chaoyuan.org/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) and [Saining Xie](https://sainingxie.com)\
Facebook AI Research, UC Berkeley

--- 

## Usage
```
python main.py --epochs 100 --model ResNet_v14 --data_set CIFAR --data_path ./data --warmup_epochs 0 --nb_classes 100 --cutmix 0 --mixup 0 --lr 4e-4 --enable_wandb true --wandb_ckpt true
```
