## pytorch之cifar10分类任务

使用我自己写的CNNClassifier来进行cifar10分类任务。

### 项目配置

pytorch 1.0

python 3.5.6

需要安装argparse和torchvision

### 超参数设置

优化器使用SGD+momentum(0.9)

使用pretrainedmodels的权重作为初始权重。

lr调度：MultiStepLR。具体如下：

- 对于迁移学习：0-1:0.01， 2-9:0.001

- 对于非迁移学习：1-4:0.01,5-14:0.001,15-19:0.0001

### cifar10分类结果如下：

|                    model                    | epoch | acc     |
| :-----------------------------------------: | ----- | ------- |
|   AlexNet(transfer learning and finetune)   | 2+8   | 90.674% |
|                   AlexNet                   | 20    | 88.568% |
|  resnet18(transfer learning and finetune)   | 2+8   | 94.126% |
|              resnet18(common)               | 20    | 94.284% |
|  resnet34(transfer learning and finetune)   | 2+8   | 94.057% |
|              resnet34(common)               | 20    | 94.581% |
| densenet121(transfer learning and finetune) | 2+8   | 96.064% |
|                 densenet121                 | 20    | 96.618% |


