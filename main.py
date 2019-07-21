import torch as t
import csv
import torchvision
from torch.utils.data import DataLoader
from Utils import accuracy
from Trainer import CNNClassifier
from Data_Loader import transform_test, transform
from LR_Scheduler import find_learning_rate
from torchvision.models import AlexNet, vgg16
import matplotlib.pyplot as plt
import argparse
plt.interactive(False)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr2', default=0.1, type=float, help='learning rate for cylical and SGDR')
    parser.add_argument('--batchsize', default=128, type=int, help='training batchsize')
    parser.add_argument('--valbatchsize', default=128, type=int, help='val batchsize')
    parser.add_argument('--momentum',default=0.9,type=float,help='sgd momentum')
    parser.add_argument('--epoch',default=100, type=int, help="training epoches")
    parser.add_argument('--model', default='resnet18', type=str, help="model")

    args = parser.parse_args()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=args.valbatchsize, shuffle=False,
                            num_workers=4, pin_memory=False)
    lr = [args.lr2, args.lr]
    metrics = {
        'accuracy': accuracy,
    }

    lr_param = {'milestones':[5, 15]}
    optim_param = {'momentum':args.momentum}
    trainer = CNNClassifier(model=args.model, data_loader=train_loader, lr=lr, classes=10,
                      val_loader=val_loader, pretrained=True,
                      lr_scheduler='MultiStepLR', lr_step=50, weight_decay=5e-4,
                      metrics = metrics,lr_param=lr_param,
                      optimizer='SGD', optim_param=optim_param)

    # trainer.model_init()
    #trainer.freeze()

    trainer.train(args.epoch)
    #trainer.unfreeze(lr=[0, 0.001])
    #trainer.train(8)


if __name__ == '__main__':
    main()

