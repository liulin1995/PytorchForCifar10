import torch as t
import csv
from torch.nn import init
import math
import time
from Utils import accuracy_topk, time_logger
from LR_Scheduler import CyclicLR, CosineAnnealingWarmRestarts
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from collections import Iterable
import os
from tensorboardX import SummaryWriter
from Pretrained_Models import MyPretrainModels
writer = SummaryWriter('./runs/cifar10')  # tensorboard --logdir runs


class CNNClassifier(object):

    def __init__(self,  lr, classes, data_loader=None, model='resnet18',  metrics = {},
                 lr_scheduler='Step_LR', lr_step = 5,  lr_param = {}, val_loader = None,
                 criterion=None, pretrained=True, weight_decay = 0, checkpoint_directory="./checkpoint",
                 optimizer='sgd', optim_param={}, use_cuda=True):

        if use_cuda:
            self.device = t.device('cuda:0')
        else:
            self.device = t.device('cpu')

        if classes >0 and isinstance(classes, int):
            self.classes = classes

        if model in ['vgg16','vgg19','AlexNet','resnet18','resnet34','resnet50', 'resnet101', 'densenet121', 'densenet161', 'se_resnext101_32x4d', 'se_resnet101']:
            my_models = MyPretrainModels(self.classes, device=self.device)
            if model == 'vgg16':
                self.model = my_models.get_vgg16(file_path=None, pretrained=pretrained)
            if model == 'vgg19':
                self.model = my_models.get_vgg19(file_path=None, pretrained=pretrained)
            if model == 'AlexNet':
                self.model = my_models.get_alexnet(file_path=None, pretrained=pretrained)
            if model == 'resnet18':
                self.model = my_models.get_resnet18(file_path=None, pretrained=pretrained)
            elif model == 'resnet34':
                self.model  = my_models.get_resnet18(file_path=None, pretrained=pretrained)
            elif model == 'resnet50':
                self.model = my_models.get_resnet50(file_path=None, pretrained=pretrained)
            elif model == 'resnet101':
                self.model = my_models.get_resnet101(file_path=None, pretrained=pretrained)
            elif model == 'densenet121':
                self.model = my_models.get_densenet121(file_path=None, pretrained=pretrained)
            elif model == 'densenet161':
                self.model = my_models.get_densenet161(file_path=None, pretrained=pretrained)
            elif model == 'se_resnext101_32x4d':
                self.model = my_models.get_se_resnext101_32x4d(file_path=None, pretrained=pretrained)
            elif model == 'se_resnet101':
                self.model = my_models.get_se_resnet101(file_path=None, pretrained=pretrained)
        else:
            raise ValueError("the model should in ['vgg16','vgg19','AlexNet','resnet18','resnet34','resnet50', "
                             "'resnet101', 'densenet121', "
                             "'densenet161', 'se_resnext101_32x4d', 'se_resnet101']")

        self.lr_param = lr_param
        self.optim_param = optim_param
        if isinstance(lr, list) or isinstance(lr, tuple):
            if len(lr) != 2:
                raise ValueError("expected a list {} , got {}".format(2, len(lr)))
            self.lr = lr
        self.lr_step = lr_step

        if data_loader is not None:
            self.loader = data_loader
        self.weight_decay = weight_decay
        if optimizer is None: optimizer='SGD'
        if optimizer in ['SGD','Adam']:
            bias, weights = [], []
            for name, weight in self.model.named_parameters():
                if 'bias' in name or 'bn' in name:
                    bias.append(weight)
                else:
                    weights.append(weight)
            weight_params = [{'params': weights, 'weight_decay': weight_decay}, {'params': bias}]
            if optimizer == 'SGD':
                self.optimizer = t.optim.SGD(weight_params,  lr=self.lr[1], **self.optim_param)
            elif optimizer == 'Adam':
                self.optimizer = t.optim.Adam(weight_params, lr=self.lr[1], **self.optim_param)
        else:
            raise ValueError("the optimizer should in ['sgd','Adam']")

        if lr_scheduler is None: lr_scheduler="SGDR"
        if lr_scheduler in ['Step_LR','SGDR','CyclicLR', 'MultiStepLR']:
            if lr_scheduler == 'SGDR':
                self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=lr_step, eta_min=lr[0], **lr_param)
            elif lr_scheduler == 'CyclicLR':
                self.scheduler = CyclicLR(self.optimizer, base_lr=self.lr[0], max_lr=self.lr[1], step_size=lr_step*self.loader.batch_size)
            elif lr_scheduler == 'Step_LR':
                self.scheduler = StepLR(self.optimizer, step_size=lr_step, **lr_param)
            elif lr_scheduler == 'MultiStepLR':
                self.scheduler = MultiStepLR(self.optimizer, **lr_param)
        else:
            raise ValueError("the lr_schedulear should in ['Step_LR','SGDR','CyclicLR','MultiStepLR']")

        if criterion is None: criterion = 'CrossEntropy'
        if criterion in ['CrossEntropy', 'MSE']:
            if criterion == 'CrossEntropy':
                self.criterion = t.nn.CrossEntropyLoss()
            elif criterion == 'MSE':
                self.criterion = t.nn.MSELoss()
        else:
            raise ValueError("the criterion should in ['CrossEntropy', 'MSE']")

        self.metrics = metrics
        self.val_loader = val_loader
        self.checkpoint_directory = checkpoint_directory

    def freeze_to(self, n: int) -> None:
        "Freeze layers up to layer group `n`."

        for g in list(self.model.children())[:n]:
            for l in g.parameters():
                l.requires_grad = False
        for g in list(self.model.children())[n:]:
            for l in g.parameters():
                l.requires_grad = True

    def reset_lr(self, lr=None):
        if lr is None:
            lr = self.lr
        else:
            if isinstance(self.optimizer, t.optim.SGD):
                self.optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr[1], **self.optim_param)
            elif isinstance(self.optimizer, t.optim.Adam):
                self.optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr[1], **self.optim_param)

            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.lr_step, eta_min=self.lr[0],
                                                              **self.lr_param)
            elif isinstance(self.scheduler, CyclicLR):
                self.scheduler = CyclicLR(self.optimizer, base_lr=self.lr[0], max_lr=self.lr[1],
                                          step_size=self.lr_step * self.loader.bathsize, **self.lr_param)
            elif isinstance(self.scheduler, StepLR):
                self.scheduler = StepLR(self.optimizer, step_size=self.lr_step,  **self.lr_param)

    def freeze(self, lr=None):
        self.freeze_to(-1)
        self.reset_lr(lr)

    def unfreeze(self, lr=None):
        self.freeze_to(0)
        self.reset_lr(lr)

    def train(self, Epochs=1):
        '''
        train procedure of the trainer
        '''
        time_logger('Train BEGIN')
        self.model.train()
        best_acc = 0.0

        for epoch in range(Epochs):
            # call lr_scheduler
            if self.scheduler is not None and isinstance(self.scheduler, (StepLR, CosineAnnealingWarmRestarts)):
                self.scheduler.step()
            summ = []
            for i, (inputs, label) in enumerate(self.loader, 1):

                if self.scheduler is not None and isinstance(self.scheduler, CyclicLR):
                    self.scheduler.batch_step()

                # optimizer procedure
                inputs, label = inputs.to(self.device), label.long().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                iter_loss = self.criterion(outputs, label)
                iter_loss.backward()
                self.optimizer.step()

                # compute all metrics on this batch

                summary_batch = {metric: self.metrics[metric](outputs, label) for metric in self.metrics}
                summary_batch['loss'] = iter_loss.item()
                summ.append(summary_batch)

            metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            time_logger('[Epoch:{}]'.format(epoch))
            time_logger('[Train] ',metrics_string)
            writer.add_scalar('train_loss', metrics_mean['loss'], epoch)

            # validate
            val_metrics = self.evaluate(self.val_loader)

            metrics_string = " ; ".join("{}:  {:04.3f}".format(k, v) for k, v in val_metrics.items())
            time_logger('[Valid] ',metrics_string)
            writer.add_scalar('val_loss', val_metrics['loss'], epoch)
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                self.save(self.checkpoint_directory, 'best.pth')
                time_logger('saving a checkpoint.')

    def evaluate(self, dataloader):
        '''
        for input type is iterable
        :return: val_loss, val_acc
        '''
        if dataloader is None or not isinstance(dataloader, Iterable):
            raise ValueError('The data_loader is None or is not Iterable')

        summ = []
        self.model.eval()
        with t.no_grad():

            for i, (inputs, label) in enumerate(dataloader, 1):
                inputs, label = inputs.to(self.device), label.long().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, label)

                summary_batch = {metric: self.metrics[metric](outputs, label) for metric in self.metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
        return metrics_mean

    def save(self, output_path, file_name):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        file_path = os.path.join(output_path, file_name)
        t.save(self.model.state_dict(), file_path)

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError('The given path is not exists.')
        state_dict = t.load(model_path)
        self.model.load_state_dict(state_dict)

    def model_init(self):
        for m in self.model.modules():
            if isinstance(m, t.nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, t.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, t.nn.Linear):
                init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


