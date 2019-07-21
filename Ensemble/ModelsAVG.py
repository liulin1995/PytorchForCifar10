import torch as t
import numpy as np
from Utils import accuracy_topk
import time
import csv
t.backends.cudnn.benchmark = True


class ModelsAVG(object):

    def __init__(self, models, dataloader, device):
        if models is None or dataloader is None:
            raise ValueError('TheModels is none or dataloader is none')
        else:
            self._models = models
            self._dataloader = dataloader
        if device is None:
            self._device = t.device('cuda:0')
        else:
            self._device = device
        self._sample_nums = self._dataloader.batch_size * len(self._dataloader)
        self._accs = np.array([0.0, 0.0, 0.0])

    def run(self, num_classes=2019):
        for model in self._models:
            model.eval()
        with t.no_grad():
            for i, (inputs, label) in enumerate(self._dataloader, 1):
                inputs, label = inputs.to(self._device), label.long().squeeze(1).to(self._device)
                outputs = t.zeros([inputs.size()[0], num_classes]).to(self._device)
                for model in self._models:
                    outputs += model(inputs)
                res = accuracy_topk(outputs, label, (1, 2, 3))
                self._accs += res
        self._accs = self._accs / self._sample_nums
        return self._accs

    def test(self, test_loader, file_name):
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'predicted'])
        for model in self._models:
            model.eval()
        with t.no_grad():
            for i, (inputs, file_names) in enumerate(test_loader, 1):
                inputs = inputs.to(self._device)
                outputs = t.zeros([inputs.size()[0], 2019]).to(self._device)
                for model in self._models:
                    outputs += model(inputs)
                _, pred = outputs.topk(3)
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for ii in range(pred.size()[0]):
                        name = file_names[ii]
                        preds = '%d %d %d' % (pred[ii][0], pred[ii][1], pred[ii][2])
                        writer.writerow([name, preds])

    def save(self):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        file_name = time_str + '_ModelsAVGs.pkl'
        cp = {}
        for i, model in enumerate(self._models):
            cp[str(i)] = model.state_dict()
        cp['accs'] = self._accs
        t.save(cp, file_name)




