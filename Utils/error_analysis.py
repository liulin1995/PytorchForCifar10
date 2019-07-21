import torch as t



class ErrorAnalysis(object):

    def __init__(self):
        self.model = None
        self.device = None
        self.num_classes = None
        self.dataloader = None

    def get_confusion_matrix(self):
        confusion_matrix = t.zeros(self.n_classes, self.n_classes)
        with t.no_grad():
            for i, (inputs, classes) in enumerate(self.dataloaders):
                inputs = inputs.to(self.device)
                classes = classes.to(self.device)
                outputs = self.model(inputs)
                _, preds = t.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)

    def sample_num_each_class(self):
        num_arr = t.zeros(self.num_classes,1)
        for i, (inputs, labels) in enumerate(self.dataloaders):
            labels = labels.to(self.device)
            for label in labels:
                num_arr[label] += 1

        return num_arr

    def sample_loss(self):
        losses = t.zeros(len(self.dataloader)*self.dataloader.batchsize, 1)
