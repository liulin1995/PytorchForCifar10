import time
import torch as t


def accuracy_topk(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
        # res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_confusion_matrix(model, n_classes, dataloaders, device):
    confusion_matrix = t.zeros(n_classes, n_classes)
    with t.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = t.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)


def time_logger(*args):
    ime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    ime_str += '  '
    for v in args:
        ime_str += str(v)
    with open('time_log.log', 'a+') as logger:
        logger.write(ime_str + '\n')
    print(ime_str)


def get_pytorch_version():
    print(t.__version__ ) # PyTorch version
    print(t.version.cuda)  # Corresponding CUDA version
    print(t.backends.cudnn.version())  # Corresponding cuDNN version
    print(t.cuda.get_device_name(0) ) # GPU type


def guding_seed(seed=0):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
