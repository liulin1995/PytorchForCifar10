from torchvision import models
from .TheModels import se_resnext101_32x4d,se_resnet101, se_resnext50_32x4d,senet154
import torch as t
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

'''
Network	Top-1 error	Top-5 error model_size(MB)
ResNet-18	30.24	10.92       
ResNet-34	26.70	8.58         85
ResNet-50	23.85	7.13         100
ResNet-101	22.63	6.44         174
ResNet-152	21.69	5.94         235
Inception v3	22.55	6.44     106
AlexNet	43.45	20.91
VGG-11	30.98	11.37
VGG-13	30.07	10.75
VGG-16	28.41	9.62
VGG-19	27.62	9.12
SqueezeNet 1.0	41.90	19.58
SqueezeNet 1.1	41.81	19.38
Densenet-121	25.35	7.83     31
Densenet-169	24.00	7.00     56
Densenet-201	22.80	6.43     79
Densenet-161	22.35	6.20     113

'''


class MyPretrainModels(object):
    def __init__(self, num_classes, device):
        self._num_classes = num_classes
        self._device = device

    def get_vgg16(self, pretrained=True, file_path=None):
        model = models.vgg16(pretrained=pretrained)
        in_features = model.classifier[-1].in_features

        # model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.classifier[-1] = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_vgg19(self, pretrained=True, file_path=None):
        model = models.vgg19(pretrained=pretrained)
        in_features = model.classifier[-1].in_features

        # model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.classifier[-1] = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_alexnet(self, pretrained=True, file_path=None):
        model = models.alexnet(pretrained=pretrained)
        in_features = model.classifier[-1].in_features

        # model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.classifier[-1] = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_resnet18(self, pretrained=True, file_path=None):
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features

        # model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.fc = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_resnet34(self, pretrained=True, file_path=None):
        model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.fc = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_resnet50(self, pretrained=True, file_path=None):
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.fc = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_resnet101(self, pretrained=True, file_path=None):
        model = models.resnet101(pretrained=pretrained)
        in_features = model.fc.in_features
        model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.fc = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_resnet152(self, pretrained=True, file_path=None):
        model = models.resnet152(pretrained=pretrained)
        in_features = model.fc.in_features
        model.avgpool = t.nn.AdaptiveAvgPool2d(1)
        model.fc = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_densenet121(self, pretrained=True, file_path=None):
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_densenet161(self, pretrained=True, file_path=None):
        model = models.densenet161(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_densenet169(self, pretrained=True, file_path=None):
        model = models.densenet169(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_densenet201(self, pretrained=True, file_path=None):
        model = models.densenet201(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_inceptionv3(self, pretrained=True, file_path=None):
        model = models.inception_v3(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = t.nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        if file_path:
            model.load_state_dict(t.load(file_path))
        return model

    def get_se_resnext101_32x4d(self, pretrained=True, file_path=None):
        if pretrained:
            model = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model
        else:
            model = se_resnext101_32x4d(num_classes=1000, pretrained=None)
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model

    def get_se_resnet101(self, pretrained=True, file_path=None):
        if pretrained:
            model = se_resnet101(num_classes=1000, pretrained='imagenet')
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model
        else:
            model = se_resnet101(num_classes=1000, pretrained=None)
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model

    def get_se_resnext50_32x4d(self, pretrained=True, file_path=None):
        if pretrained:
            model = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model
        else:
            model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model

    def get_senet154(self, pretrained=True, file_path=None):
        if pretrained:
            model = senet154(num_classes=1000, pretrained='imagenet')
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model
        else:
            model = senet154(num_classes=1000, pretrained=None)
            in_features = model.last_linear.in_features
            model.last_linear = t.nn.Linear(in_features, self._num_classes)
            model.to(self._device)
            if file_path:
                model.load_state_dict(t.load(file_path))
            return model





