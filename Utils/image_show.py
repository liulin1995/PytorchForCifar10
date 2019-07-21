import PIL
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

mean_imagenet = [0.4914, 0.4822, 0.4465]
std_imagenet = [0.2023, 0.1994, 0.2010]

def imshow(image):
    mean = t.Tensor(mean_imagenet).view(3,1,-1)
    std = t.Tensor(std_imagenet).view(3,1,-1)
    image = (image*std + mean)
    image = t.clamp(image, 0, 1).cpu().numpy()

    plt.imshow(np.transpose(image, (1,2,0)))
    plt.show()

