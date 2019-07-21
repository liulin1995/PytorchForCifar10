# coding: utf-8

import numpy as np
from PIL import Image
import random
import os
"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


IMG_TRAIN_PATH ='C:/cvpr2019_kaggle_competition/train'
CNum = 100000     # 挑选多少图片进行计算

img_h, img_w = 224, 224
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
img_dirs = os.listdir(IMG_TRAIN_PATH)
img_dirs = [os.path.join(IMG_TRAIN_PATH, img) for img in img_dirs]

random.shuffle(img_dirs)   # shuffle , 随机挑选图片

for i in range(CNum):
    img_path = img_dirs[i]
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_h, img_w))
    img = np.asarray(img)
    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)
    if i % 1000 == 0:
        print('[Reach]', str(i))

imgs = imgs.astype(np.float32)/255.
for i in range(3):
    pixels = imgs[:, :, i].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))