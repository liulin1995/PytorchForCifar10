
import os
import matplotlib.pyplot as plt

train_path = 'F:\\cvpr2019_kaggle_competition\\train'
val_path = 'F:\\cvpr2019_kaggle_competition\\val'

train_images = os.listdir(train_path)
val_imges = os.listdir(val_path)

train_nums = [0]*2019
val_nums = [0]*2019

for img in train_images:
    class_index = int(img.split('.')[-2].split('_')[-1])
    train_nums[class_index] += 1
max_num, min_num = max(train_nums), min(train_nums)
print('In Training Set:')
print('The max number class is: %d ' % max_num)
print('The min number class is: %d ' % min_num)
plt.plot(list(range(2019)), train_nums)
plt.show()
for img in val_imges:
    class_index = int(img.split('.')[-2].split('_')[-1])
    val_nums[class_index] += 1

max_num, min_num = max(val_nums), min(val_nums)
print('In Val Set:')
print('The max number class is: %d ' % max_num)
print('The min number class is: %d ' % min_num)

plt.plot(list(range(2019)), val_nums)
plt.show()