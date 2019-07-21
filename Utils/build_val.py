import shutil
import random
import os
from Data_Config import data_config
train_root = data_config.IMG_TRAIN_PATH
val_root = data_config.IMG_VAL_PATH

if __name__ == '__main__':
    image_files = os.listdir(train_root)
    dog_files = list(filter(lambda p: 'dog' in p, image_files))
    cat_files = list(filter(lambda p: 'dog' not in p, image_files))
    print(len(dog_files),len(cat_files))
    random.shuffle(dog_files)
    random.shuffle(cat_files)

    for i in range(len(dog_files)):
        pic_name = dog_files[i].split('\\')[-1]
        if i > 0.7 * len(dog_files):
            obj_file = os.path.join(val_root, pic_name)
            shutil.move(os.path.join(train_root,dog_files[i]), obj_file)
            print(obj_file)
    for i in range(len(cat_files)):
        pic_name = dog_files[i].split('\\')[-1]
        if i > 0.7 * len(cat_files):
            obj_file = os.path.join(val_root, pic_name)
            shutil.move(os.path.join(train_root, cat_files[i]), obj_file)
            print(obj_file)