from PIL import Image
import os
import shutil
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_TRAIN_PATH ='C:/cvpr2019_kaggle_competition/train'
Little_image_PATH = 'C:\\cvpr2019_kaggle_competition\\little_images'

img_dirs = os.listdir(IMG_TRAIN_PATH)
img_dirs = [os.path.join(IMG_TRAIN_PATH, img) for img in img_dirs]

for path in img_dirs:
    try:
        img = Image.open(path)
        sz1, sz2 = img.size
        if(sz1<56 or sz2<56):
            shutil.move(path, Little_image_PATH)
            print(path)
    except IOError:
        print(path, 'not found')

