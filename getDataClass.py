import os
import pickle
from shutil import copyfile

import scipy.misc
from pycocotools.coco import COCO
from config import cfg
from PIL import Image
import urllib.request
import requests


coco = COCO(cfg.TRAIN.ANNOT_PATH)
catIds_person = coco.getCatIds(catNms=['person'])
imgIds_person = coco.getImgIds(catIds=catIds_person)
images_person = coco.loadImgs(imgIds_person)
print("type(images_person): ", type(images_person))
print(len(images_person))

catIds_car = coco.getCatIds(catNms=['car'])
imgIds_car = coco.getImgIds(catIds=catIds_car)
images_car = coco.loadImgs(imgIds_car)
print("type(images_car): ", type(images_car))
print(len(images_car))

car_and_person = imgIds_person[:10000] + imgIds_car[:10000]
# print(car_and_person)
print("car_and_person:", len(car_and_person))

# TODO: writing the list to txt file. DO NOT DELETE THE COMMENTED CODE!
# with open('car_and_person.txt', 'w') as output:
#     output.write(str(car_and_person).strip("[]"))



# TODO: saving the photos. DO NOT DELETE THE COMMENTED CODE!
# arr = os.listdir('/Users/JTSAI1/Documents/ADAI/train2014')
# dst = '/Users/JTSAI1/Documents/ADAI/train_car_person/'
#
# imgs = coco.loadImgs(ids=car_and_person)
#
# for id in car_and_person:
#     img = coco.loadImgs(ids=[id])
#     # print(img)
#     # break
#     image_url = img[0]['coco_url']
#     img_data = requests.get(image_url).content
#     with open(dst + str(id) + '.jpg', 'wb') as handler:
#         handler.write(img_data)
#         print("copyied")