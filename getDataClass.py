import random

import numpy as np
from pycocotools.coco import COCO
from config import cfg


coco = COCO(cfg.TRAIN.ANNOT_PATH)
catIds_person = coco.getCatIds(catNms=['person'])
imgIds_person = coco.getImgIds(catIds=catIds_person)
images_person = coco.loadImgs(imgIds_person)
print("type(images_person): ", type(images_person))
print(len(images_person))    # 45174

catIds_car = coco.getCatIds(catNms=['car'])
imgIds_car = coco.getImgIds(catIds=catIds_car)
images_car = coco.loadImgs(imgIds_car)
print("type(images_car): ", type(images_car))
print(len(images_car))     # 8606

car_and_person = imgIds_person[:8606] + imgIds_car[:8606]
# print(car_and_person)
print("car_and_person:", len(car_and_person))
# random.shuffle(car_and_person)

train_set = imgIds_person[:3500] + imgIds_car[:3500]
test_set = imgIds_person[3500:4000] + imgIds_car[3500:4000]
print(len(train_set))
print(len(test_set))


random.shuffle(train_set)
random.shuffle(test_set)

# TODO: writing the list to txt file. DO NOT DELETE THE COMMENTED CODE!
with open('car_and_person_train.txt', 'w') as output:
    output.write(str(train_set).strip("[]"))

with open('car_and_person_test.txt', 'w') as output:
    output.write(str(test_set).strip("[]"))

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

# TODO: do the shuffle for car and person list
# def load_car_person_list():
#     open_file = open('car_and_person.txt').read()
#     id_str = ""
#     for line in open_file:
#         id_str += line
#
#     id_str = id_str.strip("[]").split(", ")
#
#     for i in range(0, len(id_str)):
#         id_str[i] = int(id_str[i])
#
#     np.random.shuffle(id_str)
#     # print(id_str)
#     with open('car_and_person.txt', 'w') as output:
#         output.write(str(id_str).strip("[]"))
#     return id_str


