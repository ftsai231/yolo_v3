import numpy as np
import tensorflow as tf
import cv2

import utils as utils

# image_path = '/Users/JTSAI1/Documents/ADAI/train_car_person/' + str(141003) + '.jpg'
# image = np.array(cv2.imread(image_path))
# image = np.copy(image)
# image_data = utils.image_preporcess(image, [416, 416])
#
# print(image_data.shape)
arr = [1,2,1,2,0,2,3,4,3,2,1,3]
arr = np.array(arr)
print(arr)

boxes = tf.boolean_mask(arr, arr > 1)
print(boxes)