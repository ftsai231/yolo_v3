import numpy as np
import tensorflow as tf
import cv2

import utils as utils

# image_path = '/Users/JTSAI1/Documents/ADAI/train_car_person/' + str(141003) + '.jpg'
# image = np.array(cv2.imread(image_path))
# image = np.copy(image)
#
# print(image_data.shape)
arr = [[1, 2, 1, 2, 0, 2, 3, 4, 3, 2, 1, 3], [2, 3, 5, 6, 3, 2, 4, 2, 5, 6, 7, 3]]
arr = np.array(arr)
print(arr / 2)

boxes = tf.boolean_mask(arr, arr > 1)
print(boxes)
