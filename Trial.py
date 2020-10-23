import numpy as np
import tensorflow as tf
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont
from config import cfg

import utils as utils



list1 = [[190, 110, 375, 385,   1],
 [136, 157, 242, 232,   1],
 [ 79, 155, 164, 201,   1],
 [  1, 167,  19, 194,   1],
 [ 88, 158, 203, 351,   0],
 [ 99, 166, 152, 236,   1],
 [  1, 156,   7, 165,   0],
 [ 74, 159,  80, 171,   0],
 [ 20, 160,  47, 179,   1]]
#
name = ['person', 'car']
#
# def draw_boxes(list, class_names, model_size):
#     img_name = '23429.jpg'
#     img = Image.open(img_name)
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype(font='./futura/futur.ttf',
#                               size=(img.size[0] + img.size[1]) // 100)
#     colors = ((np.array(color_palette("hls", 2)) * 255)).astype(np.uint8)
#     resize_factor = \
#         (img.size[0] / model_size[0], img.size[1] / model_size[1])
#     for box in list:
#         color = colors[0]
#         xy, confidence = box[:4], 1
#         xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
#         x0, y0 = xy[0], xy[1]
#         thickness = (img.size[0] + img.size[1]) // 200
#         # for the bounding box
#         for t in np.linspace(0, 1, thickness):
#             xy[0], xy[1] = xy[0] + t, xy[1] + t
#             xy[2], xy[3] = xy[2] - t, xy[3] - t
#             draw.rectangle(xy, outline=tuple(color))
#         # for the text with a box
#         text = '{} {:.1f}%'.format(class_names[0],
#                                    confidence * 100)
#         text_size = draw.textsize(text, font=font)
#         draw.rectangle(
#             [x0, y0 - text_size[1], x0 + text_size[0], y0],
#             fill=tuple(color))
#         draw.text((x0, y0 - text_size[1]), text, fill='black',
#                   font=font)
# #
# #     # display(img)
#     img.save('test.jpg')
#     print('image saved!')
# #
# #
# draw_boxes(list1, name, cfg.YOLO.MODEL_SIZE)

''' -----------------------------------------------------------------------------------------------------------------'''

def test():
 return 1, 2, 4, 4, 5, 3, 2, 1

x = [1, 2, 4, 4, 5, 3, 2, 1]
x = x > 2
print(x)
