import numpy as np
import tensorflow as tf
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont
from config import cfg

import utils as utils



list1 = [[298,  35, 400, 160,   0],
 [174,  97, 302, 205,   0],
 [193,  51, 280, 157,   0],
 [  3,  57, 114, 374,   0],
 [271, 107, 401, 212,   0],
 [114,  57, 189, 185,   0],
 [ 56, 120, 162, 381,   0]]
#
name = ['person', 'car']
#
# def sway_index(list):
#     ind1 = list[1]
#     list[1] = list[0]
#     list[0] = ind1
#
#     ind3 = list[3]
#     list[3] = list[2]
#     list[2] = ind3
#
# sway_index(list1)
#
def draw_boxes(list, class_names, model_size):
    img_name = '414162.jpg'
    img = Image.open(img_name)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='./futura/futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)
    colors = ((np.array(color_palette("hls", 2)) * 255)).astype(np.uint8)
    resize_factor = \
        (img.size[0] / model_size[0], img.size[1] / model_size[1])
    for box in list:
        color = colors[0]
        xy, confidence = box[:4], 1
        xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
        x0, y0 = xy[0], xy[1]
        thickness = (img.size[0] + img.size[1]) // 200
        # for the bounding box
        for t in np.linspace(0, 1, thickness):
            xy[0], xy[1] = xy[0] + t, xy[1] + t
            xy[2], xy[3] = xy[2] - t, xy[3] - t
            draw.rectangle(xy, outline=tuple(color))
        # for the text with a box
        text = '{} {:.1f}%'.format(class_names[0],
                                   confidence * 100)
        text_size = draw.textsize(text, font=font)
        draw.rectangle(
            [x0, y0 - text_size[1], x0 + text_size[0], y0],
            fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black',
                  font=font)
#
#     # display(img)
    img.save('test.jpg')
    print('image saved!')
#
#
draw_boxes(list1, name, cfg.YOLO.MODEL_SIZE)
list1 = np.array(list1)
print(np.sqrt(list1[:, 1:3]))
