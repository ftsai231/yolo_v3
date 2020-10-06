import cv2
import tensorflow as tf

from Yolo import YoloV3
import utils as utils
from config import cfg
import numpy as np


class YoloPredict(object):
    def __init__(self):
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.weights_path = './checkpoint/train_checkpoint/yolov3_model-1.ckpt-1'
        self.input_size = cfg.TRAIN.INPUT_SIZE
        self.model_size = cfg.YOLO.MODEL_SIZE

        with tf.name_scope('input'):
            self.input = tf.placeholder(dtype=tf.float32, shape=[1,  self.input_size, self.input_size, 3], name='input')
            self.training = tf.placeholder(dtype=bool, name='training')

        model = YoloV3(inputs=self.input, training=self.training)
        self.boxes_dicts = model.boxes_dicts

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weights_path)

    def predict(self):
        image_path = './46211.jpg'
        image = np.array(cv2.imread(image_path))
        image = np.copy(image)
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        box_dicts = self.sess.run([self.boxes_dicts], feed_dict={
            self.input: image_data,
            self.training: False
        })

        # TODO: list of image names
        img_names = []
        img_names.append(image_path)

        utils.draw_boxes(img_names, box_dicts, self.classes, self.model_size)


if __name__ == "__main__":
    YoloPredict().predict()
