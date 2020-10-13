import cv2
import tensorflow as tf
from Yolo import YoloV3
from config import cfg
import numpy as np
import utils as utils


class YoloPredict(object):
    def __init__(self):
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.weights_path = './checkpoint/train_checkpoint/yolov3_model_test_overfit.ckpt'
        self.input_size = cfg.TRAIN.INPUT_SIZE
        self.model_size = cfg.YOLO.MODEL_SIZE

        with tf.name_scope('input'):
            self.input = tf.placeholder(dtype=tf.float32, shape=[1,  self.input_size, self.input_size, 3], name='input')
            self.training = tf.placeholder(dtype=bool, name='trainable')

        model = YoloV3(inputs=self.input, training=self.training)
        self.pred_bbox = model.pred_bbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weights_path)

    def predict(self):
        np.set_printoptions(threshold=np.inf)
        image_path = './23429.jpg'
        image = np.array(cv2.imread(image_path))
        image_shape = image.shape
        print("image_shape: ", image_shape)
        image = np.copy(image)
        # image = image / 255
        image_data = utils.image_preprocess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        # print("image_data shape: ", image_data.shape)
        pred_bbox = self.sess.run([self.pred_bbox], feed_dict={
            self.input: image_data,
            self.training: False
        })
        pred_bbox = np.array(pred_bbox[0][0])
        # print("pred_bbox before: ", pred_bbox)
        # print("pred_bbox shpae before: ", pred_bbox.shape)
        pred_bbox = utils.postprocess_boxes(pred_bbox, image_shape, 416, 0.7)
        print("pred_bbox shpae after: ", pred_bbox.shape)

        pred_bbox = utils.nms(pred_bbox, 0.45)
        print("pred_bbox after: ", pred_bbox)


        image = utils.draw_bbox(image, pred_bbox, show_label=True)
        cv2.imwrite('./test.jpg', image)


if __name__ == "__main__":
    YoloPredict().predict()
