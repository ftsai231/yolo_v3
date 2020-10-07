import numpy as np
import tensorflow as tf
from config import cfg
import utils as utils

from Yolo_model import darknet53, yolo_conv_block, yolo_detection_layer, con2d_fixed_padding, batch_norm, \
    upsample, build_boxes, non_max_suppression, loss_layer


class YoloV3:

    # for building the final yolo model
    def __init__(self, inputs, training, data_format=None):
        """Creates the model.

        Args:
            n_classes: Number of class labels.
            model_size: The input size of the model.
            max_output_size: Max number of boxes to be selected for each class.
            iou_threshold: Threshold for the IOU.
            confidence_threshold: Threshold for the confidence score.
            data_format: The input format.

        Returns:
            None.
        """
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.n_classes = len(self.classes)
        self.model_size = cfg.YOLO.MODEL_SIZE
        self.max_output_size = cfg.YOLO.max_output_size
        self.iou_threshold = cfg.YOLO.iou_threshold
        self.confidence_threshold = cfg.YOLO.confidence_threshold
        self.data_format = data_format

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        with tf.name_scope('detection_layers'):
            self.detection1, self.detection2, self.detection3, self.conv_lbbox, self.conv_mbbox, self.conv_sbbox \
                = self.build_model(inputs, training)

        detection_list = [self.detection1, self.detection2, self.detection3]

        """ make a predict function later"""
        self.boxes_dicts = non_max_suppression(detection_list, n_classes=self.n_classes,
                                              max_output_size=self.max_output_size,
                                              confidence_threshold=self.confidence_threshold,
                                              iou_threshold=self.iou_threshold)

    def build_model(self, inputs, training):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # normalize values to range [0..1]
        inputs = inputs / 255

        route1, route2, inputs = darknet53(inputs=inputs, training=training, data_format=self.data_format)
        route, inputs = yolo_conv_block(inputs=inputs, filters=512, training=training, data_format=self.data_format)
        detection1, conv_lbbox = yolo_detection_layer(inputs, n_classes=self.n_classes, anchors=cfg.YOLO.ANCHORS[6:9],
                                                      img_size=self.model_size,
                                                      data_format=self.data_format)

        inputs = con2d_fixed_padding(route, filters=256, kernel_size=1, data_format=self.data_format)
        inputs = batch_norm(inputs, training=training,
                            data_format=self.data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)
        upsample_output_size = route2.get_shape().as_list()
        inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)

        # get the channel axis
        axis = 1 if self.data_format == 'channels_first' else 3
        inputs = tf.concat((inputs, route2), axis=axis)

        route, inputs = yolo_conv_block(inputs, filters=256, training=training, data_format=self.data_format)
        detection2, conv_mbbox = yolo_detection_layer(inputs, n_classes=self.n_classes, anchors=cfg.YOLO.ANCHORS[3:6],
                                                      img_size=self.model_size,
                                                      data_format=self.data_format)
        inputs = con2d_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format)
        inputs = batch_norm(inputs, training=training,
                            data_format=self.data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)
        upsample_output_size = route1.get_shape().as_list()
        inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)
        inputs = tf.concat((inputs, route1), axis=axis)

        route, inputs = yolo_conv_block(inputs, filters=128, training=training, data_format=self.data_format)
        detection3, conv_sbbox = yolo_detection_layer(inputs, n_classes=self.n_classes, anchors=cfg.YOLO.ANCHORS[0:3],
                                                      img_size=self.model_size,
                                                      data_format=self.data_format)

        detection1 = build_boxes(detection1)
        detection2 = build_boxes(detection2)
        detection3 = build_boxes(detection3)

        return detection1, detection2, detection3, conv_lbbox, conv_mbbox, conv_sbbox

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('small_box_loss'):
            sbbox_loss = loss_layer(conv=self.conv_sbbox, pred=self.detection3, label=label_sbbox, bboxes=true_sbbox,
                                    stride=cfg.YOLO.STRIDES[0])

        with tf.name_scope('medium_box_loss'):
            mbbox_loss = loss_layer(conv=self.conv_mbbox, pred=self.detection2, label=label_mbbox, bboxes=true_mbbox,
                                    stride=cfg.YOLO.STRIDES[1])

        with tf.name_scope('large_box_loss'):
            lbbox_loss = loss_layer(conv=self.conv_lbbox, pred=self.detection1, label=label_lbbox, bboxes=true_lbbox,
                                    stride=cfg.YOLO.STRIDES[2])

        with tf.name_scope('giou_loss'):
            giou_loss = sbbox_loss[0] + mbbox_loss[0] + lbbox_loss[0]

        with tf.name_scope('giou_loss'):
            conf_loss = sbbox_loss[1] + mbbox_loss[1] + lbbox_loss[1]

        with tf.name_scope('giou_loss'):
            prob_loss = sbbox_loss[2] + mbbox_loss[2] + lbbox_loss[2]

        return giou_loss, conf_loss, prob_loss


# if __name__ == "__main__":
#     inputs_test = tf.placeholder(tf.float32, [3, 416, 416, 3])
#     # inputs_test = np.zeros((3, 416, 416, 3))
#     model = YoloV3(inputs_test, False)







