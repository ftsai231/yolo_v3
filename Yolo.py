import numpy as np
import tensorflow as tf
from config import cfg
import utils as utils

from Yolo_model import darknet53, yolo_conv_block, yolo_detection_layer, con2d_fixed_padding, batch_norm, \
    upsample, build_boxes, non_max_suppression, bbox_giou, focal, bbox_iou


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
        # if not data_format:
        #     if tf.test.is_built_with_cuda():
        #         data_format = 'channels_first'
        #     else:
        #         data_format = 'channels_last'
        self.iou_loss_thresh = 0.5
        self.num_class = 2
        self.anchor_per_scale = 3
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.n_classes = len(self.classes)
        self.model_size = cfg.YOLO.MODEL_SIZE
        self.max_output_size = cfg.YOLO.max_output_size
        self.iou_threshold = cfg.YOLO.iou_threshold
        self.confidence_threshold = cfg.YOLO.confidence_threshold
        self.data_format = data_format
        self.strides = np.array(cfg.YOLO.STRIDES)

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        with tf.variable_scope('yolo_v3_model'):
            self.anchors = utils.get_anchors()

            # normalize values to range [0..1]
            # inputs = inputs / 255

            route1, route2, inputs = darknet53(inputs=inputs, training=training, data_format=self.data_format)
            route, inputs = yolo_conv_block(inputs=inputs, filters=512, training=training, data_format=self.data_format,
                                            name='yolo_conv_block_1')
            self.detection1, self.conv_lbbox = yolo_detection_layer(inputs, n_classes=self.n_classes,
                                                                    anchors=self.anchors[2],
                                                                    img_size=self.model_size,
                                                                    data_format=self.data_format,
                                                                    name='con_lbbox_layer')
            inputs = con2d_fixed_padding(route, filters=256, kernel_size=1, data_format=self.data_format, name='conv_lbbox')
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

            upsample_output_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)

            # get the channel axis
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat((inputs, route2), axis=axis)

            route, inputs = yolo_conv_block(inputs, filters=256, training=training, data_format=self.data_format,
                                            name='yolo_conv_block_2')
            self.detection2, self.conv_mbbox = yolo_detection_layer(inputs, n_classes=self.n_classes,
                                                                    anchors=self.anchors[1],
                                                                    img_size=self.model_size,
                                                                    data_format=self.data_format,
                                                                    name='con_mbbox_layer')
            inputs = con2d_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format, name='conv_mbbox')
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)
            upsample_output_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)
            inputs = tf.concat((inputs, route1), axis=axis)

            route, inputs = yolo_conv_block(inputs, filters=128, training=training, data_format=self.data_format,
                                            name='yolo_conv_block_3')
            self.detection3, self.conv_sbbox = yolo_detection_layer(inputs, n_classes=self.n_classes,
                                                                    anchors=self.anchors[0],
                                                                    img_size=self.model_size,
                                                                    data_format=self.data_format,
                                                                    name='conv_sbbox')

            print("self.detection1 shape: ", self.detection1.shape)
            print("self.detection2 shape: ", self.detection2.shape)
            print("self.detection3 shape: ", self.detection3.shape)

            detection1 = tf.reshape(self.detection1, [-1, 13 * 13 * 3, 5 + self.n_classes])
            detection2 = tf.reshape(self.detection2, [-1, 26 * 26 * 3, 5 + self.n_classes])
            detection3 = tf.reshape(self.detection3, [-1, 52 * 52 * 3, 5 + self.n_classes])

            self.pred_bbox = tf.concat((detection1, detection2, detection3), axis=1)

        """ make a predict function later"""
        # self.boxes_dicts = non_max_suppression(detection, n_classes=self.n_classes,
        #                                       max_output_size=self.max_output_size,
        #                                       confidence_threshold=self.confidence_threshold,
        #                                       iou_threshold=self.iou_threshold)

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.detection3, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.detection2, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.detection1, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        # print("giou_loss: ", giou_loss, "conf_loss: ", conf_loss, "prob_loss: ", prob_loss)
        return giou_loss, conf_loss, prob_loss

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        # Box loss with x, y, w, h
        box_pred_xywh = tf.concat([pred_xywh[:, :, :, :, 0:2], tf.sqrt(pred_xywh[:, :, :, :, 2:4])], axis=-1)
        box_label_xywh = tf.concat((label_xywh[:, :, :, :, 0:2], tf.sqrt(label_xywh[:, :, :, :, 2:4])), axis=-1)
        bbox_loss = respond_bbox * focal(box_pred_xywh, box_label_xywh)
        bbox_loss = tf.reduce_mean(tf.reduce_sum(bbox_loss, axis=[1, 2, 3, 4])) * 0.5
        # bbox_loss = tf.reduce_sum(bbox_loss) * 0.5
        # print("bbox_loss: ", bbox_loss)
        # print("bbox_loss.shape: ", tf.shape(bbox_loss))

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        # prob_loss = 0
        # conf_loss = 0

        return bbox_loss, conf_loss, prob_loss

    def precision(self, boxes1, boxes2):
        # boxes1 = label
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        precision = 1.0 * inter_area / boxes1_area

        return precision

# if __name__ == "__main__":
#     inputs_test = tf.placeholder(tf.float32, [3, 416, 416, 3])
#     model = YoloV3(inputs_test, False)
