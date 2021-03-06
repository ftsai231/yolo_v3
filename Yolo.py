import numpy as np
import tensorflow as tf
from config import cfg
import utils as utils

from Yolo_model import darknet53, yolo_conv_block, yolo_detection_layer, con2d_fixed_padding, batch_norm, \
    upsample, bbox_giou, focal


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
        self.iou_loss_thresh = 0.5
        self.num_class = 2
        self.anchor_per_scale = 3
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.n_classes = len(self.classes)
        self.model_size = cfg.YOLO.MODEL_SIZE
        self.max_output_size = cfg.YOLO.MAX_OUTPUT_SIZE
        self.iou_threshold = cfg.YOLO.IOU_THRESHOLD
        self.data_format = data_format
        self.strides = np.array(cfg.YOLO.STRIDES)
        self._giou_loss_backup = None

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        with tf.variable_scope('yolo_v3_model'):
            self.anchors = utils.get_anchors()

            route1, route2, inputs = darknet53(inputs=inputs, training=training, data_format=self.data_format)
            route, inputs = yolo_conv_block(inputs=inputs, filters=512, training=training, data_format=self.data_format,
                                            name='yolo_conv_block_1')
            self.conv_lbbox = yolo_detection_layer(inputs, n_classes=self.n_classes,
                                                                    anchors=self.anchors[2],
                                                                    img_size=self.model_size,
                                                                    data_format=self.data_format,
                                                                    name='con_lbbox_layer',
                                                                    strides=self.strides[2])
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
            self.conv_mbbox = yolo_detection_layer(inputs, n_classes=self.n_classes,
                                                                    anchors=self.anchors[1],
                                                                    img_size=self.model_size,
                                                                    data_format=self.data_format,
                                                                    name='con_mbbox_layer',
                                                                    strides=self.strides[1])
            inputs = con2d_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format, name='conv_mbbox')
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)
            upsample_output_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)
            inputs = tf.concat((inputs, route1), axis=axis)

            route, inputs = yolo_conv_block(inputs, filters=128, training=training, data_format=self.data_format,
                                            name='yolo_conv_block_3')
            self.conv_sbbox = yolo_detection_layer(inputs, n_classes=self.n_classes,
                                                                    anchors=self.anchors[0],
                                                                    img_size=self.model_size,
                                                                    data_format=self.data_format,
                                                                    name='conv_sbbox',
                                                                    strides=self.strides[0])

            with tf.variable_scope('pred_sbbox'):
                self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
                print("self.pred_sbbox: ", self.pred_sbbox.shape)

            with tf.variable_scope('pred_mbbox'):
                self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

            with tf.variable_scope('pred_lbbox'):
                self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        print("compute loss")
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        debug = loss_sbbox[3] + loss_mbbox[3] + loss_lbbox[3]

        return giou_loss, conf_loss, prob_loss, debug

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        # np.set_printoptions(threshold=np.inf)
        debug = []

        conv = tf.cast(conv, tf.float64)
        pred = tf.cast(pred, tf.float64)
        label = tf.cast(label, tf.float64)
        bboxes = tf.cast(bboxes, tf.float64)

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
        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        giou = tf.where(tf.is_nan(giou), tf.zeros_like(giou), giou)
        giou = tf.cast(giou, tf.float64)
        line = tf.convert_to_tensor("------------------------------------------------------------------------------")
        debug.append(giou)
        debug.append(line)

        input_size = tf.cast(input_size, tf.float64)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        giou_loss = tf.where(tf.is_nan(giou_loss), tf.zeros_like(giou_loss), giou_loss)
        giou_loss_64 = tf.cast(giou_loss, tf.float64)
        debug.append(giou_loss_64)
        debug.append(line)

        # Box loss with x, y, w, h (need to scale either pred or label)
        # box_pred_xywh = tf.concat([pred_xywh[:, :, :, :, 0:2], tf.sqrt(pred_xywh[:, :, :, :, 2:4])], axis=-1)
        # box_label_xywh = tf.concat((label_xywh[:, :, :, :, 0:2], tf.sqrt(label_xywh[:, :, :, :, 2:4])), axis=-1)
        # bbox_loss = respond_bbox * focal(box_pred_xywh, box_label_xywh)
        # bbox_loss = tf.reduce_mean(tf.reduce_sum(bbox_loss, axis=[1, 2, 3, 4]))

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float64)

        conf_focal = focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )
        conf_loss = tf.where(tf.is_nan(conf_loss), tf.zeros_like(conf_loss), conf_loss)
        conf_loss_64 = tf.cast(conf_loss, dtype=tf.float64)

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        prob_loss = tf.where(tf.is_nan(prob_loss), tf.zeros_like(prob_loss), prob_loss)
        prob_loss_64 = tf.cast(prob_loss, dtype=tf.float64)
        # cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))

        conf_loss_mean = tf.reduce_mean(tf.reduce_sum(conf_loss_64, axis=[1, 2, 3, 4]))
        prob_loss_mean = tf.reduce_mean(tf.reduce_sum(prob_loss_64, axis=[1, 2, 3, 4]))
        giou_loss_mean = tf.reduce_mean(tf.reduce_sum(giou_loss_64, axis=[1, 2, 3, 4]))

        debug.append(giou_loss_mean)

        return giou_loss_mean, conf_loss_mean, prob_loss_mean, debug

    def bbox_iou(self, boxes1, boxes2):
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
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / tf.maximum(union_area, 1e-7)
        # iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return iou

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)
        print("conv_output before reshape: ", conv_output.shape)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

if __name__ == "__main__":
    inputs_test = tf.placeholder(tf.float32, [3, 416, 416, 3])
    model = YoloV3(inputs_test, False)