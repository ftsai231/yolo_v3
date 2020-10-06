import numpy as np
import tensorflow as tf
from config import cfg


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=cfg.YOLO.BATCH_NORM_DECAY, epsilon=cfg.YOLO.BATCH_NORM_EPSILON,
        scale=True, training=training)


def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding.

    Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
        data_format: The input format.
    Returns:
        A tensor with the same format as the input.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


# use stride to control the output size
def con2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)


def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    residual = inputs
    inputs = con2d_fixed_padding(inputs, filters, kernel_size=1, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, 2 * filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = inputs + residual

    return inputs


def darknet53(inputs, training, data_format):
    inputs = con2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=32, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=128, kernel_size=3, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=256, kernel_size=3, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=512, kernel_size=3, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    route1 = inputs

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=1024, kernel_size=3, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    route2 = inputs

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512, training=training, data_format=data_format)

    return route1, route2, inputs


def yolo_conv_block(inputs, filters, training, data_format):
    inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    route = inputs

    inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    return route, inputs


def yolo_detection_layer(inputs, n_classes, anchors, img_size, data_format):
    n_anchors = len(anchors)

    # detection kernel
    inputs = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes), kernel_size=1, strides=1, use_bias=True,
                                    data_format=data_format)

    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    inputs = tf.reshape(inputs, [-1, grid_shape[0], grid_shape[1], n_anchors, 5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    box_center, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    conv_bbox = tf.concat([box_center, box_shapes, confidence, classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, grid_shape[0], grid_shape[0], n_anchors, 2])
    box_centers = tf.nn.sigmoid(box_center)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    anchors = tf.reshape(anchors, [grid_shape[0], grid_shape[1], n_anchors, 2])
    box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    return inputs, conv_bbox


def upsample(inputs, out_shape, data_format):
    """Upsamples to `out_shape` using nearest neighbor interpolation."""
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def build_boxes(inputs):
    """Computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


# Performs non-max suppression separately for each class.
# def non_max_suppression(detection_list, n_classes, max_output_size, confidence_threshold, iou_thresould):
#     # batch = tf.unstack(inputs)
#     boxes_dicts = []
#     detection_list_shape = len(detection_list)
#     for i in range(detection_list_shape):
#         batch = tf.unstack(detection_list[i])
#         print("batch.shape: \n", batch)
#         # for boxes in batch:
#         # the 4th place is the confidence score
#         # print("boxes: ", boxes)
#         boxes = np.array(batch)[0]
#         print(boxes[0])
#
#         boxes = tf.boolean_mask(boxes, boxes[:, :, :, 4] > confidence_threshold)
#         # get the classes
#         classes = tf.argmax(boxes[:, 5:], axis=-1)
#         # add one more dim in classes
#         classes = tf.expand_dims(tf.to_float(classes), axis=-1)
#         # add the result in the last dim in the box
#         boxes = tf.concat([boxes[:, :5], classes], axis=-1)
#
#         boxes_dict = dict()
#         for cls in range(n_classes):
#             mask = tf.equal(boxes[:, 5], cls)
#             mask_shape = mask.get_shape()
#             if mask_shape.ndims != 0:
#                 class_boxes = tf.boolean_mask(boxes, mask)
#                 boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
#                                                               [4, 1, -1],
#                                                               axis=-1)
#                 boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
#                 indices = tf.image.non_max_suppression(boxes_coords,
#                                                        boxes_conf_scores,
#                                                        max_output_size,
#                                                        cfg.YOLO.iou_threshold)
#                 class_boxes = tf.gather(class_boxes, indices)
#                 boxes_dict[cls] = class_boxes[:, :5]
#
#         boxes_dicts.append(boxes_dict)
#
#     return boxes_dicts

def non_max_suppression(detection_list, n_classes, max_output_size, confidence_threshold, iou_threshold):
    """Performs non-max suppression separately for each class.

    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
    Returns:
        A list containing class-to-boxes dictionaries
            for each sample in the batch.
    """
    boxes_dicts = []
    detection_list_shape = len(detection_list)
    for i in range(detection_list_shape):
        batch = tf.unstack(detection_list[i])

        for boxes in batch:
            boxes = tf.boolean_mask(boxes, boxes[:, :, :, 4] > confidence_threshold)
            classes = tf.argmax(boxes[:, 5:], axis=-1)
            classes = tf.expand_dims(tf.to_float(classes), axis=-1)
            boxes = tf.concat([boxes[:, :5], classes], axis=-1)

            boxes_dict = dict()
            for cls in range(n_classes):
                mask = tf.equal(boxes[:, 5], cls)
                mask_shape = mask.get_shape()
                if mask_shape.ndims != 0:
                    class_boxes = tf.boolean_mask(boxes, mask)
                    boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                                  [4, 1, -1],
                                                                  axis=-1)
                    boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                    indices = tf.image.non_max_suppression(boxes_coords,
                                                           boxes_conf_scores,
                                                           max_output_size,
                                                           iou_threshold)
                    class_boxes = tf.gather(class_boxes, indices)
                    boxes_dict[cls] = class_boxes[:, :5]

            boxes_dicts.append(boxes_dict)

    return boxes_dicts


def loss_layer(conv, pred, label, bboxes, stride):
    n_classes = cfg.TRAIN.N_CLASSES
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    conv = tf.reshape(conv, [batch_size, output_size, output_size, cfg.YOLO.ANCHOR_PER_SCALE, 5 + n_classes])

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    # respond_bbox相當於mask，對應正樣本的anchor為1,負樣本為0
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    input_size = tf.cast(input_size, tf.float32)

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    # 如果一個背景anchor的iou如果超過一定閾值，那我們不計算它的損失
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < cfg.YOLO.IOU_LOSS_THRESH, tf.float32)
    conf_focal = focal(target=respond_bbox, actual=pred_conf)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    ## 多分类损失
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss


def bbox_iou(boxes1, boxes2):
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
    iou = 1.0 * inter_area / union_area

    return iou


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def focal(target, actual, alpha=1, gamma=2):
    focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss

