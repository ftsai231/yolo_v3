import tensorflow as tf
from config import cfg
import numpy as np

def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(inputs, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=training)

    # return tf.layers.batch_normalization(
    #         inputs=inputs,
    #         momentum=cfg._BATCH_NORM_DECAY, epsilon=cfg._BATCH_NORM_EPSILON,
    #         scale=True, training=training)

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
def con2d_fixed_padding(inputs, filters, name, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    with tf.variable_scope(name):
        inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False, data_format=data_format)
    return inputs


def darknet53_residual_block(inputs, filters, training, name, data_format, strides=1):
    with tf.variable_scope(name):
        residual = inputs
        inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format,
                                     name='residual_conv_1')
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        inputs = con2d_fixed_padding(inputs, 2 * filters, kernel_size=3, strides=strides, data_format=data_format,
                                     name='residual_conv_2')
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        inputs = inputs + residual

    return inputs


def darknet53(inputs, training, data_format):
    inputs = con2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format,
                                 name='darknet_con2d_fixed_padding_1')
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=2, data_format=data_format,
                                 name='darknet_con2d_fixed_padding_2')
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=32, training=training, data_format=data_format,
                                      name='darknet53_residual_block_1')

    inputs = con2d_fixed_padding(inputs, filters=128, kernel_size=3, strides=2, data_format=data_format,
                                 name='darknet_con2d_fixed_padding_3')
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    for i in range(2):
        name = 'darknet53_residual_block_1_' + str(i)
        inputs = darknet53_residual_block(inputs, filters=64, training=training, data_format=data_format, name=name)

    inputs = con2d_fixed_padding(inputs, filters=256, kernel_size=3, data_format=data_format, strides=2,
                                 name='darknet_con2d_fixed_padding_4')
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    for i in range(8):
        name = 'darknet53_residual_block_2_' + str(i)
        inputs = darknet53_residual_block(inputs, filters=128, training=training, data_format=data_format, name=name)

    route1 = inputs

    inputs = con2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=2, data_format=data_format,
                                 name='darknet_con2d_fixed_padding_5')
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    for i in range(8):
        name = 'darknet53_residual_block_3_' + str(i)
        inputs = darknet53_residual_block(inputs, filters=256, training=training, data_format=data_format, name=name)

    route2 = inputs

    inputs = con2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=2, data_format=data_format,
                                 name='darknet_con2d_fixed_padding_6')
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    for i in range(4):
        name = 'darknet53_residual_block_4_' + str(i)
        inputs = darknet53_residual_block(inputs, filters=512, training=training, data_format=data_format, name=name)

    return route1, route2, inputs


def yolo_conv_block(inputs, filters, training, data_format, name):
    with tf.variable_scope(name):
        inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format,
                                     name='block_conv_1')
        inputs = batch_norm(inputs, training=training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format,
                                     name='block_conv_2')
        inputs = batch_norm(inputs, training=training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format,
                                     name='block_conv_3')
        inputs = batch_norm(inputs, training=training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format,
                                     name='block_conv_4')
        inputs = batch_norm(inputs, training=training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format,
                                     name='block_conv_5')
        inputs = batch_norm(inputs, training=training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

        route = inputs

        inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format,
                                     name='block_conv_6')
        inputs = batch_norm(inputs, training=training, data_format=data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha=cfg.YOLO.LEAKY_RELU)

    return route, inputs


def yolo_detection_layer(inputs, n_classes, anchors, img_size, data_format, name, strides):
    n_anchors = len(anchors)

    # detection kernel
    inputs = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes), kernel_size=1, strides=1, use_bias=True,
                              data_format=data_format)

    #############
    batch_size = inputs.shape[0]
    output_size = inputs.shape[1]
    n_classes = cfg.TRAIN.N_CLASSES
    anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
    #############

    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    # print("grid shape: ", grid_shape)
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    print("inputs before reshape: ", inputs.shape)

    conv_bbox = inputs
    #
    # ########################
    # inputs = tf.reshape(inputs, [-1, grid_shape[0], grid_shape[0], anchor_per_scale, 5 + n_classes])
    # strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    #
    # box_center, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)
    #
    # x = tf.range(grid_shape[0], dtype=tf.float32)
    # y = tf.range(grid_shape[1], dtype=tf.float32)
    # x_offset, y_offset = tf.meshgrid(x, y)
    # x_offset = tf.reshape(x_offset, (-1, 1))
    # y_offset = tf.reshape(y_offset, (-1, 1))
    # x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    # x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    # x_y_offset = tf.reshape(x_y_offset, [1, grid_shape[0], grid_shape[0], n_anchors, 2])
    # box_centers = tf.nn.sigmoid(box_center)
    # box_centers = (box_centers + x_y_offset) * strides
    #
    # anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    # anchors = tf.reshape(anchors, [grid_shape[0], grid_shape[1], n_anchors, 2])
    # box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)
    #
    # confidence = tf.nn.sigmoid(confidence)
    #
    # classes = tf.nn.sigmoid(classes)
    #
    # with tf.variable_scope(name):
    #     inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    #####

    # 原始网络输出的(x, y, w, h, score, probability)
    # conv_raw_dxdy = conv_bbox[:, :, :, :, 0:2]  # 中心位置的偏移量 x,y
    # conv_raw_dwdh = conv_bbox[:, :, :, :, 2:4]  # 预测框长宽的偏移量 w,h
    # conv_raw_conf = conv_bbox[:, :, :, :, 4:5]  # 预测框的置信度 score
    # conv_raw_prob = conv_bbox[:, :, :, :, 5:]  # 预测框类别概率 prob
    #
    # # tf.tile:按列数(output_size)扩展input,见下面的例子
    # # [[0]    [[0 0 0]
    # #  [1]     [1 1 1]
    # #  [2]]    [2 2 2]]
    # y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    # # 按行数(output_size)扩展input
    # x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    #
    # # 构造与输出shape一致的xy_grid,便于将预测的(x,y,w,h)反算到原图
    # xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    #
    # xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    # xy_grid = tf.cast(xy_grid, tf.float32)  # 计算网格左上角坐标，即Cx,Cy
    # # xy_grid shape
    # # (4, 13, 13, 3, 2)
    # # 2:存储的是网格的偏移量,类似与遍历网格
    # # [[[0. 12.]
    # #   [0. 12.]
    # #   [0. 12.]]
    # #
    # # [[1. 12.]
    # #  [1. 12.]
    # #  [1. 12.]]
    # #  ...
    # # [[12. 12.]
    # #  [12. 12.]
    # #  [12. 12.]]]
    #
    # # 计算相对于每个grid的
    # # tf.sigmoid(conv_raw_dxdy): sigmoid使得值处于[0,1],即相对于grid的偏移
    # # 乘以stride: 恢复到最终的输出下的尺度下
    # pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides
    # pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * strides
    # pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    #
    # # 置信度和分类概率使用sigmoid
    # pred_conf = tf.sigmoid(conv_raw_conf)
    # pred_prob = tf.sigmoid(conv_raw_prob)
    # inputs = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    return conv_bbox


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


# x, y, w, h
# def bbox_iou(boxes1, boxes2):
#     boxes1_area = boxes1[..., 2] * boxes1[..., 3]
#     boxes2_area = boxes2[..., 2] * boxes2[..., 3]
#
#     boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
#                         boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
#     boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
#                         boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
#
#     left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
#     right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
#
#     inter_section = tf.maximum(right_down - left_up, 0.0)
#     inter_area = inter_section[..., 0] * inter_section[..., 1]
#     union_area = boxes1_area + boxes2_area - inter_area
#     iou = 1.0 * inter_area / union_area
#
#     return iou


# x, y, w, h
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
    iou = inter_area / tf.maximum(union_area, 1e-7)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-5)
    # giou = tf.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return giou


def focal(target, actual, alpha=1, gamma=2):
    focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss
