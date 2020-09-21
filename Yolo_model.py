import tensorflow as tf
import numpy as np

cell_size = 7
box_per_cell = 2
classes = ["car", "person"]
num_of_class = len(classes)
threshold = 0.5  # confidence score threshold
iou_threshold = 0.5
max_output_size = 5

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        scale=True, training=training)


def fix_padding(inputs, kernel_size, mode='CONSTANT', **kwargs):
    pad_tot = kernel_size - 1
    pad_beg = pad_tot // 2
    pad_end = pad_tot - pad_beg

    if kwargs['data_format'] == 'NCHW':
        pad_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
    else:
        pad_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode=mode)

    return pad_inputs


# use stride to control the output size
def con2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fix_padding(inputs, kernel_size)
    return tf.keras.layers.Conv2D(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)


def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    residual = inputs
    inputs = tf.keras.layers.Conv2D(inputs, filters, kernel_size=1, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, )

    inputs = tf.keras.layers.Conv2D(inputs, filters * 2, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, )

    inputs = inputs + residual

    return inputs


def darknet53(inputs, training, data_format):
    inputs = con2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=64, kernel_size=3, data_format=data_format, strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=32, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=128, kernel_size=3, training=training, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=256, kernel_size=3, training=training, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=512, kernel_size=3, training=training, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    route1 = inputs

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256, training=training, data_format=data_format)

    inputs = con2d_fixed_padding(inputs, filters=1024, kernel_size=3, training=training, data_format=data_format,
                                 strides=2)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    route2 = inputs

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512, training=training, data_format=data_format)

    return route1, route2, inputs


def yolo_conv_block(inputs, filters, training, data_format):
    inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = con2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    route = inputs

    inputs = con2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    return route, inputs


def yolo_detection_layer(inputs, n_classes, anchors, img_size, data_format):
    n_anchors = len(anchors)

    # detection kernel
    inputs = tf.keras.layers.Conv2D(inputs, filters=n_anchors*(5+n_classes), kernel_size=1, strides=1, use_bias=True,
                                    data_format=data_format)

    shape = tf.shape(inputs).get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                 5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    box_center, box_shape, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_center)
    box_centers = (box_centers + x_y_offset) * strides


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
def non_max_suppression(inputs, n_classes, max_output_size, confidence_threshold, iou_thresould):
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        # the 4th place is the confidence score
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        # get the classes
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        # add one more dim in classes
        classes = tf.expand_dims(tf.to_float(classes), axis=-1)
        # add the result in the last dim in the box
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
