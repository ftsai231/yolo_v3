import tensorflow as tf

from Yolo_model import darknet53, yolo_conv_block, _ANCHORS, yolo_detection_layer, con2d_fixed_padding, batch_norm, \
    _LEAKY_RELU, upsample, build_boxes, non_max_suppression, _MODEL_SIZE, max_output_size, iou_threshold, threshold



class YoloV3:

    # for building the final yolo model
    def __init__(self, n_classes, model_size, max_output_size, iou_threshold, confidence_threshold, data_format=None):
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

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        with tf.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # normalize values to range [0..1]
            inputs = inputs / 255

            route1, route2, inputs = darknet53(inputs=inputs, training=training, data_format=self.data_format)
            route, inputs = yolo_conv_block(inputs=inputs, filters=512, training=training, data_format=self.data_format)
            detection1 = yolo_detection_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[6:9],
                                              img_size=self.model_size,
                                              data_format=self.data_format)
            inputs = con2d_fixed_padding(route, filters=256, kernel_size=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_output_size = tf.shape(route2).get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)

            # get the channel axis
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat(inputs, route2, axis=axis)

            route, inputs = yolo_conv_block(inputs, filters=256, training=training, data_format=self.data_format)
            detection2 = yolo_detection_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[3:6],
                                              img_size=self.model_size,
                                              data_format=self.data_format)
            inputs = con2d_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_output_size = tf.shape(route1).get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_output_size, data_format=self.data_format)
            inputs = tf.concat(inputs, route1, axis=axis)

            route, inputs = yolo_conv_block(inputs, filters=128, training=training, data_format=self.data_format)
            detection3 = yolo_detection_layer(inputs, n_classes=self.n_classes, anchors=_ANCHORS[0:3],
                                              img_size=self.model_size,
                                              data_format=self.data_format)

            inputs = tf.concat([detection1, detection2, detection3], axis=1)

            inputs = build_boxes(inputs)

            boxes_dict = non_max_suppression(inputs, n_classes=self.n_classes, max_output_size=self.max_output_size,
                                             confidence_threshold=self.confidence_threshold,
                                             iou_thresould=self.iou_threshold)


model = tf.keras.Model(YoloV3(2, _MODEL_SIZE, max_output_size, iou_threshold, threshold))
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
