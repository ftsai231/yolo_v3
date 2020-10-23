import cv2
import numpy as np
import utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "yolo_v3_model/pred_sbbox/concat_2:0", "yolo_v3_model/pred_mbbox/concat_2:0",
                     "yolo_v3_model/pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_path      = "./576527.jpg"
num_classes     = 2
input_size      = 416
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

print("pred_bbox: ", pred_sbbox)
print("pred_bbox shape: ", np.array(pred_sbbox).shape)

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.13)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()
