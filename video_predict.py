import cv2
import numpy as np
import utils as utils
import tensorflow as tf
from PIL import Image


return_elements = ["input/input_data:0", "yolo_v3_model/pred_sbbox/concat_2:0", "yolo_v3_model/pred_mbbox/concat_2:0",
                     "yolo_v3_model/pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "./landon.mov"
num_classes     = 2
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    success, frame = vid.read()
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # VideoWriter_fourcc为视频编解码器，20为帧播放速率
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output_3.mp4', fourcc, 20.0, size)
    num_frame = 0

    while success:
        frame_size = frame.shape[:2]
        image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.4)
        bboxes = utils.nms(bboxes, 0.3, method='nms')
        image = utils.draw_bbox(frame, bboxes)
        out.write(image)

        result = np.asarray(image)
        success, frame = vid.read()
        num_frame += 1
        print("number of frame: ", num_frame)

    vid.release()
    out.release()
    print("end of program")
