import tensorflow as tf
from Yolo import YoloV3
from config import cfg
import numpy as np


np.set_printoptions(threshold=np.inf)
pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/train_checkpoint/yolov3_test_loss=14.3637.ckpt-700"
output_node_names = ["input/input_data", "yolo_v3_model/pred_bbox/concat"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, 416,
                                                    416, 3], name='input_data')

model = YoloV3(input_data, training=False)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

'''print all the nodes in the graph. DO NOT DELETE'''
# for v in sess.graph.get_operations():
#                 print(v.name)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
