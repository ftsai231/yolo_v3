import tensorflow as tf
from Yolo import YoloV3

pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=7.1838.ckpt-26"
output_node_names = ["input/input_data", "yolo_v3_model/pred_sbbox/concat_2", "yolo_v3_model/pred_mbbox/concat_2",
                     "yolo_v3_model/pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=[1, 416, 416, 3], name='input_data')

model = YoloV3(input_data, training=False)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

# print all the nodes in the graph
for v in sess.graph.get_operations():
                print(v.name)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
