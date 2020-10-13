import os
import time

import cv2
import tensorflow as tf
import numpy as np
from Dataset import Dataset
from config import cfg
import utils as utils
from Yolo import YoloV3
from tqdm import tqdm


class YoloTrain(object):
    def __init__(self):
        print("initialize values...")
        tf.reset_default_graph()
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)

        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_epoch = cfg.TRAIN.WARMUP_EPOCHS
        self.epoch = cfg.TRAIN.EPOCH

        # self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = cfg.YOLO.max_output_size

        self.train_logdir = "./train_log/mydata_test1/"
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_epoch = len(self.trainset)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.name_scope('define_input'):
            print("defining inpuuts...")

            self.input_data = tf.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, self.trainset.train_input_size,
                                                                      self.trainset.train_input_size, 3],
                                             name='input_data')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbbox = tf.placeholder(dtype=tf.float32, name='true_sbbox')
            self.true_mbbox = tf.placeholder(dtype=tf.float32, name='true_mbbox')
            self.true_lbbox = tf.placeholder(dtype=tf.float32, name='true_lbbox')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            print("defining loss...")
            self.model = YoloV3(inputs=self.input_data, training=self.trainable)
            self.network_para = tf.global_variables()
            giou_loss, conf_loss, prob_loss = \
                self.model.compute_loss(self.label_sbbox, self.label_mbbox, self.label_lbbox, self.true_sbbox,
                                        self.true_mbbox, self.true_lbbox)
            self.loss = giou_loss + conf_loss + prob_loss

        # TODO: change this part if training doesnt do good
        with tf.name_scope('define_learning_rate'):
            print("defining learning rate...")
            self.learning_rate = cfg.TRAIN.LEARN_RATE_INIT

        with tf.name_scope('learn_rate'):
            print("defining learn rate...")
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            global_step_update = tf.assign_add(self.global_step, 1.0)

        # TODO: change this part if training doesnt do good
        with tf.name_scope("define_weight_decay"):
            print("defining weight decay...")
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('summary'):
            print("summary...")
            tf.summary.scalar("giou_loss", giou_loss)
            tf.summary.scalar("conf_loss", conf_loss)
            tf.summary.scalar("prob_loss", prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            if os.path.exists(self.train_logdir):
                print('pass')
            else:
                os.mkdir(self.train_logdir)

            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.train_logdir, graph=self.sess.graph)

        with tf.name_scope("define_first_stage_train"):
            print("define_first_stage_train")
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                # print("tf.trainable_variables(): ", tf.trainable_variables())
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[1] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            print("define_second_stage_train")
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                      var_list=second_stage_trainable_var_list)
            # second_stage_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
            #                                                                                   var_list=second_stage_trainable_var_list)  # the training step

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        # with tf.name_scope('define_train'):
        #     print("defining train...")
        #     trainable_var_list = tf.trainable_variables()
        #     # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
        #     #                                                                               var_list=trainable_var_list)  # the training step
        #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
        #                                                                                   var_list=trainable_var_list)  # the training step
        #
        #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # normalize batch from the last round
        #         with tf.control_dependencies([optimizer]):
        #             with tf.control_dependencies([moving_ave]):  # decay
        #                 self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            print("defining loader and saver...")
            self.loader = tf.train.Saver(self.network_para)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    def train(self):
        self.sess.run(tf.global_variables_initializer())

        print('=> Restoring weights...')
        self.loader.restore(self.sess, './checkpoint/train_checkpoint/yolov3_model_test_overfit.ckpt')
        print("loaded pretrained model successfully!")

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            print("epoch: ", epoch)
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                        self.input_data: train_data[0],
                        self.label_sbbox: train_data[1],
                        self.label_mbbox: train_data[2],
                        self.label_lbbox: train_data[3],
                        self.true_sbbox: train_data[4],
                        self.true_mbbox: train_data[5],
                        self.true_lbbox: train_data[6],
                        self.trainable: True,
                    })

                train_epoch_loss.append(train_step_loss)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            if epoch != 0 and epoch % 50 == 0:
                ckpt_file = "./checkpoint/train_checkpoint/yolov3_model_test_overfit.ckpt"
                self.saver.save(self.sess, ckpt_file)

            # for test_data in self.testset:
            #     test_step_loss = self.sess.run(self.loss, feed_dict={
            #         self.input_data: test_data[0],
            #         self.label_sbbox: test_data[1],
            #         self.label_mbbox: test_data[2],
            #         self.label_lbbox: test_data[3],
            #         self.true_sbbox: test_data[4],
            #         self.true_mbbox: test_data[5],
            #         self.true_lbbox: test_data[6],
            #         self.trainable: False,
            #     })
            #
            #     test_epoch_loss.append(test_step_loss)
            #
            # train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            # ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            # log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == "__main__":
    YoloTrain().train()
