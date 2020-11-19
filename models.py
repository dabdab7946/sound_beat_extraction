

import numpy as np
import matplotlib.pyplot as plt
import cv2


import tensorflow as tf





# =====================================================================================================
# Beat-type Classfication 모델

class BeatClassifier:
    def __init__(self, graph, model_name, device='',
                 FFT_SIZE=512, n_classes=3, C_LAYERS=2, D_LAYERS=2, KERNEL=128,
                 learning_rate=1e-5, cost_wgt=1):
        self.Graph = graph
        self.Device = device
        self.ModelName = model_name
        self.FFT_SIZE = FFT_SIZE
        print(model_name)
        #
        # -----------------------------------------------------------
        # Make Network
        self.CONV_LAYERS = C_LAYERS
        self.DENSE_LAYERS = D_LAYERS
        self.DENSE_NODE = 32
        #
        KERNEL = KERNEL
        FILTER = 32
        POOL = 2
        DROP_OUT = 0.7
        STRIDE = 2
        tf.set_random_seed(777)
        #
        self.X = tf.placeholder(tf.float32, [None, FFT_SIZE, 1])
        self.Y = tf.placeholder(tf.int64, [None, n_classes])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        net_sig = self.X
        ffwd = net_sig
        for n in range(self.CONV_LAYERS):
            print('net_sig', net_sig.shape)
            FILTER = FILTER * (n % 2 + 1)
            net_sig = tf.layers.conv1d(inputs=net_sig, filters=FILTER, kernel_size=KERNEL, padding='same', activation=tf.nn.relu)
            net_sig = tf.layers.max_pooling1d(inputs=net_sig, pool_size=POOL, strides=STRIDE, padding='same')
            net_sig = tf.layers.batch_normalization(inputs=net_sig, center=True, scale=True, training=self.IS_TRAIN)
            net_sig = tf.layers.dropout(inputs=net_sig, rate=DROP_OUT, training=self.IS_TRAIN)
            ffwd = tf.layers.max_pooling1d(inputs=ffwd, pool_size=POOL, strides=STRIDE, padding='same')
            net_sig = tf.concat([net_sig, ffwd], axis=2)
        # -----------------------------------------------------------
        # Flatten and Concatenation
        self.net_sig_cam = net_sig
        #
        net_sig_flat = tf.reshape(net_sig, [-1, net_sig.shape[1]._value * net_sig.shape[2]._value])
        self.net_flat = net_sig_flat
        print('net_flat:', self.net_flat)
        for i in range(self.DENSE_LAYERS):
            self.net_flat = tf.layers.dense(self.net_flat, self.DENSE_NODE, activation=tf.nn.relu)
            self.net_flat = tf.layers.dropout(self.net_flat, DROP_OUT)
            self.DENSE_NODE = self.DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net_flat, n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.nn.softmax(self.logits)
        print('net_flat:', self.net_flat, ', net_sig_cam:', self.net_sig_cam)
        #
        # -----------------------------------------------------------
        # for CAM
        NET_DEPTH = self.net_sig_cam.shape[2]._value
        self.gap = tf.reduce_mean(self.net_sig_cam, (1))
        self.gap_w = tf.get_variable('cam_w1', shape=[NET_DEPTH, n_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
        # self.sig_cam = get_class_map(0, self.net_sig_cam, FFT_SIZE, self.gap_w)
        # -----------------------------------------------------------
        # Batch Normalization
        self.cost = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.logits, weights=cost_wgt))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        # -----------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # -----------------------------------------------------------
        # Make Session
        if self.Device != '':
            self.config = tf.ConfigProto()
            self.config.gpu_options.visible_device_list = self.Device
            self.sess = tf.Session(graph=self.Graph, config=self.config)
        else:
            self.sess = tf.Session(graph=self.Graph)
        #
        self.sess.run(tf.global_variables_initializer())
        if model_name != '':
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except Exception as ex:
                print(str(ex), model_name)
    #
    # ------------------------------------------------------------------------------------------
    # Training 함수
    def train(self, x, y, is_train):
        if is_train:
            _ = self.sess.run(self.optimizer,
                              feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        c, l, p, a = self.sess.run([self.cost, self.logits, self.predict, self.accuracy],
                              feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    #
    # Testing 함수
    def test(self, x):
        l, p = self.sess.run([self.logits, self.predict],
                            feed_dict={self.X: x, self.IS_TRAIN: False})
        return l, p

    # 모델 저장
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        return


