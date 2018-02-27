import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../utils')
from data_provider import *
import pickle
import matplotlib.pyplot as plt


provider = MNISTProvider('../../data/mnist-valid.npz', 50)
means = pickle.load(open('means.npz', 'rb'))

input_dim = 784
hidden_dim1 = 256
hidden_dim2 = 64
hidden_dim3 = 16

graph = tf.Graph()
with graph.as_default():
    inputs_placeholder = tf.placeholder(tf.float32, [None, input_dim])
    code_placeholder = tf.placeholder(tf.float32, [None, hidden_dim3])
    with tf.variable_scope('encoder_1'):
        e_w1 = tf.get_variable('e_w1', shape = [input_dim, hidden_dim1], initializer = tf.truncated_normal_initializer)
        e_b1 = tf.get_variable('e_b1', shape = [hidden_dim1], initializer = tf.truncated_normal_initializer)
        e_o1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs_placeholder, e_w1), e_b1))

    with tf.variable_scope('encoder_2'):
        e_w2 = tf.get_variable('e_w2', shape = [hidden_dim1, hidden_dim2], initializer = tf.truncated_normal_initializer)
        e_b2 = tf.get_variable('e_b2', shape = [hidden_dim2], initializer = tf.truncated_normal_initializer)
        e_o2 = tf.nn.sigmoid(tf.add(tf.matmul(e_o1, e_w2), e_b2))

    with tf.variable_scope('encoder_3'):
        e_w3 = tf.get_variable('e_w3', shape = [hidden_dim2, hidden_dim3], initializer = tf.truncated_normal_initializer)
        e_b3 = tf.get_variable('e_b3', shape = [hidden_dim3], initializer = tf.truncated_normal_initializer)
        e_o3 = tf.nn.sigmoid(tf.add(tf.matmul(e_o2, e_w3), e_b3))

    with tf.variable_scope('decoder_1'):
        d_w1 = tf.get_variable('d_w1', shape = [hidden_dim3, hidden_dim2], initializer = tf.truncated_normal_initializer)
        d_b1 = tf.get_variable('d_b1', shape = [hidden_dim2], initializer = tf.truncated_normal_initializer)
        d_o1 = tf.nn.sigmoid(tf.add(tf.matmul(e_o3, d_w1), d_b1))
        rebuild_o1 = tf.nn.sigmoid(tf.add(tf.matmul(code_placeholder, d_w1), d_b1))

    with tf.variable_scope('decoder_2'):
        d_w2 = tf.get_variable('d_w2', shape = [hidden_dim2, hidden_dim1], initializer = tf.truncated_normal_initializer)
        d_b2 = tf.get_variable('d_b2', shape = [hidden_dim1], initializer = tf.truncated_normal_initializer)
        d_o2 = tf.nn.sigmoid(tf.add(tf.matmul(d_o1, d_w2), d_b2))
        rebuild_o2 = tf.nn.sigmoid(tf.add(tf.matmul(rebuild_o1, d_w2), d_b2))

    with tf.variable_scope('decoder_3'):
        d_w3 = tf.get_variable('d_w3', shape = [hidden_dim1, input_dim], initializer = tf.truncated_normal_initializer)
        d_b3 = tf.get_variable('d_b3', shape = [input_dim], initializer = tf.truncated_normal_initializer)
        d_o3 = tf.nn.sigmoid(tf.add(tf.matmul(d_o2, d_w3), d_b3))
        rebuild_o3 = tf.nn.sigmoid(tf.add(tf.matmul(rebuild_o2, d_w3), d_b3))
        rebuild = tf.reshape(rebuild_o3, [-1, 28, 28])

    loss = tf.reduce_mean((d_o3 - inputs_placeholder)**2)
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, 'models/model.ckpt')
    # hidden_units = {}
    t = 0
    for batch_inputs, batch_targets in provider:
        feed_dict = {inputs_placeholder: batch_inputs, code_placeholder:np.zeros((1, hidden_dim3))}
        features = sess.run(e_o3, feed_dict = feed_dict)
        real_targets = np.argmax(batch_targets, 1)
        values = np.argmax(features.dot(means.T), 1)
        # plt.stem(batch_targets[2])
        # plt.stem(features.dot(means.T)[2])

        t += np.sum(real_targets == values)
    print('acc: %f'%(t/provider._n_samples))
        # break
    # plt.show()
