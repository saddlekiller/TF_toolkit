import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../utils')
from data_provider import *



provider = MNISTProvider('../../data/mnist-train.npz', 50, isShuffle=False)

output_dim = 16
input_dim = 784
hidden_dim1 = 512
hidden_dim2 = 128
hidden_dim3 = 64



graph = tf.Graph()
with graph.as_default():

    def Generator(inputs):

        with tf.variable_scope('Generator'):
            Generator_w1 = tf.get_variable('Generator_w1', initializer = tf.truncated_normal([output_dim, hidden_dim3]))
            Generator_b1 = tf.get_variable('Generator_b1', initializer = tf.truncated_normal([hidden_dim3]))
            Generator_w2 = tf.get_variable('Generator_w2', initializer = tf.truncated_normal([hidden_dim3, hidden_dim2]))
            Generator_b2 = tf.get_variable('Generator_b2', initializer = tf.truncated_normal([hidden_dim2]))
            Generator_w3 = tf.get_variable('Generator_w3', initializer = tf.truncated_normal([hidden_dim2, hidden_dim1]))
            Generator_b3 = tf.get_variable('Generator_b3', initializer = tf.truncated_normal([hidden_dim1]))
            Generator_w  = tf.get_variable('Generator_w' , initializer = tf.truncated_normal([hidden_dim1, input_dim]))
            Generator_b  = tf.get_variable('Generator_b' , initializer = tf.truncated_normal([input_dim]))

            Generator_o1 = tf.nn.relu(tf.add(tf.matmul(inputs      , Generator_w1), Generator_b1))
            Generator_o2 = tf.nn.relu(tf.add(tf.matmul(Generator_o1, Generator_w2), Generator_b2))
            Generator_o3 = tf.nn.relu(tf.add(tf.matmul(Generator_o2, Generator_w3), Generator_b3))
            Generator_o  = tf.nn.relu(tf.add(tf.matmul(Generator_o3, Generator_w ), Generator_b ))

        return Generator_o

    def Discriminator(inputs, reuse = False):

        with tf.variable_scope('Discriminator'):
            Discriminator_w1 = tf.get_variable('Discriminator_w1', initializer = tf.truncated_normal([input_dim, hidden_dim1]), reuse = reuse)
            Discriminator_b1 = tf.get_variable('Discriminator_b1', initializer = tf.truncated_normal([hidden_dim1]), reuse = reuse)
            Discriminator_w2 = tf.get_variable('Discriminator_w2', initializer = tf.truncated_normal([hidden_dim1, hidden_dim2]), reuse = reuse)
            Discriminator_b2 = tf.get_variable('Discriminator_b2', initializer = tf.truncated_normal([hidden_dim2]), reuse = reuse)
            Discriminator_w3 = tf.get_variable('Discriminator_w3', initializer = tf.truncated_normal([hidden_dim2, hidden_dim3]), reuse = reuse)
            Discriminator_b3 = tf.get_variable('Discriminator_b3', initializer = tf.truncated_normal([hidden_dim3]), reuse = reuse)
            Discriminator_w  = tf.get_variable('Discriminator_w', initializer = tf.truncated_normal([hidden_dim3, 1]), reuse = reuse)
            Discriminator_b  = tf.get_variable('Discriminator_b', initializer = tf.truncated_normal([1]), reuse = reuse)

            Discriminator_o1 = tf.nn.relu(tf.add(tf.matmul(inputs          , Discriminator_w1), Discriminator_b1))
            Discriminator_o2 = tf.nn.relu(tf.add(tf.matmul(Discriminator_o1, Discriminator_w2), Discriminator_b2))
            Discriminator_o3 = tf.nn.relu(tf.add(tf.matmul(Discriminator_o2, Discriminator_w3), Discriminator_b3))
            Discriminator_o  = tf.nn.relu(tf.add(tf.matmul(Discriminator_o3, Discriminator_w ), Discriminator_b ))

    data_placeholder = tf.placeholder('')
    Generator_out = Generator()
