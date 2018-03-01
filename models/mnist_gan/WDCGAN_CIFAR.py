import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../utils')
from data_provider import *
from tools import *
from tensorflow.contrib.layers import *


batch_size = 100
provider = MNISTProvider('../../data/cifar-10-train.npz', batch_size, isShuffle=False)

output_dim = 512
input_dim = 784
hidden_dim1 = 512
hidden_dim2 = 128
hidden_dim3 = 64

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


graph = tf.Graph()
with graph.as_default():

    def Generator(inputs):
    #
        with tf.variable_scope('Generator'):

            Generator_affine_1  = fully_connected(inputs, 256,
                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                activation_fn=tf.nn.relu)
            Generator_reshape_1 = tf.reshape(Generator_affine_1, [batch_size, 1, 1, 256])
            Generator_deconv_1  = convolution2d_transpose(Generator_reshape_1, 64, [2, 2], [2, 2],
                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                activation_fn=tf.nn.relu)

            Generator_deconv_2  = convolution2d_transpose(Generator_deconv_1, 128, [2, 2], [2, 2],
                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                activation_fn=tf.nn.relu)

            Generator_deconv_3  = convolution2d_transpose(Generator_deconv_2, 64, [2, 2], [2, 2],
                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                activation_fn=tf.nn.relu)

            Generator_deconv_4  = convolution2d_transpose(Generator_deconv_3, 16, [2, 2], [2, 2],
                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                activation_fn=tf.nn.relu)

            Generator_deconv_5  = convolution2d_transpose(Generator_deconv_4, 3, [2, 2], [2, 2],
                                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                                activation_fn=tf.nn.sigmoid)

            print('*'*49)
            print('*'*1 + ' '*19 + 'Generator' + ' '*19 + '*'*1)
            print('*'*49)
            print(Generator_affine_1)
            print(Generator_deconv_1)
            print(Generator_deconv_2)
            print(Generator_deconv_3)
            print(Generator_deconv_4)
            print(Generator_deconv_5)
            print('*'*49)


        return Generator_deconv_5

    def Discriminator(inputs, reuse = False):

        with tf.variable_scope('Discriminator', reuse=reuse):

            Discriminator_k1 = tf.get_variable('Discriminator_k1', initializer = tf.truncated_normal([5, 5, 3, 16]))
            Discriminator_b1 = tf.get_variable('Discriminator_b1', initializer = tf.truncated_normal([16]))
            Discriminator_k2 = tf.get_variable('Discriminator_k2', initializer = tf.truncated_normal([5, 5, 16, 64]))
            Discriminator_b2 = tf.get_variable('Discriminator_b2', initializer = tf.truncated_normal([64]))
            Discriminator_k3 = tf.get_variable('Discriminator_k3', initializer = tf.truncated_normal([5, 5, 64, 128]))
            Discriminator_b3 = tf.get_variable('Discriminator_b3', initializer = tf.truncated_normal([128]))
            Discriminator_w4 = tf.get_variable('Discriminator_w4', initializer = tf.truncated_normal([128, 64]))
            Discriminator_b4 = tf.get_variable('Discriminator_b4', initializer = tf.truncated_normal([64]))
            Discriminator_w5 = tf.get_variable('Discriminator_w5', initializer = tf.truncated_normal([64, 16]))
            Discriminator_b5 = tf.get_variable('Discriminator_b5', initializer = tf.truncated_normal([16]))
            Discriminator_w6 = tf.get_variable('Discriminator_w6', initializer = tf.truncated_normal([16,  1]))
            Discriminator_b6 = tf.get_variable('Discriminator_b6', initializer = tf.truncated_normal([ 1]))

            Discriminator_conv_1    = leaky_relu(tf.add(tf.nn.conv2d(input = inputs,                  filter = Discriminator_k1, padding = 'VALID', strides = [1, 1, 1, 1]), Discriminator_b1))
            Discriminator_pooling_1 = tf.nn.max_pool(Discriminator_conv_1, padding = 'VALID', strides = [1, 2, 2, 1], ksize = [1, 2, 2, 1])
            Discriminator_conv_2    = leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_pooling_1, filter = Discriminator_k2, padding = 'VALID', strides = [1, 1, 1, 1]), Discriminator_b2))
            Discriminator_pooling_2 = tf.nn.max_pool(Discriminator_conv_2, padding = 'VALID', strides = [1, 2, 2, 1], ksize = [1, 2, 2, 1])
            Discriminator_conv_3    = leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_pooling_2, filter = Discriminator_k3, padding = 'VALID', strides = [1, 1, 1, 1]), Discriminator_b3))
            Discriminator_reshape_1 = tf.reshape(Discriminator_conv_3, [-1, 128])
            Discriminator_affine_1  = leaky_relu(tf.add(tf.matmul(Discriminator_reshape_1, Discriminator_w4), Discriminator_b4))
            Discriminator_affine_2  = leaky_relu(tf.add(tf.matmul(Discriminator_affine_1 , Discriminator_w5), Discriminator_b5))
            Discriminator_affine_3  = leaky_relu(tf.add(tf.matmul(Discriminator_affine_2 , Discriminator_w6), Discriminator_b6))
            #
            print('*'*53)
            print('*'*1 + ' '*19 + 'Discriminator' + ' '*19 + '*'*1)
            print('*'*53)
            print(inputs)
            print(Discriminator_conv_1)
            print(Discriminator_pooling_1)
            print(Discriminator_conv_2)
            print(Discriminator_pooling_2)
            print(Discriminator_conv_3)
            print(Discriminator_reshape_1)
            print(Discriminator_affine_1)
            print(Discriminator_affine_2)
            print(Discriminator_affine_3)
            print('*'*53)

        # return Discriminator_affine_3


    data_placeholder  = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    prior_placeholder = tf.placeholder(tf.float32, [batch_size, output_dim])
    #
    Generator_out          = Generator(prior_placeholder)
    Discriminator_fake_out = Discriminator(Generator_out, False)
    Discriminator_real_out = Discriminator(data_placeholder, True)

    Generator_loss          = tf.reduce_mean(Discriminator_fake_out)
    Discriminator_loss      = tf.reduce_mean(Discriminator_real_out) - tf.reduce_mean(Discriminator_fake_out)

    Generator_variables     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
    Discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
    # print(Generator_variables)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        Generator_optimizer     = tf.train.RMSPropOptimizer(0.0001).minimize(Generator_loss,     var_list = Generator_variables)
        Discriminator_optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(Discriminator_loss, var_list = Discriminator_variables)


    Clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in Discriminator_variables]
    # a = [var for var in tf.global_variables() if 'Discriminator' in var.name]
    # Clip = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")]

    # print('-'*50)
    # for v in Clip:
    #     print(v)
    # print('-'*50)
    # for ai in a:
    #     print(ai)
    # print('-'*50)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # merged_all = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tensorboard', sess.graph)


    for i in range(500):
        d_losses = []
        g_losses = []
        for batch_inputs, batch_targets in provider:
            for j in range(5):
                noise = np.random.uniform(-1, 1, [batch_size, output_dim]).astype(np.float32)
                feed_dict = {data_placeholder: batch_inputs.reshape(-1, 28, 28, 1), prior_placeholder: noise}
                _, d_loss          = sess.run([Discriminator_optimizer, Discriminator_loss]       , feed_dict = feed_dict)
                sess.run(Clip)
            noise = np.random.uniform(-1, 1, [batch_size, output_dim]).astype(np.float32)
            feed_dict = {data_placeholder: batch_inputs.reshape(-1, 28, 28, 1), prior_placeholder: noise}
            _, g_loss, g_image = sess.run([Generator_optimizer, Generator_loss, Generator_out], feed_dict = feed_dict)

            d_losses.append(d_loss)
            g_losses.append(g_loss)
        print('EPOCH %d, D_LOSS: %f, G_LOSS: %f '%(i, np.mean(d_losses), np.mean(g_losses)))
        g_image = g_image.reshape([-1, 28, 28]).transpose([1,2,0])
        merge_image = build_image(g_image, 10)
        plt.imsave(str(i)+'.png', merge_image)
