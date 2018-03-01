import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../utils')
from data_provider import *
from tools import *



batch_size = 100
provider = MNISTProvider('../../data/mnist-train.npz', batch_size, isShuffle=True)

output_dim = 16
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

            Generator_w1 = tf.get_variable('Generator_w1', initializer = tf.truncated_normal([output_dim, 128]))
            Generator_b1 = tf.get_variable('Generator_b1', initializer = tf.truncated_normal([128]))
            Generator_k2 = tf.get_variable('Generator_w2', initializer = tf.truncated_normal([7, 7, 64, 128]))
            Generator_b2 = tf.get_variable('Generator_b2', initializer = tf.truncated_normal([64]))
            Generator_k3 = tf.get_variable('Generator_w3', initializer = tf.truncated_normal([2, 2, 16, 64]))
            Generator_b3 = tf.get_variable('Generator_b3', initializer = tf.truncated_normal([16]))
            Generator_k4 = tf.get_variable('Generator_w4', initializer = tf.truncated_normal([2, 2, 1, 16]))
            Generator_b4 = tf.get_variable('Generator_b4', initializer = tf.truncated_normal([1]))

            Generator_affine_1  = tf.nn.relu(tf.add(tf.matmul(inputs      , Generator_w1), Generator_b1))
            Generator_reshape_1 = tf.reshape(Generator_affine_1, [batch_size, 1, 1, 128])
            Generator_deconv_1  = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(Generator_reshape_1, Generator_k2, [batch_size, 7, 7, 64], [1, 7, 7, 1], 'SAME'), Generator_b2))
            Generator_deconv_2  = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(Generator_deconv_1, Generator_k3, [batch_size, 14, 14, 16], [1, 2, 2, 1], 'SAME'), Generator_b3))
            Generator_deconv_3  = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(Generator_deconv_2, Generator_k4, [batch_size, 28, 28, 1], [1, 2, 2, 1], 'SAME'), Generator_b4))

            print('*'*49)
            print('*'*1 + ' '*19 + 'Generator' + ' '*19 + '*'*1)
            print('*'*49)
            print(Generator_affine_1)
            print(Generator_reshape_1)
            print(Generator_deconv_1)
            print(Generator_deconv_2)
            print(Generator_deconv_3)
            print('*'*49)

        return Generator_deconv_3

    def Discriminator(inputs, reuse = False):

        with tf.variable_scope('Discriminator', reuse=reuse):

            Discriminator_k1 = tf.get_variable('Discriminator_k1', initializer = tf.truncated_normal([5, 5, 1, 16]))
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

            Discriminator_conv_1    = leaky_relu(tf.add(tf.nn.conv2d(input = inputs,                  filter = Discriminator_k1, padding = 'SAME', strides = [1, 1, 1, 1]), Discriminator_b1))
            Discriminator_pooling_1 = tf.nn.max_pool(Discriminator_conv_1, padding = 'VALID', strides = [1, 2, 2, 1], ksize = [1, 2, 2, 1])
            Discriminator_conv_2    = leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_pooling_1, filter = Discriminator_k2, padding = 'VALID', strides = [1, 1, 1, 1]), Discriminator_b2))
            Discriminator_pooling_2 = tf.nn.max_pool(Discriminator_conv_2, padding = 'VALID', strides = [1, 2, 2, 1], ksize = [1, 2, 2, 1])
            Discriminator_conv_3    = leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_pooling_2, filter = Discriminator_k3, padding = 'VALID', strides = [1, 1, 1, 1]), Discriminator_b3))
            Discriminator_reshape_1 = tf.reshape(Discriminator_conv_3, [-1, 128])
            Discriminator_affine_1  = leaky_relu(tf.add(tf.matmul(Discriminator_reshape_1, Discriminator_w4), Discriminator_b4))
            Discriminator_affine_2  = leaky_relu(tf.add(tf.matmul(Discriminator_affine_1 , Discriminator_w5), Discriminator_b5))
            Discriminator_affine_3  = leaky_relu(tf.add(tf.matmul(Discriminator_affine_2 , Discriminator_w6), Discriminator_b6))

            print('*'*53)
            print('*'*1 + ' '*19 + 'Discriminator' + ' '*19 + '*'*1)
            print('*'*53)
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

        return Discriminator_affine_3


    data_placeholder  = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
    prior_placeholder = tf.placeholder(tf.float32, [batch_size, output_dim])
    #
    Generator_out          = Generator(prior_placeholder)
    Discriminator_fake_out = Discriminator(Generator_out)
    Discriminator_real_out = Discriminator(data_placeholder, True)

    Generator_loss          = tf.reduce_mean(Discriminator_real_out)
    Discriminator_loss      = tf.reduce_mean(Discriminator_fake_out) - tf.reduce_mean(Discriminator_real_out)

    Generator_variables     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
    Discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")

    Generator_optimizer     = tf.train.RMSPropOptimizer(0.00005).minimize(Generator_loss,     var_list = Generator_variables)
    Discriminator_optimizer = tf.train.RMSPropOptimizer(0.00005) .minimize(Discriminator_loss, var_list = Discriminator_variables)
    Clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")]

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
                sess.run(clip)
                _, d_loss          = sess.run([Discriminator_optimizer, Discriminator_loss]       , feed_dict = feed_dict)
            noise = np.random.uniform(-1, 1, [batch_size, output_dim]).astype(np.float32)
            feed_dict = {data_placeholder: batch_inputs.reshape(-1, 28, 28, 1), prior_placeholder: noise}
            _, g_loss, g_image = sess.run([Generator_optimizer, Generator_loss, Generator_out], feed_dict = feed_dict)

            d_losses.append(d_loss)
            g_losses.append(g_loss)
        print('EPOCH %d, D_LOSS: %f, G_LOSS: %f '%(i, np.mean(d_losses), np.mean(g_losses)))
        g_image = g_image.reshape([-1, 28, 28]).transpose([1,2,0])
        merge_image = build_image(g_image, 10)
        plt.imsave(str(i)+'.png', merge_image)
