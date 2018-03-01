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
            Generator_o  = tf.nn.sigmoid(tf.add(tf.matmul(Generator_o3, Generator_w ), Generator_b ))

        return Generator_o

    def Discriminator(inputs, reuse = False):

        with tf.variable_scope('Discriminator', reuse=reuse):
            Discriminator_w1 = tf.get_variable('Discriminator_w1', initializer = tf.truncated_normal([input_dim, hidden_dim1]))
            Discriminator_b1 = tf.get_variable('Discriminator_b1', initializer = tf.truncated_normal([hidden_dim1]))
            Discriminator_w2 = tf.get_variable('Discriminator_w2', initializer = tf.truncated_normal([hidden_dim1, hidden_dim2]))
            Discriminator_b2 = tf.get_variable('Discriminator_b2', initializer = tf.truncated_normal([hidden_dim2]))
            Discriminator_w3 = tf.get_variable('Discriminator_w3', initializer = tf.truncated_normal([hidden_dim2, hidden_dim3]))
            Discriminator_b3 = tf.get_variable('Discriminator_b3', initializer = tf.truncated_normal([hidden_dim3]))
            Discriminator_w  = tf.get_variable('Discriminator_w', initializer = tf.truncated_normal([hidden_dim3, 1]))
            Discriminator_b  = tf.get_variable('Discriminator_b', initializer = tf.truncated_normal([1]))

            Discriminator_o1 = leaky_relu(tf.add(tf.matmul(inputs          , Discriminator_w1), Discriminator_b1))
            Discriminator_o2 = leaky_relu(tf.add(tf.matmul(Discriminator_o1, Discriminator_w2), Discriminator_b2))
            Discriminator_o3 = leaky_relu(tf.add(tf.matmul(Discriminator_o2, Discriminator_w3), Discriminator_b3))
            Discriminator_o  = tf.identity(tf.add(tf.matmul(Discriminator_o3, Discriminator_w ), Discriminator_b ))
        return Discriminator_o


    data_placeholder  = tf.placeholder(tf.float32, [batch_size, input_dim])
    prior_placeholder = tf.placeholder(tf.float32, [batch_size, output_dim])

    Generator_out          = Generator(prior_placeholder)
    Discriminator_fake_out = Discriminator(Generator_out)
    Discriminator_real_out = Discriminator(data_placeholder, True)

    Discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Discriminator_fake_out, labels = tf.zeros([batch_size, 1])))
    Discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Discriminator_real_out, labels = tf.ones ([batch_size, 1])))
    Generator_loss          = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Discriminator_fake_out, labels = tf.ones ([batch_size, 1])))
    Discriminator_loss      = Discriminator_fake_loss + Discriminator_real_loss

    Generator_variables     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
    Discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")

    Generator_optimizer     = tf.train.RMSPropOptimizer(0.00005).minimize(Generator_loss,     var_list = Generator_variables)
    Discriminator_optimizer = tf.train.RMSPropOptimizer(0.00005) .minimize(Discriminator_loss, var_list = Discriminator_variables)

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
                feed_dict = {data_placeholder: batch_inputs*2 - 1, prior_placeholder: noise}
                _, d_loss          = sess.run([Discriminator_optimizer, Discriminator_loss]       , feed_dict = feed_dict)
            noise = np.random.uniform(-1, 1, [batch_size, output_dim]).astype(np.float32)
            feed_dict = {data_placeholder: batch_inputs*2 - 1, prior_placeholder: noise}
            _, g_loss, g_image = sess.run([Generator_optimizer, Generator_loss, Generator_out], feed_dict = feed_dict)

            d_losses.append(d_loss)
            g_losses.append(g_loss)
        print('EPOCH %d, D_LOSS: %f, G_LOSS: %f '%(i, np.mean(d_losses), np.mean(g_losses)))
        g_image = g_image.reshape([-1, 28, 28]).transpose([1,2,0])
        merge_image = build_image(g_image, 10)
        plt.imsave(str(i)+'.png', merge_image)
