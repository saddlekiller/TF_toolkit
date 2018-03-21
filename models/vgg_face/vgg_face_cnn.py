import tensorflow as tf
import sys
sys.path.append('../../utils')
from data_provider import *

batch_size = 50
image_shape = [256, 128, 3]
image_dir = '../../data/FACE/reshape_images'
label_dir = '../../data/FACE/reshape_labels.txt'

provider = VGGFaceProvider(image_dir, label_dir, batch_size, 'png')
vgg_loc_graph = tf.Graph()

with vgg_loc_graph.as_default():

    image_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]])
    bbox_placeholder = tf.placeholder(tf.float32, [None, 4])
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(inputs = image_placeholder, filters = 16, kernel_size = [64, 64], padding = 'same', activation = tf.nn.relu)
        layers.append(conv1)

    with tf.variable_scope('avg_pool1'):
        avg_pool1 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool1)

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(inputs = layers[-1], filters = 32, kernel_size = [32, 32], padding = 'same', activation = tf.nn.relu)
        layers.append(conv2)

    with tf.variable_scope('avg_pool12'):
        avg_pool2 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool2)

    with tf.variable_scope('conv3'):
        conv3 = tf.layers.conv2d(inputs = layers[-1], filters = 64, kernel_size = [16, 16], padding = 'same', activation = tf.nn.relu)
        layers.append(conv3)

    with tf.variable_scope('avg_pool13'):
        avg_pool13 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool13)

    with tf.variable_scope('conv4'):
        conv4 = tf.layers.conv2d(inputs = layers[-1], filters = 128, kernel_size = [8, 8], padding = 'same', activation = tf.nn.relu)
        layers.append(conv4)

    with tf.variable_scope('avg_pool14'):
        avg_pool14 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool14)

    with tf.variable_scope('conv5'):
        conv5 = tf.layers.conv2d(inputs = layers[-1], filters = 256, kernel_size = [4, 4], padding = 'same', activation = tf.nn.relu)
        layers.append(conv5)

    with tf.variable_scope('avg_pool15'):
        avg_pool15 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool15)

    with tf.variable_scope('conv6'):
        conv6 = tf.layers.conv2d(inputs = layers[-1], filters = 512, kernel_size = [2, 2], padding = 'same', activation = tf.nn.relu)
        layers.append(conv6)

    with tf.variable_scope('avg_pool16'):
        avg_pool16 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool16)

    with tf.variable_scope('conv7'):
        conv7 = tf.layers.conv2d(inputs = layers[-1], filters = 1024, kernel_size = [1, 1], padding = 'same', activation = tf.nn.relu)
        layers.append(conv7)

    with tf.variable_scope('avg_pool17'):
        avg_pool17 = tf.layers.average_pooling2d(inputs = layers[-1], pool_size = 2, strides = 2)
        layers.append(avg_pool17)

    with tf.variable_scope('reshape1'):
        reshape1 = tf.reshape(layers[-1], [-1, 2048])
        layers.append(reshape1)

    with tf.variable_scope('affine1'):
        affine1 = tf.layers.dense(inputs=layers[-1], units=512, activation = tf.nn.relu)
        layers.append(affine1)

    with tf.variable_scope('affine2'):
        affine2 = tf.layers.dense(inputs=layers[-1], units=128, activation = tf.nn.relu)
        layers.append(affine2)

    with tf.variable_scope('affine3'):
        affine3 = tf.layers.dense(inputs=layers[-1], units=4, activation = tf.nn.relu)
        layers.append(affine3)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum((layers[-1]-bbox_placeholder)**2))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    for layer in layers:
        print(layer)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):

        errs = []
        for batch_images, batch_locs in provider:
            feed_dict = {image_placeholder:batch_images, bbox_placeholder:batch_locs[:, :4]}
            _, err = sess.run([optimizer, loss], feed_dict = feed_dict)
            errs.append(err)
        print('Epoch %d, Loss => %f' % (i, np.mean(errs)))
