import tensorflow as tf
import pickle
import sys
sys.path.append('../../utils')
from data_provider import *
from tools import *
import matplotlib.pyplot as plt
import os

provider = CIFARProvider('../../data/cifar-10-valid.npz', 100)
graph = tf.Graph()
with graph.as_default():
    layers = dict()
    layer_names = []
    inputs_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    targets_placeholder = tf.placeholder(tf.float32, [None, 10])
    layer_names.append('conv1')
    with tf.variable_scope(layer_names[-1]):
        kernel_size = 5
        input_dim = 3
        output_dim = 16
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['kernel'] = tf.get_variable(name = 'kernel', shape = [kernel_size, kernel_size, input_dim, output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['bias'] = tf.get_variable(name = 'bias', shape = [output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['hidden'] = tf.nn.relu(tf.add(tf.nn.conv2d(input = inputs_placeholder, filter = layers[layer_names[-1]]['kernel'], strides = [1, 1, 1, 1], padding = "VALID"), layers[layer_names[-1]]['bias']))
        print(layers[layer_names[-1]])
    # 32 x 32 x 3 => 28 x 28 x 16
    layer_names.append('pooling1')
    with tf.variable_scope(layer_names[-1]):
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['hidden'] = tf.layers.average_pooling2d(inputs = layers[layer_names[-2]]['hidden'], pool_size = 2, strides = 2)
        print(layers[layer_names[-1]])

    layer_names.append('conv2')
    with tf.variable_scope(layer_names[-1]):
        kernel_size = 5
        input_dim = 16
        output_dim = 64
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['kernel'] = tf.get_variable(name = 'kernel', shape = [kernel_size, kernel_size, input_dim, output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['bias'] = tf.get_variable(name = 'bias', shape = [output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['hidden'] = tf.nn.relu(tf.add(tf.nn.conv2d(input = layers[layer_names[-2]]['hidden'], filter = layers[layer_names[-1]]['kernel'], strides = [1, 1, 1, 1], padding = "VALID"), layers[layer_names[-1]]['bias']))
        print(layers[layer_names[-1]])

    layer_names.append('pooling2')
    with tf.variable_scope(layer_names[-1]):
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['hidden'] = tf.layers.average_pooling2d(inputs = layers[layer_names[-2]]['hidden'], pool_size = 2, strides = 2)
        print(layers[layer_names[-1]])

    layer_names.append('conv3')
    with tf.variable_scope(layer_names[-1]):
        kernel_size = 5
        input_dim = 64
        output_dim = 128
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['kernel'] = tf.get_variable(name = 'kernel', shape = [kernel_size, kernel_size, input_dim, output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['bias'] = tf.get_variable(name = 'bias', shape = [output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['hidden'] = tf.nn.relu(tf.add(tf.nn.conv2d(input = layers[layer_names[-2]]['hidden'], filter = layers[layer_names[-1]]['kernel'], strides = [1, 1, 1, 1], padding = "VALID"), layers[layer_names[-1]]['bias']))
        print(layers[layer_names[-1]])

    layer_names.append('reshape1')
    with tf.variable_scope(layer_names[-1]):
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['hidden'] = tf.reshape(layers[layer_names[-2]]['hidden'], [-1, 128])
        print(layers[layer_names[-1]])

    layer_names.append('affine1')
    with tf.variable_scope(layer_names[-1]):
        input_dim = 128
        output_dim = 128
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['weight'] = tf.get_variable(name = 'weight', shape = [input_dim, output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['bias'] = tf.get_variable(name = 'bias', shape = [output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['hidden'] = tf.nn.relu(tf.add(tf.matmul(layers[layer_names[-2]]['hidden'], layers[layer_names[-1]]['weight']), layers[layer_names[-1]]['bias']))
        print(layers[layer_names[-1]])

    layer_names.append('affine2')
    with tf.variable_scope(layer_names[-1]):
        input_dim = 128
        output_dim = 64
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['weight'] = tf.get_variable(name = 'weight', shape = [input_dim, output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['bias'] = tf.get_variable(name = 'bias', shape = [output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['hidden'] = tf.nn.relu(tf.add(tf.matmul(layers[layer_names[-2]]['hidden'], layers[layer_names[-1]]['weight']), layers[layer_names[-1]]['bias']))
        print(layers[layer_names[-1]])

    layer_names.append('affine3')
    with tf.variable_scope(layer_names[-1]):
        input_dim = 64
        output_dim = 10
        layers[layer_names[-1]] = dict()
        layers[layer_names[-1]]['weight'] = tf.get_variable(name = 'weight', shape = [input_dim, output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['bias'] = tf.get_variable(name = 'bias', shape = [output_dim], initializer = tf.truncated_normal_initializer)
        layers[layer_names[-1]]['hidden'] = tf.identity(tf.add(tf.matmul(layers[layer_names[-2]]['hidden'], layers[layer_names[-1]]['weight']), layers[layer_names[-1]]['bias']))
        print(layers[layer_names[-1]])

    for layer in layers.values():
        print(layer['hidden'].shape)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = layers[layer_names[-1]]['hidden'], labels = targets_placeholder))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layers[layer_names[-1]]['hidden'], 1), tf.argmax(targets_placeholder, 1)), tf.float32))

    loss_sum = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = layers[layer_names[-1]]['hidden'], labels = targets_placeholder))
    accuracy_sum = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(layers[layer_names[-1]]['hidden'], 1), tf.argmax(targets_placeholder, 1)), tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("acc", accuracy)


    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    merged_all = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tensorboard', sess.graph)

    # tf.summary.image("conv1_hidden", build_image(layers['conv1']['hidden'][0].eval(), 4).reshape([1, 132, 132, 1]))
    try:
        os.mkdir('outputs')
        os.mkdir('outputs/conv1')
        os.mkdir('outputs/conv2')
        os.mkdir('outputs/pooling1')
        os.mkdir('outputs/pooling2')
    except:
        pass
    sample_inputs = None
    sample_targets = None
    flag = False
    for i in range(100):
        accs = 0.0
        errs = 0.0
        for batch_inputs, batch_targets in provider:
            _, merge, err, acc = sess.run([optimizer, merged_all, loss_sum, accuracy_sum], feed_dict = {inputs_placeholder:batch_inputs, targets_placeholder:batch_targets})
            # print(err, acc)
            accs += acc
            errs += err
        print('Epoch %d, ERR: %f, ACC:%f '%(i, errs/provider._n_samples, accs/provider._n_samples))

        writer.add_summary(merge, i)
        if flag == False:
            sample_inputs = batch_inputs[0].reshape([1, 32, 32, 3])
            sample_targets = batch_targets[0].reshape([1, 10])
            flag = True
        values = sess.run([layers['conv1']['hidden'],layers['conv2']['hidden'],layers['pooling1']['hidden'],layers['pooling2']['hidden']], feed_dict = {inputs_placeholder:sample_inputs, targets_placeholder:sample_targets})
        # writer.add_summary(merge[0], i)
        # print(values[0].shape)
        print('Saving images ... ')
        plt.imsave('outputs/conv1/conv1_hidden_'+str(i)+'.png', build_image(values[0][0], 4))
        plt.imsave('outputs/conv2/conv2_hidden_'+str(i)+'.png', build_image(values[1][0], 8))
        plt.imsave('outputs/pooling1/pooling1_hidden_'+str(i)+'.png', build_image(values[2][0], 4))
        plt.imsave('outputs/pooling2/pooling2_hidden_'+str(i)+'.png', build_image(values[3][0], 8))
        
#     outputs = sess.run(layers['affine3']['hidden'], feed_dict = {inputs_placeholder:batch_inputs, targets_placeholder:batch_targets})
#
# f1 = plt.figure()
# ax1 = f1.add_subplot(111)
# ax1.imshow(outputs)
# f2 = plt.figure()
# ax2 = f2.add_subplot(111)
# ax2.imshow(batch_targets)
# plt.show()
