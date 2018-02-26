import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../utils')
from data_provider import *
import pickle


provider = MNISTProvider('../../data/mnist-train.npz', 50)
# for batch_inputs, batch_targets in provider:
#     print(batch_inputs.shape, batch_targets.shape)
#     break
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

    for i in range(100):
        errs = []
        for batch_inputs, batch_targets in provider:
            feed_dict = {inputs_placeholder: batch_inputs, code_placeholder:np.zeros((1, hidden_dim3))}
            _, err = sess.run([optimizer, loss], feed_dict = feed_dict)
            errs.append(err)
        print('#Epoch %d, Loss: %f'%(i, np.mean(errs)))
    save_path = saver.save(sess, 'models/model.ckpt')

    saver.restore(sess, 'models/model.ckpt')
    hidden_units = {}
    for batch_inputs, batch_targets in provider:
        feed_dict = {inputs_placeholder: batch_inputs, code_placeholder:np.zeros((1, hidden_dim3))}
        features = sess.run(e_o3, feed_dict = feed_dict)
        for target, feature in zip(batch_targets, features):
            key = np.argmax(target)
            if key not in hidden_units.keys():
                hidden_units[key] = []
            hidden_units[key].append(feature)
    means = [None]*10
    for key in hidden_units.keys():
        hidden_units[key] = np.array(hidden_units[key])
        means[key] = np.mean(hidden_units[key], 0)
        print(np.mean(hidden_units[key], 0).shape)
    means = np.array(means)
    print(means.shape)
    feed_dict = {inputs_placeholder: batch_inputs, code_placeholder:means}
    image = sess.run(rebuild, feed_dict = feed_dict)
    print(image.shape)
    import matplotlib.pyplot as plt
    f = plt.figure()
    for i in range(10):
        ax = f.add_subplot(3, 4, i+1)
        ax.imshow(image[i])
        plt.title(i)
    plt.show()
    # pickle.dump(hidden_units, open('features.npz', 'wb'))
