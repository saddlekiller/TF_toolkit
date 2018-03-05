import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../utils')
from data_provider import *

# provider = PaddedSeqProvider('../../data/anonymous_raw_poi_valid_trimmed.txt', '../data/raw_poiwords.dict', '../data/raw_poilabel_map.npz', 50, 35)
#
# max_seq_len = provider.max_word
# batch_size = provider.batch_size
# voc_size = provider.voc_size
#
# input_placeholder = tf.placeholder(tf.float32, [batch_size, max_seq_len])
data = np.random.random((20, 10))
label = np.random.random((20, 1))
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    input_placeholder = tf.placeholder(tf.float32, [20, 10])
    target_placeholder = tf.placeholder(tf.float32, [20, 1])

    w = tf.get_variable('w', initializer = tf.truncated_normal([10, 1]), dtype = tf.float32)
    b = tf.get_variable('b', initializer = tf.truncated_normal([1]), dtype = tf.float32)

    i = tf.constant(0)
    def ad(i):
        print('hello', i)
        return tf.add(i, 1)
    cx = lambda i: tf.less(i, 10)
    bx = lambda i: ad(i)

    rx = tf.while_loop(cx, bx, [i])

    o = tf.matmul(input_placeholder, w) + b

    loss = tf.reduce_mean((o - target_placeholder)**2)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    sess.run(tf.global_variables_initializer())
    #
    _, loss = sess.run([optimizer, loss], feed_dict = {input_placeholder: data, target_placeholder: label})
    print(loss)
