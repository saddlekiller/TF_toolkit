import tensorflow as tf
import numpy as np



a = tf.constant(np.ones([2, 3, 4]).astype(np.float32))
b = tf.constant(np.arange(6).reshape([2, 3]).astype(np.float32))
# b1 = tf.tile(b, [1, 3, 1])
# c = tf.reduce_sum(tf.multiply(a, b1), 2)
# d = tf.nn.softmax(c)
sess = tf.Session()
# print(sess.run(a))
# print(sess.run(b))
# print(sess.run(b1))
# print(sess.run(c))
# print(sess.run(c).shape)
# print(sess.run(tf.reduce_sum(c, 2)))
# print(sess.run(c).shape)


alpha = tf.constant(np.log(np.array(
    [
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4.]
        ],
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4.]
        ]
    ]).astype(np.float32)

))
alpha = tf.transpose(tf.nn.softmax(tf.transpose(alpha, [0, 2, 1])), [0, 2, 1])
data = tf.constant(np.ones((2, 4, 3)).astype(np.float32))
temp = tf.multiply(alpha, data)
# print(data.shape)
print(alpha)
print(data)
print(sess.run(tf.reduce_sum(temp, 1)))
