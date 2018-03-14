import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../utils')
from data_provider import *
from logging_io import *
from tensorflow.contrib.layers import *
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import tf_logging as logging

def embedding_layer(inputs, n_hidden, activation = tf.identity):

    with tf.variable_scope('embedding'):

        shapes = tf.shape(inputs)
        n_batch = shapes[0]
        n_step = shapes[1]
        n_feature = shapes[2]
        inputs_reshape = tf.reshape(inputs, [n_batch * n_step, n_feature])
        weight = tf.get_variable('weight', initializer = tf.truncated_normal([n_feature, n_hidden]))
        # weight = tf.ones([n_feature, n_hidden])

        outputs = activation(tf.matmul(inputs_reshape, weight))
        return tf.reshape(outputs, [n_batch, n_step, n_hidden])


class AttentionLSTMCell(rnn_cell_impl.RNNCell):

    def __init__(self, num_units, reuse=None):

        super(AttentionLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._state_is_tuple = True
        self._reuse = reuse


    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            'kernel',
            shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(
            'bias',
            shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        # print(state)
        # print('|'*50)
        c, h = state
        gate_inputs = tf.add(tf.matmul(tf.concat([inputs, h], 1), self._kernel), self._bias)
        i, j, f, o = tf.split(value = gate_inputs, num_or_size_splits = 4, axis = 1)
        new_c = tf.add(tf.multiply(c, tf.nn.sigmoid(f)), tf.multiply(i, tf.nn.tanh(j)))
        new_h = tf.multiply(tf.nn.sigmoid(o), tf.nn.tanh(new_c))
        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
        return new_h, new_state

# inputs = tf.placeholder(tf.float32, [10, 5, 10])
# cell = AttentionLSTMCell(10)
# outputs, states = tf.nn.dynamic_rnn(
#                 cell=cell,
#                 dtype=tf.float32,
#                 inputs=inputs)

batch_size = 100
max_word = 35

# logging_io.DEBUG_INFO('Generating and saving TRAINING corpus')
# train_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TRAIN.txt', batch_size = 50, max_word = 35, isGenerate = True, isTrain = True)
# logging_io.DEBUG_INFO('Generating and saving VALIDATION corpus')
# valid_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TEST.txt', batch_size = 50, max_word = 35, isGenerate = True, isTrain = False)

logging_io.DEBUG_INFO('Loading TRAINING corpus')
train_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TRAIN.txt', batch_size = batch_size, max_word = max_word, isGenerate = False, isTrain = True)
logging_io.DEBUG_INFO('Loading VALIDATION corpus')
valid_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TEST.txt', batch_size = batch_size, max_word = max_word, isGenerate = False, isTrain = False)

n_word = train_provider.n_word
n_intent = train_provider.n_intent
n_mention = train_provider.n_mention
max_seq_len = train_provider.max_word
max_intent_len = train_provider.max_intent_len

isInference = False
n_lstm_hidden = 128

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    sentence_placeholder = tf.placeholder(tf.float32, [None, max_seq_len, n_word])
    intent_placeholder = tf.placeholder(tf.float32, [None, max_intent_len, n_intent])
    mention_placeholder = tf.placeholder(tf.float32, [None, max_seq_len, n_mention])
    intent_len_placeholder = tf.placeholder(tf.int32, [None])
    mention_len_placeholder = tf.placeholder(tf.int32, [None])

    sentence_shape = tf.shape(sentence_placeholder)
    sentence_reshape = tf.reshape(sentence_placeholder, [sentence_shape[0] * max_seq_len, n_word])

    with tf.variable_scope('encoder_affine', reuse = isInference):
        encoder_affine  = fully_connected(sentence_placeholder,n_lstm_hidden,weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=tf.nn.relu)

    with tf.variable_scope('encoder_reshape', reuse = isInference):
        encoder_reshape = tf.reshape(encoder_affine, [sentence_shape[0], max_seq_len, n_lstm_hidden])

    with tf.variable_scope('encoder_decoder', reuse = isInference):
        encoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_lstm_hidden, state_is_tuple=True)
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                dtype=tf.float32,
                sequence_length=mention_len_placeholder,
                inputs=encoder_reshape)
        print(encoder_states)
        print('#'*50)

    with tf.variable_scope('encoder_decoder', reuse = False):
        decoder_cell = AttentionLSTMCell(num_units=n_lstm_hidden)
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
                initial_state = encoder_states,
                cell=decoder_cell,
                dtype=tf.float32,
                sequence_length=mention_len_placeholder,
                inputs=encoder_outputs)
    # logging_io.WARNING_INFO(mention_len_placeholder)
    with tf.variable_scope('decoder_mention_reshape'):
        decoder_mention_reshape = tf.reshape(decoder_outputs, [sentence_shape[0] * max_seq_len, n_lstm_hidden])

    with tf.variable_scope('decoder_mention_affine'):
        decoder_mention_affine = fully_connected(decoder_mention_reshape,n_mention,weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=tf.nn.relu)

    mention_placeholder_reshape = tf.reshape(mention_placeholder, [-1, n_mention])

    # logging_io.DEBUG_INFO(decoder_mention_affine)
    # # 3500*60
    # logging_io.DEBUG_INFO(mention_placeholder)
    # # 100*35*60
    # logging_io.DEBUG_INFO(mention_placeholder_reshape)
    # # 3500*60
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = decoder_mention_affine, labels = mention_placeholder_reshape))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(decoder_mention_affine, 1), tf.argmax(mention_placeholder_reshape, 1)), tf.float32))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        errs = []
        accs = []
        for batch in train_provider:
            feed_dict = {
                    sentence_placeholder:batch[0],
                    intent_placeholder:batch[1],
                    mention_placeholder:batch[2],
                    intent_len_placeholder:batch[3],
                    mention_len_placeholder:batch[4]
            }
            _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = feed_dict)
            errs.append(err)
            accs.append(acc)
            logging_io.RESULT_INFO('TRAINING Epoch %5d => LOSS: %8f, ACC:%8f'%(i, np.mean(errs), np.mean(accs)))
        # if i % 5 == 0:
        #     errs = []
        #     accs = []
        #     for batch in valid_provider:
        #         feed_dict = {
        #                 sentence_placeholder:batch[0],
        #                 intent_placeholder:batch[1],
        #                 mention_placeholder:batch[2],
        #                 intent_len_placeholder:batch[3],
        #                 mention_len_placeholder:batch[4]
        #         }
        #         _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = feed_dict)
        #         errs.append(err)
        #         accs.append(acc)
        #     logging_io.RESULT_INFO('____|VALIDATION Epoch %5d => LOSS: %8f, ACC:%8f'%(i, np.mean(errs), np.mean(accs)))
