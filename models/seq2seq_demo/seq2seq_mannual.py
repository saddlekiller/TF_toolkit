import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../utils')
from data_provider import *
from logging_io import *
from tensorflow.contrib.layers import *

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


class AttentionLSTMCell(tf.nn.rnn_cell.LSTMCell):


  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None, name=None):
    super(AttentionLSTMCell, self).__init__(_reuse=reuse, name=name)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

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
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state

batch_size = 100
max_word = 35

logging_io.DEBUG_INFO('Generating and saving TRAINING corpus')
train_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TRAIN.txt', batch_size = 50, max_word = 35, isGenerate = True, isTrain = True)
logging_io.DEBUG_INFO('Generating and saving VALIDATION corpus')
valid_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TEST.txt', batch_size = 50, max_word = 35, isGenerate = True, isTrain = False)

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

    with tf.variable_scope('encoder_decoder', reuse = True):
        decoder_cell = AttentionLSTMCell(num_units=n_lstm_hidden, state_is_tuple=True)
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
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
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # for i in range(200):
    #     errs = []
    #     accs = []
    #     for batch in train_provider:
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
    #     logging_io.RESULT_INFO('TRAINING Epoch %5d => LOSS: %8f, ACC:%8f'%(i, np.mean(errs), np.mean(accs)))
    #     if i % 5 == 0:
    #         errs = []
    #         accs = []
    #         for batch in valid_provider:
    #             feed_dict = {
    #                     sentence_placeholder:batch[0],
    #                     intent_placeholder:batch[1],
    #                     mention_placeholder:batch[2],
    #                     intent_len_placeholder:batch[3],
    #                     mention_len_placeholder:batch[4]
    #             }
    #             _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = feed_dict)
    #             errs.append(err)
    #             accs.append(acc)
    #         logging_io.RESULT_INFO('____|VALIDATION Epoch %5d => LOSS: %8f, ACC:%8f'%(i, np.mean(errs), np.mean(accs)))
