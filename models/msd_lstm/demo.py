import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base as base_layer


tf.logging.set_verbosity(tf.logging.INFO)
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def basic_lstm(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10
	rnn_cell = tf.contrib.rnn.LSTMCell(n_lstm_hidden)
	dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	affine1 = tf.layers.dense(dynamic_rnn_outputs[:,-1,:], units = n_affine_hidden, activation = tf.nn.relu)
	affine2 = tf.layers.dense(affine1, units = n_affine_hidden, activation = tf.nn.sigmoid)
	affine3 = tf.layers.dense(affine2, units = n_affine_hidden, activation = tf.nn.sigmoid)
	logits = tf.layers.dense(affine3, units = n_classes, activation = tf.identity)

	predictions = {
		'classes': tf.argmax(input = logits, axis = 1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels = labels, predictions = predictions['classes'], name = 'accuracy'
		)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	# acc = tf.metrics.accuracy(
	# 	labels = labels, predictions = predictions['classes'], name = 'accuracy'
	# )

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)


def normalized_affine(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10

	reshape1 = tf.reshape(features['x'], [-1, 120 * 25])
	with tf.variable_scope('affine1'):
		affine_w1_ = tf.get_variable(name = 'w', shape = [3000, 1024], initializer = tf.truncated_normal_initializer)
		affine_w1 = affine_w1_ / tf.reduce_sum(affine_w1_, 0)
		affine_b1_ = tf.get_variable(name = 'b', shape = [1024], initializer = tf.truncated_normal_initializer)
		affine_b1 = affine_b1_ / tf.reduce_sum(affine_b1_, 0)
		affine1 = tf.nn.sigmoid(tf.add(tf.matmul(reshape1, affine_w1), affine_b1))

	with tf.variable_scope('affine2'):
		affine_w2_ = tf.get_variable(name = 'w', shape = [1024, 256], initializer = tf.truncated_normal_initializer)
		affine_w2 = affine_w2_ / tf.reduce_sum(affine_w2_, 0)
		affine_b2_ = tf.get_variable(name = 'b', shape = [256], initializer = tf.truncated_normal_initializer)
		affine_b2 = affine_b2_ / tf.reduce_sum(affine_b2_, 0)
		affine2 = tf.nn.sigmoid(tf.add(tf.matmul(affine1, affine_w2), affine_b2))

	with tf.variable_scope('affine3'):
		affine_w3_ = tf.get_variable(name = 'w', shape = [256, 10], initializer = tf.truncated_normal_initializer)
		# affine_w3 = tf.nn.softmax(affine_w3_, 0)
		affine_w3 = affine_w3_ / tf.reduce_sum(affine_w3_, 0)
		affine_b3_ = tf.get_variable(name = 'b', shape = [10], initializer = tf.truncated_normal_initializer)
		# affine_b3 = tf.nn.softmax(affine_b3_)
		affine_b3 = affine_b3_ / tf.reduce_sum(affine_b3_, 0)


		logits = tf.identity(tf.add(tf.matmul(affine2, affine_w3), affine_b3))

	predictions = {
		'classes': tf.argmax(input = logits, axis = 1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels = labels, predictions = predictions['classes'], name = 'accuracy'
		)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	# acc = tf.metrics.accuracy(
	# 	labels = labels, predictions = predictions['classes'], name = 'accuracy'
	# )

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)

def partial_normalized_affine(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10

	reshape1 = tf.reshape(features['x'], [-1, 120 * 25])
	with tf.variable_scope('affine1'):
		affine_w1 = tf.get_variable(name = 'w', shape = [3000, 1024], initializer = tf.truncated_normal_initializer)
		affine_b1 = tf.get_variable(name = 'b', shape = [1024], initializer = tf.truncated_normal_initializer)
		affine1 = tf.nn.sigmoid(tf.add(tf.matmul(reshape1, affine_w1), affine_b1))

	with tf.variable_scope('affine2'):
		affine_w2 = tf.get_variable(name = 'w', shape = [1024, 256], initializer = tf.truncated_normal_initializer)
		affine_b2 = tf.get_variable(name = 'b', shape = [256], initializer = tf.truncated_normal_initializer)
		affine2 = tf.nn.sigmoid(tf.add(tf.matmul(affine1, affine_w2), affine_b2))

	with tf.variable_scope('affine3'):
		affine_w3_ = tf.get_variable(name = 'w', shape = [256, 10], initializer = tf.truncated_normal_initializer)
		affine_w3 = affine_w3_ / tf.reduce_sum(affine_w3_, 0)
		affine_b3_ = tf.get_variable(name = 'b', shape = [10], initializer = tf.truncated_normal_initializer)
		affine_b3 = affine_b3_ / tf.reduce_sum(affine_b3_, 0)
		logits = tf.identity(tf.add(tf.matmul(affine2, affine_w3), affine_b3))

	predictions = {
		'classes': tf.argmax(input = logits, axis = 1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels = labels, predictions = predictions['classes'], name = 'accuracy'
		)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	# acc = tf.metrics.accuracy(
	# 	labels = labels, predictions = predictions['classes'], name = 'accuracy'
	# )

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)

def pure_affine(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10

	reshape1 = tf.reshape(features['x'], [-1, 120 * 25])
	with tf.variable_scope('affine1'):
		affine_w1 = tf.get_variable(name = 'w', shape = [3000, 1024], initializer = tf.truncated_normal_initializer)
		affine_b1 = tf.get_variable(name = 'b', shape = [1024], initializer = tf.truncated_normal_initializer)
		affine1 = tf.nn.sigmoid(tf.add(tf.matmul(reshape1, affine_w1), affine_b1))

	with tf.variable_scope('affine2'):
		affine_w2 = tf.get_variable(name = 'w', shape = [1024, 256], initializer = tf.truncated_normal_initializer)
		affine_b2 = tf.get_variable(name = 'b', shape = [256], initializer = tf.truncated_normal_initializer)
		affine2 = tf.nn.sigmoid(tf.add(tf.matmul(affine1, affine_w2), affine_b2))

	with tf.variable_scope('affine3'):
		affine_w3 = tf.get_variable(name = 'w', shape = [256, 10], initializer = tf.truncated_normal_initializer)
		affine_b3 = tf.get_variable(name = 'b', shape = [10], initializer = tf.truncated_normal_initializer)
		logits = tf.identity(tf.add(tf.matmul(affine2, affine_w3), affine_b3))

	predictions = {
		'classes': tf.argmax(input = logits, axis = 1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels = labels, predictions = predictions['classes'], name = 'accuracy'
		)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	# acc = tf.metrics.accuracy(
	# 	labels = labels, predictions = predictions['classes'], name = 'accuracy'
	# )

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)

class NormalizedRNNCell(tf.nn.rnn_cell.BasicRNNCell):

	def __init__(self, num_units, activation=None, reuse=None, name=None, dtype=None):
		super(NormalizedRNNCell, self).__init__(num_units = num_units)

		# Inputs must be 2-dimensional.
		self.input_spec = base_layer.InputSpec(ndim=2)

		self._num_units = num_units
		self._activation = activation or tf.nn.tanh

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def build(self, inputs_shape):
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
											 % inputs_shape)
		input_depth = inputs_shape[1].value
		self.input_depth = input_depth
		self._kernel = self.add_variable(
				_WEIGHTS_VARIABLE_NAME,
				shape=[input_depth + self._num_units, self._num_units])
		self._bias = self.add_variable(
				_BIAS_VARIABLE_NAME,
				shape=[self._num_units])
				# initializer=init_ops.zeros_initializer(dtype=self.dtype))

		self.built = True

	def call(self, inputs, state):
		"""Most basic RNN: output = new_state = act(W * input + U * state + B)."""

		self._kernel_1 = tf.slice(self._kernel, [0, 0], [self.input_depth, self._num_units])
		self._kernel_2 = tf.slice(self._kernel, [self.input_depth, 0], [self._num_units, self._num_units])
		self._kernel_1 = self._kernel_1 / tf.reduce_sum(self._kernel_1, 0)
		self._kernel_2 = self._kernel_2 / tf.reduce_sum(self._kernel_2, 0)
		self._kernel = tf.concat([self._kernel_1, self._kernel_2], 0)
		self._bias = self._bias / tf.reduce_sum(self._bias)

		gate_inputs = tf.matmul(
				tf.concat([inputs, state], 1), self._kernel)
		gate_inputs = tf.add(gate_inputs, self._bias)
		output = self._activation(gate_inputs)
		return output, output

def normalized_rnn(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10
	rnn_cell = NormalizedRNNCell(n_lstm_hidden)
	dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	affine1 = tf.layers.dense(dynamic_rnn_outputs[:,-1,:], units = n_affine_hidden, activation = tf.nn.relu)
	affine2 = tf.layers.dense(affine1, units = n_affine_hidden, activation = tf.nn.sigmoid)
	affine3 = tf.layers.dense(affine2, units = n_affine_hidden, activation = tf.nn.sigmoid)
	logits = tf.layers.dense(affine3, units = n_classes, activation = tf.identity)

	predictions = {
		'classes': tf.argmax(input = logits, axis = 1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels = labels, predictions = predictions['classes'], name = 'accuracy'
		)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	# acc = tf.metrics.accuracy(
	# 	labels = labels, predictions = predictions['classes'], name = 'accuracy'
	# )

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)

def basic_rnn(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10
	rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_lstm_hidden)
	dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	affine1 = tf.layers.dense(dynamic_rnn_outputs[:,-1,:], units = n_affine_hidden, activation = tf.nn.relu)
	affine2 = tf.layers.dense(affine1, units = n_affine_hidden, activation = tf.nn.sigmoid)
	affine3 = tf.layers.dense(affine2, units = n_affine_hidden, activation = tf.nn.sigmoid)
	logits = tf.layers.dense(affine3, units = n_classes, activation = tf.identity)

	predictions = {
		'classes': tf.argmax(input = logits, axis = 1),
		'probs': tf.nn.softmax(logits, name='softmax_tensor')
	}
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels = labels, predictions = predictions['classes'], name = 'accuracy'
		)
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
	# acc = tf.metrics.accuracy(
	# 	labels = labels, predictions = predictions['classes'], name = 'accuracy'
	# )

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)

model_mapping = {
	'basic_lstm': basic_lstm,
	'pure_affine': pure_affine,
	'normalized_affine': normalized_affine,
	'partial_normalized_affine': partial_normalized_affine,
	'normalized_rnn': normalized_rnn,
	'basic_rnn': basic_rnn
}

def main(argv):
	# print(argv)

	train_= np.load('../../data/MSD/msd-10-genre-train.npz')
	valid_= np.load('../../data/MSD/msd-10-genre-valid.npz')

	train_inputs = np.array(train_['inputs'], dtype = np.float32)
	train_targets = np.array(train_['targets'], dtype = np.int32)
	valid_inputs = np.array(valid_['inputs'], dtype = np.float32)
	valid_targets = np.array(valid_['targets'], dtype = np.int32)


	# msd_classifier = tf.estimator.Estimator(model_fn = basic_lstm, model_dir = './models/msd_model')
	# msd_classifier = tf.estimator.Estimator(model_fn = normalized_affine, model_dir = './models/msd_model_all_prob')

	msd_classifier = tf.estimator.Estimator(model_fn = model_mapping[argv[1]], model_dir = './models/' + argv[2])

	tensors_to_log = {"probabilities": "softmax_tensor"}
	# logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 1)

	def get_input_fn(inputs, targets, mode, batch_size = 50):
		if mode == tf.estimator.ModeKeys.TRAIN:
			return tf.estimator.inputs.numpy_input_fn(
				x = {'x': inputs},
				y = targets,
				batch_size = batch_size,
				num_epochs = None,
				shuffle = True)
		elif mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.inputs.numpy_input_fn(
				x = {'x': inputs},
				y = targets,
				batch_size = batch_size,
				num_epochs = 1,
				shuffle = False)
		return None

	for i in range(200):
		msd_classifier.train(
			input_fn = get_input_fn(train_inputs, train_targets, tf.estimator.ModeKeys.TRAIN),
			steps = 5
		)
		if (i + 1)%1 == 0:
			msd_classifier.evaluate(
			input_fn = get_input_fn(valid_inputs, valid_targets, tf.estimator.ModeKeys.EVAL)
			)

if __name__ == '__main__':
	tf.app.run()
	# temp = tf.constant(np.random.random((4, 2)))
	# temp1 = tf.nn.softmax(temp, 0)
	# # print(temp1)
	# sess = tf.Session()
	# print(sess.run(temp))
	# print(sess.run(temp1))
