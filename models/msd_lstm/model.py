import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base as base_layer
from cell import *

tf.logging.set_verbosity(tf.logging.INFO)
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def BasicRNNModel(features, labels, mode, params):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10
	rnn_cell = params['celltype'](n_lstm_hidden)
	dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	affine1 = tf.layers.dense(dynamic_rnn_outputs[:,-1,:], units = n_affine_hidden, activation = params['activation'])
	affine2 = tf.layers.dense(affine1, units = n_affine_hidden, activation = params['activation'])
	affine3 = tf.layers.dense(affine2, units = n_affine_hidden, activation = params['activation'])
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

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)

def NormRNNModel(features, labels, mode, params):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10
	rnn_cell = params['celltype'](n_lstm_hidden)
	dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	with tf.variable_scope('affine1'):
		affine_w1_ = tf.get_variable(name = 'w', shape = [n_lstm_hidden, n_affine_hidden], initializer = tf.truncated_normal_initializer)
		affine_b1_ = tf.get_variable(name = 'b', shape = [n_affine_hidden], initializer = tf.truncated_normal_initializer)
		affine1 = tf.nn.sigmoid(tf.add(tf.matmul(dynamic_rnn_outputs[:, -1, :], affine_w1), affine_b1))

	with tf.variable_scope('affine2'):
		affine_w2_ = tf.get_variable(name = 'w', shape = [n_affine_hidden, n_affine_hidden], initializer = tf.truncated_normal_initializer)
		affine_b2_ = tf.get_variable(name = 'b', shape = [n_affine_hidden], initializer = tf.truncated_normal_initializer)
		affine2 = tf.nn.sigmoid(tf.add(tf.matmul(affine1, affine_w2), affine_b2))

	with tf.variable_scope('affine3'):
		affine_w3_ = tf.get_variable(name = 'w', shape = [n_affine_hidden, n_classes], initializer = tf.truncated_normal_initializer)
		affine_w3 = affine_w3_ / tf.reduce_sum(affine_w3_, 0)
		affine_b3_ = tf.get_variable(name = 'b', shape = [n_classes], initializer = tf.truncated_normal_initializer)
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

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions, train_op = train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops, predictions = predictions)


def main(argv):
	train_= np.load('../../data/MSD/msd-10-genre-train.npz')
	valid_= np.load('../../data/MSD/msd-10-genre-valid.npz')

	train_inputs = np.array(train_['inputs'], dtype = np.float32)
	train_targets = np.array(train_['targets'], dtype = np.int32)
	valid_inputs = np.array(valid_['inputs'], dtype = np.float32)
	valid_targets = np.array(valid_['targets'], dtype = np.int32)

	_CELL_MAPPING = {
	   'BasicRNNCell': tf.nn.rnn_cell.BasicRNNCell,
	   'NormalizedRNNCell': NormalizedRNNCell,
	   'LSTMCell': tf.nn.rnn_cell.BasicLSTMCell,
	   'GRUCell': tf.nn.rnn_cell.GRUCell
	}
	_MODEL_MAPPING = {
		'BasicRNNModel': BasicRNNModel,
		'NormRNNModel': NormRNNModel
	}

	modeltype = argv[1]
	celltype = argv[2]

	params={
		'celltype': _CELL_MAPPING[celltype],
		'activation': tf.nn.sigmoid,
		'learning_rate': 0.005}
	msd_classifier = tf.estimator.Estimator(model_fn = _MODEL_MAPPING[modeltype], params = params, model_dir = './models/' + modeltype + '_' + celltype)

	tensors_to_log = {"probabilities": "softmax_tensor"}

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

	for i in range(100):
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
