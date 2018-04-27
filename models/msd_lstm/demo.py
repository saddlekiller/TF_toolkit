import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def basic_lstm(features, labels, mode):
	n_lstm_hidden = 25
	n_affine_hidden = 200
	n_classes = 10
	rnn_cell = tf.nn.rnn_cell.LSTMCell(n_lstm_hidden)
	dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	affine1 = tf.layers.dense(dynamic_rnn_outputs[:,-1,:], units = n_affine_hidden, activation = tf.nn.relu)
	affine2 = tf.layers.dense(affine1, units = n_affine_hidden, activation = tf.nn.relu)
	affine3 = tf.layers.dense(affine2, units = n_affine_hidden, activation = tf.nn.relu)
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
	# rnn_cell = tf.nn.rnn_cell.LSTMCell(n_lstm_hidden)
	# dynamic_rnn_outputs, dynamic_rnn_states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = features['x'], dtype = tf.float32)
	# affine1 = tf.layers.dense(dynamic_rnn_outputs[:,-1,:], units = n_affine_hidden, activation = tf.nn.relu)
	# affine2 = tf.layers.dense(affine1, units = n_affine_hidden, activation = tf.nn.relu)
	# affine3 = tf.layers.dense(affine2, units = n_affine_hidden, activation = tf.nn.relu)
	# logits = tf.layers.dense(affine3, units = n_classes, activation = tf.identity)

	reshape1 = tf.reshape(features['x'], [-1, 120 * 25])
	with tf.variable_scope('affine1'):
		affine_w1 = tf.get_variable(name = 'w', shape = [3000, 1024], initializer = tf.truncated_normal_initializer)
		affine_b1 = tf.get_variable(name = 'b', shape = [1024], initializer = tf.truncated_normal_initializer)
		affine1 = tf.nn.relu(tf.add(tf.matmul(reshape1, affine_w1), affine_b1))

	with tf.variable_scope('affine2'):
		affine_w2 = tf.get_variable(name = 'w', shape = [1024, 256], initializer = tf.truncated_normal_initializer)
		affine_b2 = tf.get_variable(name = 'b', shape = [256], initializer = tf.truncated_normal_initializer)
		affine2 = tf.nn.relu(tf.add(tf.matmul(affine1, affine_w2), affine_b2))

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


def main(unused_argv):
	train_= np.load('../../data/MSD/msd-10-genre-train.npz')
	valid_= np.load('../../data/MSD/msd-10-genre-valid.npz')

	train_inputs = np.array(train_['inputs'], dtype = np.float32)
	train_targets = np.array(train_['targets'], dtype = np.int32)
	valid_inputs = np.array(valid_['inputs'], dtype = np.float32)
	valid_targets = np.array(valid_['targets'], dtype = np.int32)


	# msd_classifier = tf.estimator.Estimator(model_fn = basic_lstm, model_dir = './models/msd_model')
	msd_classifier = tf.estimator.Estimator(model_fn = normalized_affine, model_dir = './models/msd_model')

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

	for i in range(1000):
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
