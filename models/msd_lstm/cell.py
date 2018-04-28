import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base as base_layer

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
		# self._kernel_1 = self._kernel_1 / tf.reduce_sum(self._kernel_1, 0)
		self._kernel_2 = self._kernel_2 / tf.reduce_sum(self._kernel_2, 0)
		self._kernel = tf.concat([self._kernel_1, self._kernel_2], 0)
		self._bias = self._bias / tf.reduce_sum(self._bias)

		gate_inputs = tf.matmul(
				tf.concat([inputs, state], 1), self._kernel)
		gate_inputs = tf.add(gate_inputs, self._bias)
		output = self._activation(gate_inputs)
		return output, output
