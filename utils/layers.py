import tensorflow as tf
from params import *

class layer(object):

    def __init__(self):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError

    def outputs(self):
        raise NotImplementedError

    def add_summary(self):
        raise NotImplementedError

    def __str__():
        return self.__class__.__name__

class affine_layer(layer):

    def __init__(self, scope_name, shape, activation):
        self.scope_name = scope_name
        self.shape = shape
        self.activation = activation
        with tf.variable_scope(self.scope_name):
            self.param = dict()
            self.param['w'] = truncated_normal('weight', self.shape).get()
            self.param['b'] = truncated_normal('bias', self.shape[-1]).get()

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            self.outputs = self.activation(tf.add(tf.matmul(inputs, self.param['w']), self.param['b']))
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(affine_layer).__str__())

class convolution_layer(layer):

    def __init__(self, scope_name, shape, activation, padding, strides):
        self.scope_name = scope_name
        self.shape = shape
        self.activation = activation
        self.padding = padding
        self.strides = strides
        with tf.variable_scope(self.scope_name):
            self.param = dict()
            self.param['w'] = truncated_normal('weight', self.shape).get()
            self.param['b'] = truncated_normal('bias', self.shape[-1]).get()

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            self.outputs = self.activation(tf.add(tf.nn.conv2d(input = inputs, filter = self.param['w'], padding = self.padding, strides = self.strides), self.param['b']))
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(convolution_layer).__str__())

class reshape_layer(layer):

    def __init__(self, scope_name, shape):
        self.scope_name = scope_name
        self.shape = shape

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            self.outputs = tf.reshape(inputs, self.shape)
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(reshape_layer).__str__())


class maxpooling_layer(layer):

    def __init__(self, scope_name, padding, strides, ksize):
        self.scope_name = scope_name
        self.padding = padding
        self.strides = strides
        self.ksize = ksize

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            self.outputs = tf.nn.max_pool(inputs, padding = self.padding, strides = self.strides, ksize = self.ksize)
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(maxpooling_layer).__str__())
