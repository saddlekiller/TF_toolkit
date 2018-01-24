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
        self.activation = activation
        with tf.variable_scope(self.scope_name):
            self.param = dict()
            self.param['w'] = truncated_normal('weight', shape).get()
            self.param['b'] = truncated_normal('bias', shape[-1]).get()

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            self.outputs = self.activation(tf.add(tf.matmul(inputs, self.param['w']), self.param['b']))
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(affine_layer).__str__())
