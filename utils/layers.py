import tensorflow as tf
from params import *
from tensorflow.python.layers import utils

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
            logging_io.WARNING_INFO(self.scope_name)
            logging_io.WARNING_INFO(tf.nn.conv2d(input = inputs, filter = self.param['w'], padding = self.padding, strides = self.strides).shape)
            self.outputs = self.activation(tf.add(tf.nn.conv2d(input = inputs, filter = self.param['w'], padding = self.padding, strides = self.strides), self.param['b']))
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(convolution_layer).__str__())

class deconvolution_layer(layer):

    def __init__(self, scope_name, shape, activation, padding, strides, output_shape):
        self.scope_name = scope_name
        self.shape = shape
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.output_shape = output_shape
        # logging_io.WARNING_INFO(self.shape)
        with tf.variable_scope(self.scope_name):
            self.param = dict()
            self.param['w'] = truncated_normal('weight', self.shape).get()
            self.param['b'] = truncated_normal('bias', self.shape[2]).get()
        # print(self.shape)
        # print('----------------------------------------------')
        # pass

            # height, width = (1,1)
            # kernel_h, kernel_w = (self.shape[0], self.shape[1])
            # stride_h, stride_w = (self.strides[1], self.strides[2])
            # # Infer the dynamic output shape:
            # out_height = utils.deconv_output_length(height,kernel_h,padding,stride_h)
            # out_width = utils.deconv_output_length(width,kernel_w,padding,stride_w)
            # # logging_io.WARNING_INFO(out_height)
            # # logging_io.WARNING_INFO(out_width)
            # self.output_shape = [50, out_height, out_width, self.shape[2]]

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            # print(self.output_shape)
            # print(tf.nn.conv2d_transpose(inputs,self.param['w'], output_shape=self.output_shape, padding = self.padding, strides=self.strides))
            # print(self.shape)
            # logging_io.DEBUG_INFO('******************************************************')
            logging_io.WARNING_INFO(self.scope_name)
            logging_io.WARNING_INFO(self.output_shape)
            self.outputs = self.activation(tf.add(tf.nn.conv2d_transpose(inputs,self.param['w'], output_shape=self.output_shape, padding = self.padding, strides=self.strides), self.param['b']))
            # self.outputs = self.activation(tf.add(tf.nn.conv2d(input = inputs, filter = self.param['w'], padding = self.padding, strides = self.strides), self.param['b']))
        return self.outputs
        # pass

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(deconvolution_layer).__str__())

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
            # logging_io.WARNING_INFO(self.scope_name)
            # logging_io.WARNING_INFO(self.output_shape)
            self.outputs = tf.nn.max_pool(inputs, padding = self.padding, strides = self.strides, ksize = self.ksize)
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(maxpooling_layer).__str__())

class upsampling_layer(layer):

    def __init__(self, scope_name, ksize, output_shape):
        self.scope_name = scope_name
        self.ksize = ksize
        self.kernel = tf.ones(self.ksize)
        self.output_shape = output_shape

    def outputs(self, inputs):
        with tf.variable_scope(self.scope_name):
            logging_io.WARNING_INFO(self.scope_name)
            logging_io.WARNING_INFO(self.output_shape)
            self.outputs = tf.nn.conv2d_transpose(inputs, self.kernel, output_shape=self.output_shape, padding = 'SAME', strides=[1,2,2,1])
            # self.outputs = tf.nn.max_pool(inputs, padding = self.padding, strides = self.strides, ksize = self.ksize)
        return self.outputs

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(upsampling_layer).__str__())

class lstm_layer(layer):

    def __init__(self, name_scope, cell_size, forget_bias = 0.0, reuse = False):
        self.name_scope = name_scope
        self.cell_size = cell_size
        self.forget_bias = forget_bias
        self.reuse = reuse

    def outputs(self, inputs):
        with tf.variable_scope(self.name_scope):
            self.outputs, self.states = tf.nn.dynamic_rnn(cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias = self.forget_bias, reuse = self.reuse
            ), inputs=inputs, dtype=tf.float32)
        return self.outputs, self.states

    def __str__(self):
        return 'scope name: {0:}, class name: {1:}'.format(self.scope_name, super(lstm_layer).__str__())
