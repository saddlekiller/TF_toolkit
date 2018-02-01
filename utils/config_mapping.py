import tensorflow as tf
# from cmd_io import *
from logging_io import *
from assertions import *
from layers import *

def placeholder_mapping(config):
    try:
        name = config['name']
        shape = config['shape']
        if config['shape'][0] == -1:
            shape[0] = None
        dtype = config['dtype']
        dtype_assertion(dtype)
        if dtype == 'float32':
            dtype = tf.float32
        elif dtype == 'int32':
            dtype = tf.int32
        elif dtype == 'bool':
            dtype = tf.bool
        return tf.placeholder(name = name, shape = shape, dtype = dtype)
    except:
        raise Exception(logging_io.ERROR_INFO('PLACEHOLDER MAPPING FAILED'))
        # raise Exception(cmd_print(2, 'PLACEHOLDER MAPPING FAILED', False))

def layers_mapping(name, config):

    if 'activation' in config.keys():
        activation = config['activation']
        activation_assertion(activation)
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'tanh':
            activation = tf.nn.tanh
        elif activation == 'softplus':
            activation = tf.nn.softplus
        elif activation == '':
            activation = tf.identity
    layer_assertion(config)
    if config['layer_type'] == 'affine_layer':
        affine_assertion(config)
        shape = [config['input_dim'], config['output_dim']]
        return affine_layer(scope_name = name, shape = shape, activation = activation)
    elif config['layer_type'] == 'convolution_layer':
        convolution_assertion(config)
        shape = [config['kernel_size1'], config['kernel_size2'], config['input_dim'], config['output_dim']]
        padding = config['padding']
        strides = config['strides']
        return convolution_layer(scope_name = name, shape = shape, activation = activation, padding = padding, strides = strides)
    elif config['layer_type'] == 'maxpooling_layer':
        maxpooling_assertion(config)
        ksize = config['ksize']
        padding = config['padding']
        strides = config['strides']
        return maxpooling_layer(scope_name = name, padding = padding, strides = strides, ksize = ksize)
    elif config['layer_type'] == 'reshape_layer':
        reshape_assertion(config)
        shape = config['shape']
        return reshape_layer(scope_name = name, shape = shape)

def optimizer_mapping(config):

    if config['opt_type'] == 'adam':
        return tf.train.AdamOptimizer(config['lr'])
    elif config['opt_type'] == 'sgd':
        return tf.train.GradientDescentOptimizer(config['lr'])

def loss_mapping(config):
    if config['loss_type'] == 'cross_entropy':
        return tf.nn.sigmoid_cross_entropy_with_logits
