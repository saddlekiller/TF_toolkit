import tensorflow as tf
from cmd_io import *
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
        raise Exception(cmd_print(2, 'PLACEHOLDER MAPPING FAILED', False))

def layers_mapping(name, config):

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

    if config['layer_type'] == 'affine_layer':
        shape = [config['input_dim'], config['output_dim']]
    return affine_layer(scope_name = name, shape = shape, activation = activation)

def optimizer_mapping(config):

    if config['opt_type'] == 'adam':
        return tf.train.AdamOptimizer(config['lr'])
    elif config['opt_type'] == 'sgd':
        return tf.train.GradientDescentOptimizer(config['lr'])

def loss_mapping(config):
    if config['loss_type'] == 'cross_entropy':
        return tf.nn.sigmoid_cross_entropy_with_logits
