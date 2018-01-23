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
    
def affine_layer(layer):
    
    def __init__(self, scope_name, shape):
        with tf.variable_scope(scope_name):
            self.param = dict()
            self.param['w'] = truncated_normal('weight', shape)
            self.param['b'] = truncated_normal('bias', shape[-1])
            
    def outputs(self, inputs, activation):
        self.outputs = activation(tf.add(tf.matmul(inputs, self.param['w']), self.param['b']))
        return self.outputs
        
        
    