def dtype_assertion(dtype):
    assert(dtype in ['float32', 'int32', 'bool'])

def activation_assertion(activation):
    assert(activation in ['relu', 'sigmoid', 'tanh', 'softplus', ''])

def optimizer_assertion(optimizer):
    assert(optimizer in ['adam', 'sgd', 'adamgrad'])

def run_mode_assertion(run_mode):
    assert(run_mode in ['train', 'valid', 'test', 'continue_train'])

def layer_assertion(config):
    assert('layer_type' in list(config.keys()))

def affine_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'input_dim', 'output_dim', 'activation']))

def convolution_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'input_dim', 'output_dim', 'kernel_size1', 'kernel_size2', 'padding', 'strides', 'activation']))
    assert(config['padding'] in ['VALID', 'SAME'])

def deconvolution_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'output_shape', 'layer_type', 'input_dim', 'output_dim', 'kernel_size1', 'kernel_size2', 'padding', 'strides', 'activation']))
    assert(config['padding'] in ['VALID', 'SAME'])

def maxpooling_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'input_dim', 'output_dim', 'kernel_size1', 'kernel_size2', 'padding', 'strides', 'ksize']))

def reshape_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'shape']))

def scope_assertion(scopes):
    assert(len(scopes) == len(list(set(scopes))))
