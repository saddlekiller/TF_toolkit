from logging_io import *



def dtype_assertion(dtype):
    assert(dtype in ['float32', 'int32', 'bool'])
    logging_io.SUCCESS_INFO('DTYPE ASSERTION PASS')

def activation_assertion(activation):
    assert(activation in ['relu', 'sigmoid', 'tanh', 'softplus', ''])
    logging_io.SUCCESS_INFO('ACTIVATION ASSERTION PASS')

def optimizer_assertion(optimizer):
    assert(optimizer in ['adam', 'sgd', 'adamgrad', 'momentum', 'rms'])
    logging_io.SUCCESS_INFO('OPTIMIZER ASSERTION PASS')

def optimizer_option_assertion(name, config):
    # logging_io.WARNING_INFO(str(config.keys()))
    if name in ['adam', 'sgd', 'adamgrad']:
        assert(set(config.keys()).issubset(['opt_type', 'lr']))
    elif name == 'momentum':
        assert(set(config.keys()).issubset(['opt_type', 'lr', 'mom']))
    elif name == 'rms':
        assert(set(config.keys()).issubset(['opt_type', 'lr', 'decay']))
    else:
        logging_io.WARNING_INFO('OPTIMIZER NAME NOT FOUND!')
    logging_io.SUCCESS_INFO('OPTIMIZER ASSERTION PASS')

def run_mode_assertion(run_mode):
    assert(run_mode in ['train', 'valid', 'test', 'continue_train'])
    logging_io.SUCCESS_INFO('RUN MODE ASSERTION PASS')

def layer_assertion(config):
    assert('layer_type' in list(config.keys()))
    logging_io.SUCCESS_INFO('LAYER ASSERTION PASS')

def affine_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'input_dim', 'output_dim', 'activation']))
    logging_io.SUCCESS_INFO('AFFINE ASSERTION PASS')

def convolution_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'input_dim', 'output_dim', 'kernel_size1', 'kernel_size2', 'padding', 'strides', 'activation']))
    logging_io.SUCCESS_INFO('OPTION ASSERTION PASS')
    assert(config['padding'] in ['VALID', 'SAME'])
    logging_io.SUCCESS_INFO('PADDING ASSERTION PASS')
    logging_io.SUCCESS_INFO('CONVOLUTION ASSERTION PASS')

def deconvolution_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'output_shape', 'layer_type', 'input_dim', 'output_dim', 'kernel_size1', 'kernel_size2', 'padding', 'strides', 'activation']))
    logging_io.SUCCESS_INFO('OPTION ASSERTION PASS')
    assert(config['padding'] in ['VALID', 'SAME'])
    logging_io.SUCCESS_INFO('PADDING ASSERTION PASS')
    logging_io.SUCCESS_INFO('DECONVOLUTION ASSERTION PASS')

def maxpooling_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'input_dim', 'output_dim', 'padding', 'strides', 'ksize']))
    logging_io.SUCCESS_INFO('MAXPOOLING ASSERTION PASS')

def upsampling_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'output_shape', 'ksize']))
    logging_io.SUCCESS_INFO('UPSAMPLING ASSERTION PASS')

def reshape_assertion(config):
    assert(set(config.keys()).issubset(['inputs', 'outputs', 'layer_type', 'shape']))
    logging_io.SUCCESS_INFO('RESHAPE ASSERTION PASS')

def scope_assertion(scopes):
    assert(len(scopes) == len(list(set(scopes))))
    logging_io.SUCCESS_INFO('SCOPE ASSERTION PASS')

def graph_assertion(config):
    pairs = []
    for key, item in config.items():
        pairs.append((item['inputs'], key))
    assert(len(pairs) == len(list(set(pairs))))
    logging_io.SUCCESS_INFO('GRAPH ASSERTION PASS')
    return pairs
