def dtype_assertion(dtype):
    assert(dtype in ['float32', 'int32', 'bool'])

def activation_assertion(activation):
    assert(activation in ['relu', 'sigmoid', 'tanh', 'softplus', ''])

def optimizer_assertion(optimizer):
    assert(optimizer in ['adam', 'sgd', 'adamgrad'])

def run_mode_assertion(run_mode):
    assert(run_mode in ['train', 'valid', 'test', 'continue_train'])
