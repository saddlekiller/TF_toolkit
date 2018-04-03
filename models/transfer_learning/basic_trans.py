import tensorflow as tf
import numpy as np

config = {
    'image_shape': {'shape':[1, 450, 309, 3]},
    'layers': {
        'conv1': {
            'shape': [14, 10, 3, 8],
            'padding': 'VALID',
            'strides': 2
        },
        'conv2': {
            'shape': [14, 10, 8, 16],
            'padding': 'VALID',
            'strides': 2
        },
        'conv3': {
            'shape': [14, 10, 16, 32],
            'padding': 'VALID',
            'strides': 2
        },
        'conv4': {
            'shape': [14, 10, 32, 64],
            'padding': 'VALID',
            'strides': 2
        },
        'conv5': {
            'shape': [14, 10, 64, 128],
            'padding': 'VALID',
            'strides': 2
        },
        'conv6': {
            'shape': [2, 1, 128, 256],
            'padding': 'VALID',
            'strides': 2
        },
        'deconv1': {
            'kernel_size': [2, 1, 128, 256],
            'output_shape': [1, 2, 1, 128],
            'strides': 2
        },
        'deconv2': {
            'kernel_size': [14, 10, 64, 128],
            'output_shape': [1, 16, 11, 64],
            'strides': 2
        },
        'deconv3': {
            'kernel_size': [14, 10, 32, 64],
            'output_shape': [1, 45, 31, 32],
            'strides': 2
        },
        'deconv4': {
            'kernel_size': [14, 10, 16, 32],
            'output_shape': [1, 103, 71, 16],
            'strides': 2
        },
        'deconv5': {
            'kernel_size': [14, 10, 8, 16],
            'output_shape': [1, 219, 150, 8],
            'strides': 2
        },
        'deconv6': {
            'kernel_size': [14, 10, 3, 8],
            'output_shape': [1, 450, 309, 3],
            'strides': 2
        }
    }
}


class TransferModel(object):

    def __init__(self, config):
        self._config = config
        self._model_builder()
        pass

    def _model_builder(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_placeholder = tf.placeholder(tf.float32, self._config['image_shape']['shape'], 'image')
            self.layers = [self.image_placeholder]
            for key, value in self._config['layers'].items():
                # print(key, value)
                if str(key).find('conv') != -1 and str(key).find('deconv') == -1:
                    self.layers.append(tf.layers.conv2d(inputs = self.layers[-1], filters = value['shape'][-1], \
                        kernel_size = [value['shape'][0], value['shape'][1]], \
                        padding = value['padding'], activation = tf.nn.relu, strides = value['strides']))
                elif str(key).find('reshape') != -1:
                    self.layers.append(tf.reshape(self.layers[-1], value['shape']))
                elif str(key).find('deconv') != -1:
                    # print(str(key)+'_kernel')
                    name = str(key)+'kernel'
                    kernel = tf.get_variable(name = name, initializer = tf.truncated_normal(value['kernel_size']))
                    # tf.get_variable(, tf.truncated_normal())
                    self.layers.append(tf.nn.conv2d_transpose(self.layers[-1], kernel, output_shape=value['output_shape'], strides=[1, value['strides'], value['strides'], 1], padding="SAME"))
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def run(self, run_mode):
        image = np.random.random((1, 350, 309, 3))
        for i in range(10):
            self.run(self.layers[-1], feed_dict = {self.image_placeholder: image})

    def print_graph(self):
        for layer in self.layers:
            print(layer)




if __name__ == '__main__':
    model = TransferModel(config)
    model.print_graph()
