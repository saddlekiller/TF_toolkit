import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform


config = {
	'image_shape':[1, 64, 64, 3],
	'affine1': [64*64*3, 32*32*3],
	'affine2': [32*32*3, 16*16*3],
	'affine3': [16*16*3, 32*32*3],
	'affine4': [32*32*3, 64*64*3]
}


class AffineTransferModel():

	def __init__(self, config):
		self._config = config
		self._model_builder()

	def _model_builder(self):
		self.image_placeholder = tf.placeholder(tf.float16, self._config['image_shape'])
		self.layers = [tf.reshape(self.image_placeholder, [1, -1])]
		for key, value in self._config.items():
			if str(key).find('affine') != -1:
				with tf.variable_scope(str(key)):
					weight = tf.get_variable(name = 'weight', shape = value, initializer = tf.truncated_normal_initializer, dtype=tf.float16)
					bias = tf.get_variable(name = 'bias', shape = value[-1], initializer = tf.truncated_normal_initializer, dtype=tf.float16)
					self.layers.append(tf.nn.relu(tf.add(tf.matmul(self.layers[-1], weight), bias)))
					# raise TypeError
		# self.layers.append(tf.reshape(self.layers[-1], self._config['image_shape']))
		print(self.layers[0])
		print(self.layers[-1])
		# raise TypeError
		self.loss = tf.reduce_sum((self.layers[-1] - self.layers[0])**2)
		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
		# raise TypeError
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		# raise TypeError

	def run(self, inputs, run_mode, iteration = 100):
		image = self._preprocessing(inputs)
		# print(image.shape)
		feed_dict = {self.image_placeholder: image}
		for i in range(iteration):
			_, loss = self.sess.run([self.optimizer, self.loss], feed_dict = feed_dict)
			print('Epoch %d => loss: %f'%(i+1, loss))

	def _preprocessing(self, image):
		res = transform.resize(image, self._config['image_shape'][1:])
		res = np.array(res).reshape(self._config['image_shape']).astype(np.float16)
		return res

	def print_graph(self):
		for layer in self.layers:
			print(layer)



if __name__ == '__main__':

	image = plt.imread('../../data/fangao/xingkong.jpg')
	image = image.astype(np.float16) / 255
	model = AffineTransferModel(config)
	model.print_graph()
	model.run(image, 'train')
