import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform
import os

config = {
	'image_shape': {'shape':[1, 512, 512, 3]},
	'layers': {
		'conv1': {
			'shape': [16, 16, 3, 8],
			'padding': 'VALID',
			'strides': 16
		},
		'conv2': {
			'shape': [16, 16, 8, 16],
			'padding': 'VALID',
			'strides': 16
		},
		'conv3': {
			'shape': [2, 2, 16, 32],
			'padding': 'VALID',
			'strides': 2
		},
		# 'conv4': {
		# 	'shape': [2, 2, 32, 64],
		# 	'padding': 'VALID',
		# 	'strides': 2
		# },
		# 'conv5': {
		# 	'shape': [2, 2, 64, 128],
		# 	'padding': 'VALID',
		# 	'strides': 2
		# },
		# 'conv6': {
		# 	'shape': [2, 2, 128, 256],
		# 	'padding': 'VALID',
		# 	'strides': 2
		# },
		# 'conv7': {
		# 	'shape': [2, 2, 256, 512],
		# 	'padding': 'VALID',
		# 	'strides': 2
		# },
		'deconv1': {
			'kernel_size': [2, 2, 16, 32],
			'output_shape': [1, 2, 2, 16],
			'strides': 2
		},
		'deconv2': {
			'kernel_size': [16, 16, 8, 16],
			'output_shape': [1, 32, 32, 8],
			'strides': 16
		},
		'deconv3': {
			'kernel_size': [16, 16, 3, 8],
			'output_shape': [1, 512, 512, 3],
			'strides': 16
		},
		# 'deconv4': {
		# 	'kernel_size': [2, 2, 32, 64],
		# 	'output_shape': [1, 16, 16, 32],
		# 	'strides': 2
		# },
		# 'deconv5': {
		# 	'kernel_size': [2, 2, 16, 32],
		# 	'output_shape': [1, 32, 32, 16],
		# 	'strides': 2
		# },
		# 'deconv6': {
		# 	'kernel_size': [4, 4, 8, 16],
		# 	'output_shape': [1, 128, 128, 8],
		# 	'strides': 4
		# },
		# 'deconv7': {
		# 	'kernel_size': [4, 4, 3, 8],
		# 	'output_shape': [1, 512, 512, 3],
		# 	'strides': 4,
		# 	'activation': tf.nn.sigmoid
		# }
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
			self.regularizer = tf.contrib.layers.l1_regularizer(0.1)
			for key, value in self._config['layers'].items():
				# print(key, value)
				if 'activation' not in value.keys():
					activation = tf.nn.sigmoid
				else:
					activation = value['activation']
				with tf.variable_scope('transfer_network', regularizer = self.regularizer):
					if str(key).find('conv') != -1 and str(key).find('deconv') == -1:
						name = str(key)+'kernel'
						kernel = tf.get_variable(name = name, initializer = tf.truncated_normal(value['shape']), regularizer = self.regularizer)
						self.layers.append(activation(tf.nn.conv2d(input = self.layers[-1], filter = kernel, \
							padding = value['padding'], strides = [1, value['strides'], value['strides'], 1])))
					elif str(key).find('reshape') != -1:
						self.layers.append(tf.reshape(self.layers[-1], value['shape']))
					elif str(key).find('deconv') != -1:
						# print(activation)
						name = str(key)+'kernel'
						kernel = tf.get_variable(name = name, initializer = tf.truncated_normal(value['kernel_size']), regularizer = self.regularizer)
						self.layers.append(activation(tf.nn.conv2d_transpose(self.layers[-1], kernel, output_shape=value['output_shape'], strides=[1, value['strides'], value['strides'], 1], padding="SAME")))

			# print(self.layers[-1])
			self.loss = tf.reduce_sum((self.layers[-1] - self.image_placeholder)**2)
			# self.loss = tf.reduce_mean(tf.reduce_sum((self.layers[-1] - self.image_placeholder)))
			# self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.layers[-1], labels = self.image_placeholder))

			self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			tf.summary.scalar('loss', self.loss)
			tf.summary.image('generated', self.layers[-1])
			self.merged_all = tf.summary.merge_all()


	def run(self, inputs, run_mode, iteration = 100):
		image = self._preprocessing(inputs)
		feed_dict = {self.image_placeholder: image}
		res = None
		with self.graph.as_default():
			if run_mode == 'train':
				writer = tf.summary.FileWriter('tensorboard/train', self.sess.graph)
				saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
				try:
					os.mkdir('models')
				except:
					pass
				for i in range(iteration):
					_, merged, loss = self.sess.run([self.optimizer, self.merged_all, self.loss], feed_dict = feed_dict)
					print('Epoch %d => loss: %f \n'%(i + 1, loss))
					writer.add_summary(merged, i + 1)
					if i%10 == 0:
						saver.save(self.sess, 'models/model.ckpt', global_step = i)
					res = loss
			elif run_mode == 'inference':
				saver = tf.train.Saver()
				saver.restore(self.sess, 'models/model.ckpt-'+str(iteration))
				generated = self.sess.run(self.layers[-1], feed_dict = feed_dict)
				res = generated
		return res

	def _preprocessing(self, image):
		res = transform.resize(image, self._config['image_shape']['shape'][1:])
		res = np.array(res).reshape(self._config['image_shape']['shape'])
		return res

	def print_graph(self):
		for layer in self.layers:
			print(layer)




if __name__ == '__main__':

	image = plt.imread('../../data/fangao/xingkong.jpg')
	image = image.astype(np.float32) / 255
	model = TransferModel(config)
	model.print_graph()
	model.run(image, 'train', 3000)
	generated = model.run(image, 'inference')
	generated = generated.reshape([512, 512, 3])
	plt.imshow(generated)
	plt.show()
