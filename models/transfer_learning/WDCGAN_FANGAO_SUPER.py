import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../utils')
from data_provider import *
from tools import *
from tensorflow.contrib.layers import *



output_dim = 512
input_dim = 784
hidden_dim1 = 512
hidden_dim2 = 128
hidden_dim3 = 64

def leaky_relu(x, alpha=0.2):
	return tf.maximum(tf.minimum(0.0, alpha * x), x)


graph = tf.Graph()
with graph.as_default():

	def Generator(inputs):
	#
		with tf.variable_scope('Generator'):

			Generator_affine_1  = fully_connected(inputs, 4096,
								weights_initializer=tf.random_normal_initializer(stddev=0.02),
								# weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								activation_fn=tf.nn.relu)
			Generator_reshape_1 = tf.reshape(Generator_affine_1, [1, 1, 1, 4096])
			Generator_deconv_1  = convolution2d_transpose(Generator_reshape_1, 2048, [2, 2], [2, 2],
								weights_initializer=tf.random_normal_initializer(stddev=0.02),
								# weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								activation_fn=tf.nn.relu)

			Generator_deconv_2  = convolution2d_transpose(Generator_deconv_1, 1024, [2, 2], [2, 2],
								weights_initializer=tf.random_normal_initializer(stddev=0.02),
								# weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								activation_fn=tf.nn.relu)

			Generator_deconv_3  = convolution2d_transpose(Generator_deconv_2, 512, [2, 2], [2, 2],
								weights_initializer=tf.random_normal_initializer(stddev=0.02),
								# weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								activation_fn=tf.nn.relu)

			Generator_deconv_4  = convolution2d_transpose(Generator_deconv_3, 256, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.relu)

			Generator_deconv_5  = convolution2d_transpose(Generator_deconv_4, 128, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.sigmoid)

			Generator_deconv_6  = convolution2d_transpose(Generator_deconv_5, 64, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.sigmoid)

			Generator_deconv_7  = convolution2d_transpose(Generator_deconv_6, 32, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.sigmoid)

			Generator_deconv_8  = convolution2d_transpose(Generator_deconv_7, 16, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.sigmoid)

			Generator_deconv_9  = convolution2d_transpose(Generator_deconv_8, 8, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.sigmoid)

			Generator_deconv_10  = convolution2d_transpose(Generator_deconv_9, 3, [2, 2], [2, 2],
								 weights_initializer=tf.random_normal_initializer(stddev=0.02),
								 # weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
								 activation_fn=tf.nn.sigmoid)

			print('*'*49)
			print('*'*1 + ' '*19 + 'Generator' + ' '*19 + '*'*1)
			print('*'*49)
			print(Generator_affine_1)
			print(Generator_deconv_1)
			print(Generator_deconv_2)
			print(Generator_deconv_3)
			print(Generator_deconv_4)
			print(Generator_deconv_5)
			print(Generator_deconv_6)
			print(Generator_deconv_7)
			print(Generator_deconv_8)
			print(Generator_deconv_9)
			print(Generator_deconv_10)
			print('*'*49)


		return Generator_deconv_10

	def Discriminator(inputs, reuse = False):

		with tf.variable_scope('Discriminator', reuse=reuse):

			Discriminator_k1 = tf.get_variable('Discriminator_k1', initializer = tf.truncated_normal([2, 2, 3, 8]))
			Discriminator_b1 = tf.get_variable('Discriminator_b1', initializer = tf.truncated_normal([8]))
			Discriminator_k2 = tf.get_variable('Discriminator_k2', initializer = tf.truncated_normal([2, 2, 8, 16]))
			Discriminator_b2 = tf.get_variable('Discriminator_b2', initializer = tf.truncated_normal([16]))
			Discriminator_k3 = tf.get_variable('Discriminator_k3', initializer = tf.truncated_normal([2, 2, 16, 32]))
			Discriminator_b3 = tf.get_variable('Discriminator_b3', initializer = tf.truncated_normal([32]))
			Discriminator_k4 = tf.get_variable('Discriminator_k4', initializer = tf.truncated_normal([2, 2, 32, 64]))
			Discriminator_b4 = tf.get_variable('Discriminator_b4', initializer = tf.truncated_normal([64]))
			Discriminator_k5 = tf.get_variable('Discriminator_k5', initializer = tf.truncated_normal([2, 2, 64, 128]))
			Discriminator_b5 = tf.get_variable('Discriminator_b5', initializer = tf.truncated_normal([128]))
			Discriminator_k6 = tf.get_variable('Discriminator_k6', initializer = tf.truncated_normal([2, 2, 128, 256]))
			Discriminator_b6 = tf.get_variable('Discriminator_b6', initializer = tf.truncated_normal([256]))
			Discriminator_k7 = tf.get_variable('Discriminator_k7', initializer = tf.truncated_normal([2, 2, 256, 512]))
			Discriminator_b7 = tf.get_variable('Discriminator_b7', initializer = tf.truncated_normal([512]))
			Discriminator_k8 = tf.get_variable('Discriminator_k8', initializer = tf.truncated_normal([2, 2, 512, 1024]))
			Discriminator_b8 = tf.get_variable('Discriminator_b8', initializer = tf.truncated_normal([1024]))
			Discriminator_k9 = tf.get_variable('Discriminator_k9', initializer = tf.truncated_normal([2, 2, 1024, 2048]))
			Discriminator_b9 = tf.get_variable('Discriminator_b9', initializer = tf.truncated_normal([2048]))
			Discriminator_k10 = tf.get_variable('Discriminator_k10', initializer = tf.truncated_normal([2, 2, 2048, 4096]))
			Discriminator_b10 = tf.get_variable('Discriminator_b10', initializer = tf.truncated_normal([4096]))
			# Discriminator_k6 = tf.get_variable('Discriminator_k6', initializer = tf.truncated_normal([4, 4, 256, 512]))
			# Discriminator_b6 = tf.get_variable('Discriminator_b6', initializer = tf.truncated_normal([512]))

			Discriminator_wa1 = tf.get_variable('Discriminator_wa1', initializer = tf.truncated_normal([4096, 1024]))
			Discriminator_ba1 = tf.get_variable('Discriminator_ba1', initializer = tf.truncated_normal([1024]))
			Discriminator_wa2 = tf.get_variable('Discriminator_wa2', initializer = tf.truncated_normal([1024, 256]))
			Discriminator_ba2 = tf.get_variable('Discriminator_ba2', initializer = tf.truncated_normal([256]))
			Discriminator_wa3 = tf.get_variable('Discriminator_wa3', initializer = tf.truncated_normal([256,  1]))
			Discriminator_ba3 = tf.get_variable('Discriminator_ba3', initializer = tf.truncated_normal([ 1]))
			#
			Discriminator_conv_1	= leaky_relu(tf.add(tf.nn.conv2d(input = inputs,			   filter = Discriminator_k1, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b1))
			Discriminator_conv_2	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_1, filter = Discriminator_k2, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b2))
			Discriminator_conv_3	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_2, filter = Discriminator_k3, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b3))
			Discriminator_conv_4	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_3, filter = Discriminator_k4, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b4))
			Discriminator_conv_5	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_4, filter = Discriminator_k5, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b5))
			Discriminator_conv_6	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_5, filter = Discriminator_k6, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b6))
			Discriminator_conv_7	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_6, filter = Discriminator_k7, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b7))
			Discriminator_conv_8	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_7, filter = Discriminator_k8, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b8))
			Discriminator_conv_9	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_8, filter = Discriminator_k9, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b9))
			Discriminator_conv_10	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_9, filter = Discriminator_k10, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b10))


			# Discriminator_conv_6	= leaky_relu(tf.add(tf.nn.conv2d(input = Discriminator_conv_5, filter = Discriminator_k6, padding = 'VALID', strides = [1, 2, 2, 1]), Discriminator_b6))
			Discriminator_reshape_1 = tf.reshape(Discriminator_conv_10, [-1, 4096])
			Discriminator_affine_1  = leaky_relu(tf.add(tf.matmul(Discriminator_reshape_1, Discriminator_wa1), Discriminator_ba1))
			Discriminator_affine_2  = leaky_relu(tf.add(tf.matmul(Discriminator_affine_1 , Discriminator_wa2), Discriminator_ba2))
			Discriminator_affine_3  = leaky_relu(tf.add(tf.matmul(Discriminator_affine_2 , Discriminator_wa3), Discriminator_ba3))
			#
			print('*'*53)
			print('*'*1 + ' '*19 + 'Discriminator' + ' '*19 + '*'*1)
			print('*'*53)
			print(inputs)
			print(Discriminator_conv_1)
			# print(Discriminator_pooling_1)
			print(Discriminator_conv_2)
			# print(Discriminator_pooling_2)
			print(Discriminator_conv_3)
			print(Discriminator_conv_4)
			print(Discriminator_conv_5)
			print(Discriminator_conv_6)
			print(Discriminator_conv_7)
			print(Discriminator_conv_8)
			print(Discriminator_conv_9)

			print(Discriminator_reshape_1)
			print(Discriminator_affine_1)
			print(Discriminator_affine_2)
			print(Discriminator_affine_3)
			print('*'*53)

		return Discriminator_affine_3


	data_placeholder  = tf.placeholder(tf.float32, [1, 1024, 1024, 3])
	prior_placeholder = tf.placeholder(tf.float32, [1, 512])

	Generator_out		  = Generator(prior_placeholder)

	Discriminator_fake_out = Discriminator(Generator_out, False)
	Discriminator_real_out = Discriminator(data_placeholder, True)
	#
	Generator_loss		  = tf.reduce_mean(Discriminator_fake_out)
	Discriminator_loss	  = tf.reduce_mean(Discriminator_real_out) - tf.reduce_mean(Discriminator_fake_out)

	Generator_variables	 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
	Discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
	# print(Generator_variables)

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		Generator_optimizer	 = tf.train.RMSPropOptimizer(0.1).minimize(Generator_loss,	 var_list = Generator_variables)
		Discriminator_optimizer = tf.train.RMSPropOptimizer(0.1).minimize(Discriminator_loss, var_list = Discriminator_variables)


	Clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in Discriminator_variables]
	# a = [var for var in tf.global_variables() if 'Discriminator' in var.name]
	# Clip = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")]

	# print('-'*50)
	# for v in Clip:
	#	 print(v)
	# print('-'*50)
	# for ai in a:
	#	 print(ai)
	# print('-'*50)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# merged_all = tf.summary.merge_all()
	writer = tf.summary.FileWriter('tensorboard', sess.graph)

	image = plt.imread('../../data/fangao/xingkong.jpg')
	image = np.array(image.astype(np.float32) / 255)
	def preprocessing(image):
		res = transform.resize(image, [1024, 1024,3])
		res = np.array(res).reshape([1, 1024, 1024, 3])
		return res
	image = preprocessing(image)

	for i in range(10000):
		d_losses = []
		g_losses = []
		for j in range(5):
			noise = np.random.uniform(-1, 1, [1, 512]).astype(np.float32)
			feed_dict = {data_placeholder: image, prior_placeholder: noise}
			_, d_loss		  = sess.run([Discriminator_optimizer, Discriminator_loss]	   , feed_dict = feed_dict)
			sess.run(Clip)
		noise = np.random.uniform(-1, 1, [1, 512]).astype(np.float32)
		feed_dict = {data_placeholder: image, prior_placeholder: noise}
		_, g_loss, g_image = sess.run([Generator_optimizer, Generator_loss, Generator_out], feed_dict = feed_dict)

		d_losses.append(d_loss)
		g_losses.append(g_loss)
		print('EPOCH %d, D_LOSS: %f, G_LOSS: %f '%(i, np.mean(d_losses), np.mean(g_losses)))
		g_image = g_image.reshape([1024, 1024, 3])
		# merge_image = build_image_(g_image, 10)
		# print(np.array(g_image).shape)
		if i%25 == 0:
			for ii in range(3):
				g_image[:, :, ii] = (g_image[:, :, ii] - np.min(g_image[:, :, ii])) / (np.max(g_image[:, :, ii]) - np.min(g_image[:, :, ii]))
			plt.imsave('images/' + 'image_'+str(i+1)+'.png', g_image)
