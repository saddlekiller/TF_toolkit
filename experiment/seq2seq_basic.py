import tensorflow as tf
from utils import *
import numpy as np
import pickle
import sys
sys.path.append('../../Basic_Tensorflow/src/utils')
from dataProvider import *
from dataDecoder import *
sys.path.append('../utils')
from layers import *

corpus_dir = '../../Basic_Tensorflow/corpus/'
label_map = pickle.load(open(corpus_dir + 'label_map.txt', 'rb'))
dictionary = pickle.load(open(corpus_dir + 'words.dict', 'rb'))

max_word = 35
voc_size = len(dictionary)
embedding_size = 128
train_data = varLenDataProvider(corpus_dir + 'anouymous_corpus_full_train.txt', label_map, max_word = max_word)

inputs_placeholder = tf.placeholder(tf.float32, [None, None, voc_size])


shape = tf.shape(inputs_placeholder)
reshape_layer1 = reshape_layer('reshape_1', [shape[0] * shape[1], max_word])
affine_layer1 = affine_layer('affine_1', [voc_size, embedding_size], 'relu')
reshape_layer2 = reshape_layer('reshape_2', [shape[0], shape[1], embedding_size])
lstm_layer1 = lstm_layer('lstm_1', embedding_size)


reshape_layer1_o = reshape_layer1.outputs(inputs_placeholder)
affine_layer1_o = affine_layer1.outputs(reshape_layer1_o)
reshape_layer2_o = reshape_layer2.outputs(affine_layer1_o)
lstm_layer1_o = lstm_layer1.outputs(reshape_layer2_o)
