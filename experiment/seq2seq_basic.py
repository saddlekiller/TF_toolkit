import tensorflow as tf
import numpy as np
import pickle
import sys
# sys.path.append('../../Basic_Tensorflow/src/utils')
# from dataProvider import *
# from dataDecoder import *
sys.path.append('../utils')
from assertions import *
from config_mapping import *
from tools import *
from layers import *
# from tensorflow.contrib.seq2seq import *
from params import *
from logging_io import *
# from tensorflow.python.layers.core import Dense
from data_provider import *
#
#
# corpus_dir = '../../Basic_Tensorflow/corpus/'
# label_map = pickle.load(open(corpus_dir + 'poilabel_map.npz', 'rb'))
# dictionary = pickle.load(open(corpus_dir + 'poiwords.dict', 'rb'))
#
# max_word = 35
# voc_size = len(dictionary)
# embedding_size = 128
# provider = idProvider(corpus_dir + 'anouymous_corpus_full_train.txt', 50) #anonymous_raw_poi_train.txt

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed, Dense
from keras.layers.core import RepeatVector, Activation
# from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder

import matplotlib.pyplot as plt

def build_model(input_size, max_out_seq_len, hidden_size):

    model = Sequential()
    model.add( Dense(hidden_size, activation="relu") )
    model.add( RepeatVector(max_out_seq_len) )
    model.add( LSTM(hidden_size, return_sequences=True) )
    # model.add(  )
    model.compile(loss="mse", optimizer='adam')
    return model

if __name__ == '__main__':

    filename = '../../Basic_Tensorflow/corpus/anonymous_raw_poi_train_trimmed.txt'
    label_map_dir = '../../Basic_Tensorflow/corpus/raw_poilabel_map.npz'
    dictionary_dir = '../../Basic_Tensorflow/corpus/raw_poiwords.dict'
    max_word = 30
    provider = KerasSeq2SeqDataProvider(filename, 50, label_map_dir, dictionary_dir, max_word, True)
    voc_size = provider._n_voc_size
    tag_size = provider._n_classes
    hidden_dim = 128



    
















#
