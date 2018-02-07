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
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.layers.embeddings import Embedding
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder

def build_model(input_size, max_out_seq_len, hidden_size):

    model = Sequential()
    model.add( Dense(hidden_size, activation="relu") )
    model.add( RepeatVector(max_out_seq_len) )
    model.add( LSTM(hidden_size, return_sequences=True) )
    model.add( TimeDistributed(Dense(output_dim=input_size, activation="linear")) )
    model.compile(loss="mse", optimizer='adam')
    return model

if __name__ == '__main__':

    filename = '../../Basic_Tensorflow/corpus/anonymous_raw_poi_train_trimmed.txt'
    label_map_dir = '../../Basic_Tensorflow/corpus/raw_poilabel_map.npz'
    dictionary_dir = '../../Basic_Tensorflow/corpus/raw_poiwords.dict'
    provider = KerasSeq2SeqDataProvider(filename, 50, label_map_dir, dictionary_dir, 30)
    voc_size = provider._n_voc_size
    tag_size = provider._n_classes
    hidden_dim = 128
    max_in_seq_len = 30
    max_out_seq_len = 30

    # encoder = LSTM(hidden_dim, return_sequences=True)
    # decoder = AttentionDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim, output_length=max_out_seq_len, state_input=False, return_sequences=True)
    # model = Sequential()
    # model.add(Embedding(voc_size, hidden_dim, input_length=))
    for batch_input, batch_target in provider:
        break
