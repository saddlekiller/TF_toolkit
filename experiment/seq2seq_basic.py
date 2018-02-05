import tensorflow as tf
import numpy as np
import pickle
import sys
sys.path.append('../../Basic_Tensorflow/src/utils')
from dataProvider import *
from dataDecoder import *
sys.path.append('../utils')
from assertions import *
from config_mapping import *
from tools import *
from layers import *
from tensorflow.contrib.seq2seq import *
from params import *
from logging_io import *
from tensorflow.python.layers.core import Dense
#
#
corpus_dir = '../../Basic_Tensorflow/corpus/'
label_map = pickle.load(open(corpus_dir + 'raw_poilabel_map.npz', 'rb'))
dictionary = pickle.load(open(corpus_dir + 'raw_poiwords.dict', 'rb'))

class idProvider():

    def __init__(self, filename, batch_size):
        self.filename = filename
        self.batch_size = batch_size
        self._currentPosition = 0
        self.loadCorpus()
        # for key in self.batched_inputs.keys():
        #     print(key, len(self.batched_inputs[key]))

    def reset(self):
        self._currentOrder = self.keys.copy()
        self.new_epoch()

    def new_epoch(self):
        np.random.shuffle(self._currentOrder)
        self._currentPosition = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._currentPosition >= self.n_batches:
            self.new_epoch()
            raise StopIteration
        batch_input = self.batched_inputs[self._currentOrder[self._currentPosition]]
        batch_target = self.batched_targets[self._currentOrder[self._currentPosition]]
        length = int(self._currentOrder[self._currentPosition].split('_')[0])
        # print(self._currentOrder[self._currentPosition])
        self._currentPosition += 1
        return length, batch_input, batch_target

    def next(self):
        return self.__next__()

    def loadCorpus(self):
        lines = open(self.filename).readlines()
        varLen_inputs = dict()
        varLen_targets = dict()
        for line in lines:
            splited = line.split()
            ids = [int(i) for i in splited[1:]]
            targets = int(splited[0])
            key = len(ids)
            if key not in varLen_inputs.keys():
                varLen_inputs[key] = []
                varLen_targets[key] = []
            varLen_inputs[key].append(ids)
            varLen_targets[key].append(targets)
        self.batched_inputs = dict()
        self.batched_targets = dict()
        for key in varLen_targets.keys():
            if len(varLen_targets[key])%self.batch_size == 0:
                count = int(len(varLen_targets[key])/self.batch_size)
            else:
                count = int(len(varLen_targets[key])/self.batch_size) + 1
            for i in range(count):
                new_key = str(key) + '_' + str(i)
                if i == count - 1:
                    self.batched_inputs[new_key] = varLen_inputs[key][i*self.batch_size:]
                    self.batched_targets[new_key] = varLen_targets[key][i*self.batch_size:]
                else:
                    self.batched_inputs[new_key] = varLen_inputs[key][i*self.batch_size:(i+1)*self.batch_size]
                    self.batched_targets[new_key] = varLen_targets[key][i*self.batch_size:(i+1)*self.batch_size]

        self.keys = list(self.batched_targets.keys())
        self.n_batches = len(self.keys)
        self.reset()

max_word = 35
voc_size = len(dictionary)
embedding_size = 128
provider = idProvider(corpus_dir + 'anonymous_raw_poi_train.txt', 50)






# graph = tf.Graph()
# with graph.as_default():
class seq2seq():

    def __init__(self, is_inference):

        self.is_inference = is_inference
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_placeholder = tf.placeholder(tf.int32, [None, None], name = 'inputs_ids')
            encoder_embedding_weight = truncated_normal('encoder_embedding', [voc_size, embedding_size]).get()
            encoder_embedding = tf.nn.embedding_lookup(encoder_embedding_weight, self.inputs_placeholder)
            with tf.variable_scope('encoder_cell'):
                encoder_lstm_cell = tf.contrib.rnn.LSTMCell(64)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_lstm_cell, encoder_embedding, dtype=tf.float32)



            decoder_embedding_weight = truncated_normal('decoder_embedding', [voc_size, embedding_size]).get()
            # decoder_outputs, decoder_states = tf.nn.dynamic_rnn(decoder_cell, encoder_embedding)

            if self.is_inference:
                self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
                self.end_tokens = tf.placeholder(tf.int32, name='end_tokens')
                helper = GreedyEmbeddingHelper(decoder_embedding, self.start_tokens, self.end_tokens)
            else:
                self.seq_length = tf.placeholder(tf.int32, [None], name = 'sequence_length')
                decoder_embedding = tf.nn.embedding_lookup(decoder_embedding_weight, self.inputs_placeholder)
                helper = TrainingHelper(decoder_embedding, self.seq_length)

            affine_layer = Dense(voc_size)
            with tf.variable_scope('decoder_cell'):
                decoder_lstm_cell = tf.contrib.rnn.LSTMCell(64)
            logging_io.WARNING_INFO(str(decoder_lstm_cell))
            logging_io.WARNING_INFO(str(affine_layer))
            decoder = BasicDecoder(decoder_lstm_cell, helper, encoder_states, affine_layer)
            logits, final_states, final_sequence_lengths = dynamic_decode(decoder)

            if not self.is_inference:
                logits_flat = tf.reshape(logits.rnn_output, [-1, voc_size])
                labels = tf.reshape(self.inputs_placeholder, [-1])

                logging_io.WARNING_INFO(str(logits.rnn_output))

                cost = tf.losses.sparse_softmax_cross_entropy(labels, logits_flat)
                optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
            else:
                self.prob = tf.nn.softmax(logits)
            self.sess = tf.Session()


    def run(self, is_inference):
        with self.graph.as_default():
            # sess = tf.Session()
            self.is_inference = is_inference
            if is_inference:
                self.sess.run(tf.global_variables_initializer())
                for i in range(1):

                    losses = []
                    for length, batch_inputs, batch_targets in provider:
                        feed_dict = {self.inputs_placeholder: batch_inputs, self.seq_length: [length]*len(batch_targets)}
                        _, loss = self.sess.run([optimizer, cost], feed_dict = feed_dict)
                        losses.append(loss)
                        break
                    logging_io.DEBUG_INFO('EPOCH: '+ str(i+1)+' LOSS:' + str(np.mean(losses)))
            else:
                while(True):
                    sentence = input()
                    # print(sentence)
                    ids = [dictionary.index('<BEGIN>')]
                    for word in sentence:

                        try:
                            ids.append(dictionary.index(word))
                        except:
                            ids.append(dictionary.index('UNKNOWN'))
                    ids.append(dictionary.index('<END>'))
                    print(ids)
                    ids = [ids]
                    length = len(ids)
                    feed_dict = {self.inputs_placeholder: ids, self.start_tokens:['<BEGIN>']*length, self.end_tokens:'<END>'}
                    loss, p = self.sess.run([cost, prob], feed_dict = feed_dict)
                    print(p.shape)
                    # rebuilt_sentence = [dictionary[i] for i in ]
                    raise TypeError

if __name__ == '__main__':

    model = seq2seq(False)
    model.run(False)
    model.run(True)
