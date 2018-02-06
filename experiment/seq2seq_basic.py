import tensorflow as tf
import numpy as np
import pickle
import sys
sys.path.append('../../Basic_Tensorflow/src/utils')
# from dataProvider import *
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
from data_provider import *
#
#
corpus_dir = '../../Basic_Tensorflow/corpus/'
label_map = pickle.load(open(corpus_dir + 'poilabel_map.npz', 'rb'))
dictionary = pickle.load(open(corpus_dir + 'poiwords.dict', 'rb'))

max_word = 35
voc_size = len(dictionary)
embedding_size = 128
provider = idProvider(corpus_dir + 'anouymous_corpus_full_train.txt', 50) #anonymous_raw_poi_train.txt


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

                self.cost = tf.losses.sparse_softmax_cross_entropy(labels, logits_flat)
                self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.cost)
            else:
                self.prob = tf.nn.softmax(logits)
            self.sess = tf.Session()


    def run(self, model_path):
        with self.graph.as_default():
            if self.is_inference == False:
                saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
                self.sess.run(tf.global_variables_initializer())
                for i in range(1):
                    losses = []
                    for length, batch_inputs, batch_targets in provider:
                        print(length, np.array(batch_inputs).shape)
                        feed_dict = {self.inputs_placeholder: batch_inputs, self.seq_length: [length]*len(batch_targets)}
                        _, loss = self.sess.run([self.optimizer, self.cost], feed_dict = feed_dict)
                        losses.append(loss)
                        saver.save(sess, model_path, global_step = i)
                        print(losses)
                        break
                    logging_io.DEBUG_INFO('EPOCH: '+ str(i+1)+' LOSS:' + str(np.mean(losses)))
            else:
                saver = tf.train.Saver()
                saver.restore(self.sess, model_path)
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
                    loss, p = self.sess.run([self.cost, self.prob], feed_dict = feed_dict)
                    print(p.shape)
                    # rebuilt_sentence = [dictionary[i] for i in ]
                    raise TypeError

if __name__ == '__main__':

    model_path = 'model.ckpt'
    training = seq2seq(False)
    # validation = seq2seq(True)
    training.run(model_path)
    # validation.run(model_path)
