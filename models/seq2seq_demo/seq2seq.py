import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../utils')
from data_provider import *
from logging_io import *
from tensorflow.contrib.layers import *

batch_size = 100
max_word = 35

# logging_io.DEBUG_INFO('Generating and saving TRAINING corpus')
# train_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TRAIN.txt', batch_size = 50, max_word = 35, isGenerate = True, isTrain = True)
# logging_io.DEBUG_INFO('Generating and saving VALIDATION corpus')
# valid_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TEST.txt', batch_size = 50, max_word = 35, isGenerate = True, isTrain = False)

logging_io.DEBUG_INFO('Loading TRAINING corpus')
train_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TRAIN.txt', batch_size = batch_size, max_word = max_word, isGenerate = False, isTrain = True)
logging_io.DEBUG_INFO('Loading VALIDATION corpus')
valid_provider = BMWSeqProvider(corpus_dir = '../../data/BMW/TEST.txt', batch_size = batch_size, max_word = max_word, isGenerate = False, isTrain = False)

n_word = train_provider.n_word
n_intent = train_provider.n_intent
n_mention = train_provider.n_mention
max_seq_len = train_provider.max_word
max_intent_len = train_provider.max_intent_len

isInference = False
n_lstm_hidden = 128

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    sentence_placeholder = tf.placeholder(tf.float32, [None, max_seq_len, n_word])
    intent_placeholder = tf.placeholder(tf.float32, [None, max_intent_len, n_intent])
    mention_placeholder = tf.placeholder(tf.float32, [None, max_seq_len, n_mention])
    intent_len_placeholder = tf.placeholder(tf.int32, [None])
    mention_len_placeholder = tf.placeholder(tf.int32, [None])

    sentence_shape = tf.shape(sentence_placeholder)
    sentence_reshape = tf.reshape(sentence_placeholder, [sentence_shape[0] * max_seq_len, n_word])

    with tf.variable_scope('encoder_affine', reuse = isInference):
        encoder_affine  = fully_connected(sentence_placeholder,n_lstm_hidden,weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=tf.nn.relu)

    with tf.variable_scope('encoder_reshape', reuse = isInference):
        encoder_reshape = tf.reshape(encoder_affine, [sentence_shape[0], max_seq_len, n_lstm_hidden])

    with tf.variable_scope('encoder_decoder', reuse = isInference):
        encoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_lstm_hidden, state_is_tuple=True)
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                dtype=tf.float32,
                sequence_length=mention_len_placeholder,
                inputs=encoder_reshape)

    with tf.variable_scope('encoder_decoder', reuse = True):
        decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_lstm_hidden, state_is_tuple=True)
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
                cell=decoder_cell,
                dtype=tf.float32,
                sequence_length=mention_len_placeholder,
                inputs=encoder_outputs)
    # logging_io.WARNING_INFO(mention_len_placeholder)
    with tf.variable_scope('decoder_mention_reshape'):
        decoder_mention_reshape = tf.reshape(decoder_outputs, [sentence_shape[0] * max_seq_len, n_lstm_hidden])

    with tf.variable_scope('decoder_mention_affine'):
        decoder_mention_affine = fully_connected(decoder_mention_reshape,n_mention,weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=tf.nn.relu)

    mention_placeholder_reshape = tf.reshape(mention_placeholder, [-1, n_mention])

    # logging_io.DEBUG_INFO(decoder_mention_affine)
    # # 3500*60
    # logging_io.DEBUG_INFO(mention_placeholder)
    # # 100*35*60
    # logging_io.DEBUG_INFO(mention_placeholder_reshape)
    # # 3500*60
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = decoder_mention_affine, labels = mention_placeholder_reshape))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(decoder_mention_affine, 1), tf.argmax(mention_placeholder_reshape, 1)), tf.float32))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        errs = []
        accs = []
        for batch in train_provider:
            feed_dict = {
                    sentence_placeholder:batch[0],
                    intent_placeholder:batch[1],
                    mention_placeholder:batch[2],
                    intent_len_placeholder:batch[3],
                    mention_len_placeholder:batch[4]
            }
            _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = feed_dict)
            errs.append(err)
            accs.append(acc)
        logging_io.RESULT_INFO('TRAINING Epoch %5d => LOSS: %8f, ACC:%8f'%(i, np.mean(errs), np.mean(accs)))
        if i % 5 == 0:
            errs = []
            accs = []
            for batch in valid_provider:
                feed_dict = {
                        sentence_placeholder:batch[0],
                        intent_placeholder:batch[1],
                        mention_placeholder:batch[2],
                        intent_len_placeholder:batch[3],
                        mention_len_placeholder:batch[4]
                }
                _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = feed_dict)
                errs.append(err)
                accs.append(acc)
            logging_io.RESULT_INFO('____|VALIDATION Epoch %5d => LOSS: %8f, ACC:%8f'%(i, np.mean(errs), np.mean(accs)))
