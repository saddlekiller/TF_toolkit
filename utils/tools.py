import numpy as np
import tensorflow as tf
from config_mapping import *
import pickle


def acc_sum(results, targets):
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(results, 1), tf.argmax(targets, 1)), tf.float32))

def acc_mean(results, targets):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(results, 1), tf.argmax(targets, 1)), tf.float32))

def loss_sum(results, targets, config):
    return tf.reduce_sum(loss_mapping(config['loss'])(logits = results, labels = targets))

def loss_mean(results, targets, config):
    return tf.reduce_mean(loss_mapping(config['loss'])(logits = results, labels = targets))

def raw2ids(sentence, dictionary):
    words = []
    for word in sentence:
        if word in dictionary:
            words.append(dictionary.index(word))
        else:
            word.append(dictionary.index('UNKNOWN'))
    return words


def ids2raw(ids, dictionary):
    try:
        sentence = ''.join([dictionary[i] for i in ids])
        return sentence
    except:
        logging_io.ERROR_INFO('Some words are not included in vocabulary, please check it!')
    return None

# if __name__ == '__main__':
#     corpus_dir = '../../Basic_Tensorflow/corpus/'
#     # label_map = pickle.load(open(corpus_dir + 'poilabel_map.npz', 'rb'))
#     dictionary = pickle.load(open(corpus_dir + 'poiwords.dict', 'rb'))
#     raw_sentence = '我要去上海'
#     ids = raw2ids(raw_sentence, dictionary)
#     print(ids)
#     print(ids2raw(ids, dictionary))
