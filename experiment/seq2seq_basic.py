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


corpus_dir = '../../Basic_Tensorflow/corpus/'
label_map = pickle.load(open(corpus_dir + 'poilabel_map.npz', 'rb'))
dictionary = pickle.load(open(corpus_dir + 'poiwords.dict', 'rb'))

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
provider = idProvider(corpus_dir + 'anouymous_corpus_full_train.txt', 50)
# for length, inputs, targets in provider:
#     print(length, len(inputs))
    # pass


inputs_placeholder = tf.placeholder(tf.float32, [None, None], name = 'inputs_ids')
seq_length = tf.placeholder(tf.int32, 'sequence_length')
embedding = truncated_normal('embedding', [])
