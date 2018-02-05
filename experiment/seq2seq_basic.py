import tensorflow as tf
from utils import *
import numpy as np
import pickle
import sys
sys.path.append('../../Basic_Tensorflow/src/utils')
from dataProvider import *
from dataDecoder import *

corpus_dir = '../../Basic_Tensorflow/corpus/'
label_map = pickle.load(open(corpus_dir + 'label_map.txt', 'rb'))
dictionary = pickle.load(open(corpus_dir + 'words.dict', 'rb'))
train_data = varLenDataProvider(corpus_dir + 'anouymous_corpus_full_train.txt', label_map)
    
