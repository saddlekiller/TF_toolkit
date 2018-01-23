import tensorflow as tf
from params import *


graph = tf.Graph()
with graph.as_default():
    v1 = truncated_normal('v1', [1])
    v2 = random_normal('v2', [1])
    v3 = random_uniform('v3', [1])
    print(v1)
    print(v2)    
    print(v3)    