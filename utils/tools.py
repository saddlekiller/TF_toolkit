import numpy as np
import tensorflow as tf
from config_mapping import *

def acc_sum(results, targets):
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(results, 1), tf.argmax(targets, 1)), tf.float32))

def acc_mean(results, targets):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(results, 1), tf.argmax(targets, 1)), tf.float32))

def loss_sum(results, targets, config):
    return tf.reduce_sum(loss_mapping(config['loss'])(logits = results, labels = targets))

def loss_mean(results, targets, config):
    return tf.reduce_mean(loss_mapping(config['loss'])(logits = results, labels = targets))
