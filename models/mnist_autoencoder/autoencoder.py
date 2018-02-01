import json
import tensorflow as tf
import sys
sys.path.append('../../utils')
from config_mapping import *
from tools import *
from data_provider import *
# from cmd_io import *
from model_builder import *
from logging import *


data_conf_dir = 'data.conf'
model_conf_dir = 'model.conf'
data_provider = MNISTProvider
# log = basic_builder(data_conf_dir, model_conf_dir, data_provider)
if 1 == 1:
    data_conf = json.load(open(data_conf_dir, 'r'))
    model_conf = json.load(open(model_conf_dir, 'r'))
    train_provider = data_provider(data_conf['training_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    valid_provider = data_provider(data_conf['validation_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))

    iteration = model_conf['iteration']
    interval = model_conf['interval']
    run_mode = model_conf['run_mode']
    graph = tf.Graph()
    with graph.as_default():
        inputs_placeholder = placeholder_mapping(model_conf['inputs_placeholder'])
        targets_placeholder = placeholder_mapping(model_conf['targets_placeholder'])
        layers = dict()
        outputs = dict()
        outputs['inputs_placeholder'] = inputs_placeholder
        for key in model_conf['layers'].keys():
            layers[key] = layers_mapping(key, model_conf['layers'][key])
            print(layers[key])
        for layer_name, layer in layers.items():
            outputs[layer_name] = layer.outputs(outputs[model_conf['layers'][layer_name]['inputs']])
            print(outputs[layer_name])
