import json
import tensorflow as tf
import sys
sys.path.append('../../utils')
from config_mapping import *
from tools import *
from data_provider import *
from cmd_io import *
from assertions import *

data_conf = json.load(open('data.conf', 'r'))

train_provider = MNISTProvider(data_conf['training_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
valid_provider = MNISTProvider(data_conf['validation_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))

graph = tf.Graph()
model_conf = json.load(open('model.conf', 'r'))
scope_assertion(model_conf['layers'].keys())

with graph.as_default():
    inputs_placeholder = placeholder_mapping(model_conf['inputs_placeholder'])
    targets_placeholder = placeholder_mapping(model_conf['targets_placeholder'])
    layers = []
    outputs = [inputs_placeholder]
    for key in model_conf['layers'].keys():
        layer_config = model_conf['layers'][key]
        layers.append(layers_mapping(key, layer_config))
        outputs.append(layers[-1].outputs(outputs[-1]))
        print(outputs[-1])
    loss = loss_mean(outputs[-1], targets_placeholder, model_conf)
    optimizer = optimizer_mapping(model_conf['optimizer']).minimize(loss)
    accuracy = acc_sum(outputs[-1], targets_placeholder)

    iteration = model_conf['iteration']
    interval = model_conf['interval']
    run_mode = model_conf['run_mode']
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    log = {
        'train':{
            'acc':[],
            'err':[]
        },
        'valid':{
            'acc':[],
            'err':[]
        },
        'test':{
            'acc':[],
            'err':[]
        }
    }
    round_val = 5
    if run_mode == 'train':
        for i in range(1, iteration+1):
            errs = 0
            accs = 0
            for batch_train_inputs, batch_train_targets in train_provider:
                _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = {inputs_placeholder:batch_train_inputs.reshape([-1, 28, 28, 1]), targets_placeholder:batch_train_targets})
                errs += err
                accs += acc
            log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
            log['train']['acc'].append(round(accs/train_provider.n_samples(), round_val))
            cmd_print(5, 'Epoch {0:5} Training LOSS: {1:5} Training ACC: {2:5}'.format(i, log['train']['err'][-1], log['train']['acc'][-1]))
            if i % interval == 0:
                errs = 0
                accs = 0
                for batch_valid_inputs, batch_valid_targets in valid_provider:
                    err, acc = sess.run([loss, accuracy], feed_dict = {inputs_placeholder:batch_valid_inputs.reshape([-1, 28, 28, 1]), targets_placeholder:batch_valid_targets})
                    errs += err
                    accs += acc
                log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
                log['valid']['acc'].append(round(accs/valid_provider.n_samples(), round_val))
                cmd_print(5, 'Epoch {0:5} validation LOSS: {1:5} validation ACC: {2:5}'.format(i, log['valid']['err'][-1], log['valid']['acc'][-1]))
