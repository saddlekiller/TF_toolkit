import json
import tensorflow as tf
from config_mapping import *
from tools import *
from data_provider import *
# from cmd_io import *
from logging_io import *

def basic_builder(data_conf_dir, model_conf_dir, data_provider, round_val=5, saveLOG=True):

    data_conf = json.load(open(data_conf_dir, 'r'))
    model_conf = json.load(open(model_conf_dir, 'r'))
    graph_assertion(model_conf)
    train_provider = data_provider(data_conf['training_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    valid_provider = data_provider(data_conf['validation_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    assert(isinstance(train_provider, dataProvider))
    assert(isinstance(valid_provider, dataProvider))
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
    iteration = model_conf['iteration']
    interval = model_conf['interval']
    run_mode = model_conf['run_mode']

    graph = tf.Graph()
    with graph.as_default():

        layers = dict()
        outputs = dict()
        placeholder = dict()

        inputs_placeholder = placeholder_mapping(model_conf['inputs_placeholder'])
        placeholder['inputs_placeholder'] = inputs_placeholder
        # try:
        targets_placeholder = placeholder_mapping(model_conf['targets_placeholder'])
        placeholder['targets_placeholder'] = targets_placeholder
        # except:
        #     placeholder['targets_placeholder'] = None
        #     targets_placeholder = inputs_placeholder
        #     logging_io.DEBUG_INFO('Runing in autoencoder mode!')
        outputs['inputs_placeholder'] = inputs_placeholder

        for key in model_conf['layers'].keys():
            # print(key)
            layers[key] = layers_mapping(key, model_conf['layers'][key])
            logging_io.BUILD_INFO(layers[key])
        for layer_name, layer in layers.items():
            # print(layer_name)
            outputs[layer_name] = layer.outputs(outputs[model_conf['layers'][layer_name]['inputs']])

        logging_io.WARNING_INFO(outputs['outputs'])
        logging_io.WARNING_INFO(targets_placeholder)

        loss = loss_mean(outputs['outputs'], targets_placeholder, model_conf)
        logging_io.BUILD_INFO(loss)
        optimizer = optimizer_mapping(model_conf['optimizer']).minimize(loss)
        logging_io.BUILD_INFO(optimizer)
        accuracy = acc_sum(outputs["outputs"], targets_placeholder)
        logging_io.BUILD_INFO(accuracy)

        if run_mode == 'train':
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                errs = 0
                accs = 0
                for batch_train_inputs, batch_train_targets in train_provider:
                    _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = {inputs_placeholder:batch_train_inputs, targets_placeholder:batch_train_targets})
                    errs += err
                    accs += acc
                log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
                log['train']['acc'].append(round(accs/train_provider.n_samples(), round_val))
                logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5} Training ACC: {2:5}'.format(i, log['train']['err'][-1], log['train']['acc'][-1]))
                if (i+1) % interval == 0:
                    errs = 0
                    accs = 0
                    for batch_valid_inputs, batch_valid_targets in valid_provider:
                        err, acc = sess.run([loss, accuracy], feed_dict = {inputs_placeholder:batch_valid_inputs, targets_placeholder:batch_valid_targets})
                        errs += err
                        accs += acc
                    log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
                    log['valid']['acc'].append(round(accs/valid_provider.n_samples(), round_val))
                    logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5} validation ACC: {2:5}'.format(i, log['valid']['err'][-1], log['valid']['acc'][-1]))
        elif run_mode == 'valid':
            errs = 0
            accs = 0
            for batch_train_inputs, batch_train_targets in train_provider:
                err, acc = sess.run([loss, accuracy], feed_dict = {inputs_placeholder:batch_train_inputs, targets_placeholder:batch_train_targets})
                errs += err
                accs += acc
            log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
            log['train']['acc'].append(round(accs/train_provider.n_samples(), round_val))
            logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5} Training ACC: {2:5}'.format(i, log['train']['err'][-1], log['train']['acc'][-1]))
            errs = 0
            accs = 0
            for batch_valid_inputs, batch_valid_targets in valid_provider:
                err, acc = sess.run([loss, accuracy], feed_dict = {inputs_placeholder:batch_valid_inputs, targets_placeholder:batch_valid_targets})
                errs += err
                accs += acc
            log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
            log['valid']['acc'].append(round(accs/valid_provider.n_samples(), round_val))
            logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5} validation ACC: {2:5}'.format(i, log['valid']['err'][-1], log['valid']['acc'][-1]))

        if saveLOG == True:
            logging_io.LOG_COLLECTOR('LOGS')
    return log

def autoencoder_builder(data_conf_dir, model_conf_dir, data_provider, round_val=5, saveLOG=True):

    data_conf = json.load(open(data_conf_dir, 'r'))
    model_conf = json.load(open(model_conf_dir, 'r'))
    graph_assertion(model_conf)
    train_provider = data_provider(data_conf['training_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    valid_provider = data_provider(data_conf['validation_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    assert(isinstance(train_provider, dataProvider))
    assert(isinstance(valid_provider, dataProvider))
    log = {
        'train':{
            # 'acc':[],
            'err':[]
        },
        'valid':{
            # 'acc':[],
            'err':[]
        },
        'test':{
            # 'acc':[],
            'err':[]
        }
    }
    iteration = model_conf['iteration']
    interval = model_conf['interval']
    run_mode = model_conf['run_mode']

    graph = tf.Graph()
    with graph.as_default():

        layers = dict()
        outputs = dict()
        placeholder = dict()

        inputs_placeholder = placeholder_mapping(model_conf['inputs_placeholder'])
        placeholder['inputs_placeholder'] = inputs_placeholder
        # try:
        #     targets_placeholder = placeholder_mapping(model_conf['targets_placeholder'])
        #     placeholder['targets_placeholder'] = targets_placeholder
        # except:
        #     placeholder['targets_placeholder'] = None
        #     targets_placeholder = inputs_placeholder
        logging_io.DEBUG_INFO('Runing in autoencoder mode!')
        outputs['inputs_placeholder'] = inputs_placeholder

        for key in model_conf['layers'].keys():
            # print(key)
            layers[key] = layers_mapping(key, model_conf['layers'][key])
            logging_io.BUILD_INFO(layers[key])
        for layer_name, layer in layers.items():
            # print(layer_name)
            outputs[layer_name] = layer.outputs(outputs[model_conf['layers'][layer_name]['inputs']])

        logging_io.WARNING_INFO(outputs['outputs'])
        # logging_io.WARNING_INFO(targets_placeholder)

        loss = loss_mean(outputs['outputs'], inputs_placeholder, model_conf)
        logging_io.BUILD_INFO(loss)
        optimizer = optimizer_mapping(model_conf['optimizer']).minimize(loss)
        logging_io.BUILD_INFO(optimizer)
        # accuracy = acc_sum(outputs["outputs"], inputs_placeholder)
        # logging_io.BUILD_INFO(accuracy)

        if run_mode == 'train':
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                errs = 0
                # accs = 0
                for batch_train_inputs, batch_train_targets in train_provider:
                    _, err = sess.run([optimizer, loss], feed_dict = {inputs_placeholder:batch_train_inputs})
                    errs += err
                    # accs += acc
                log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
                # log['train']['acc'].append(round(accs/train_provider.n_samples(), round_val))
                logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5}'.format(i, log['train']['err'][-1]))
                if (i+1) % interval == 0:
                    errs = 0
                    # accs = 0
                    for batch_valid_inputs, batch_valid_targets in valid_provider:
                        err = sess.run([loss], feed_dict = {inputs_placeholder:batch_valid_inputs})
                        errs += err[0]
                        # accs += acc
                    log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
                    # log['valid']['acc'].append(round(accs/valid_provider.n_samples(), round_val))
                    logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5}'.format(i, log['valid']['err'][-1]))
        elif run_mode == 'valid':
            errs = 0
            # accs = 0
            for batch_train_inputs, batch_train_targets in train_provider:
                err = sess.run([loss], feed_dict = {inputs_placeholder:batch_train_inputs})
                errs += err
                # accs += acc
            log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
            # log['train']['acc'].append(round(accs/train_provider.n_samples(), round_val))
            logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5}'.format(i, log['train']['err'][-1]))
            errs = 0
            # accs = 0
            for batch_valid_inputs, batch_valid_targets in valid_provider:
                err = sess.run([loss], feed_dict = {inputs_placeholder:batch_valid_inputs})
                errs += err
                # accs += acc
            log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
            # log['valid']['acc'].append(round(accs/valid_provider.n_samples(), round_val))
            logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5}'.format(i, log['valid']['err'][-1]))

        if saveLOG == True:
            logging_io.LOG_COLLECTOR('LOGS')
    return log
