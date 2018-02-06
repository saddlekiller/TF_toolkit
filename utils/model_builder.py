import json
import tensorflow as tf
from config_mapping import *
from tools import *
from data_provider import *
# from cmd_io import *
from logging_io import *
import shutil
import os

def basic_builder(data_conf_dir, model_conf_dir, data_provider, round_val=5, saveLOG=True, needSummary = False, isAutoEncoder = False):

    data_conf = json.load(open(data_conf_dir, 'r'))
    model_conf = json.load(open(model_conf_dir, 'r'))
    graph_assertion(model_conf['layers'])
    # savemode_assertion(model_conf['save_mode'])
    # save_mode = model_conf['save_mode']
    train_provider = data_provider(data_conf['training_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']), isAutoEncoder = isAutoEncoder)
    valid_provider = data_provider(data_conf['validation_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']), isAutoEncoder = isAutoEncoder)
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
    bestModels_id = []
    iteration = model_conf['iteration']
    interval = model_conf['interval']
    run_mode_assertion(model_conf['run_mode'])
    run_mode = model_conf['run_mode']

    if run_mode == 'train':
        try:
            os.mkdir('models')
            os.mkdir('nice_models')
            logging_io.SUCCESS_INFO('Directory:[models] and [nice_models] have been created succesfully!')
            logging_io.DEBUG_INFO('Running in training mode!')
        except:
            logging_io.WARNING_INFO('Directories have already been exists, please make sure whether they should be overwritten!')
            return
    else:
        logging_io.DEBUG_INFO('Running in validation mode!')

    graph = tf.Graph()
    with graph.as_default():
        layers = dict()
        outputs = dict()
        placeholder = dict()
        inputs_placeholder = placeholder_mapping(model_conf['inputs_placeholder'])
        placeholder['inputs_placeholder'] = inputs_placeholder
        targets_placeholder = placeholder_mapping(model_conf['targets_placeholder'])
        placeholder['targets_placeholder'] = targets_placeholder
        outputs['inputs_placeholder'] = inputs_placeholder

        for key in model_conf['layers'].keys():
            layers[key] = layers_mapping(key, model_conf['layers'][key])
            logging_io.BUILD_INFO(layers[key])
        for layer_name, layer in layers.items():
            outputs[layer_name] = layer.outputs(outputs[model_conf['layers'][layer_name]['inputs']])

        # logging_io.WARNING_INFO(outputs['outputs'])
        # logging_io.WARNING_INFO(targets_placeholder)
        logging_io.WARNING_INFO('SUMMARY ASSERTION BEGIN')
        summary_assertion(model_conf, needSummary)
        logging_io.WARNING_INFO('SUMMARY ASSERTION END')

        loss = loss_mean(outputs['outputs'], targets_placeholder, model_conf)
        logging_io.BUILD_INFO(loss)
        optimizer = optimizer_mapping(model_conf['optimizer']).minimize(loss)
        logging_io.BUILD_INFO(optimizer)
        accuracy = acc_sum(outputs["outputs"], targets_placeholder)
        logging_io.BUILD_INFO(accuracy)
        sess = tf.Session()
        if needSummary:
            tf.summary.scalar('mean_loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            merged_all = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('tensorboard/train', sess.graph)
            valid_writer = tf.summary.FileWriter('tensorboard/valid')
        if run_mode == 'train':
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
        else:
            saver = tf.train.Saver()
            model_id = pickle.load(open('model_ids.npz', 'rb'))
            saver.restore(sess, 'nice_models/model.ckpt-' + str(model_id[-1]))
            iteration = 1

        for i in range(iteration):
            current_id = i + 1
            errs = 0
            accs = 0
            for batch_train_inputs, batch_train_targets in train_provider:
                if run_mode == 'train':
                    if needSummary:
                        _, err, acc, merge = sess.run([optimizer, loss, accuracy, merged_all], feed_dict = {inputs_placeholder:batch_train_inputs, targets_placeholder:batch_train_targets})
                    else:
                        _, err, acc = sess.run([optimizer, loss, accuracy], feed_dict = {inputs_placeholder:batch_train_inputs, targets_placeholder:batch_train_targets})
                else:
                    if needSummary:
                        err, acc, merge = sess.run([loss, accuracy, merged_all], feed_dict = {inputs_placeholder:batch_train_inputs, targets_placeholder:batch_train_targets})
                    else:
                        err, acc = sess.run([loss, accuracy], feed_dict = {inputs_placeholder:batch_train_inputs, targets_placeholder:batch_train_targets})
                errs += err
                accs += acc
            log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
            log['train']['acc'].append(round(accs/train_provider.n_samples(), round_val))
            if needSummary:
                train_writer.add_summary(merge, i)
            logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5} Training ACC: {2:5}'.format(current_id, log['train']['err'][-1], log['train']['acc'][-1]))
            if run_mode == 'train':
                save_path = saver.save(sess, 'models/' + 'model.ckpt', global_step=current_id)
                logging_io.DEBUG_INFO('Models have been saved in {0}'.format(save_path))

            if (current_id) % interval == 0 or run_mode == 'valid':
                errs = 0
                accs = 0
                for batch_valid_inputs, batch_valid_targets in valid_provider:
                    if needSummary:
                        err, acc, merge = sess.run([loss, accuracy, merged_all], feed_dict = {inputs_placeholder:batch_valid_inputs, targets_placeholder:batch_valid_targets})
                    else:
                        err, acc = sess.run([loss, accuracy], feed_dict = {inputs_placeholder:batch_valid_inputs, targets_placeholder:batch_valid_targets})
                    errs += err
                    accs += acc
                log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
                log['valid']['acc'].append(round(accs/valid_provider.n_samples(), round_val))
                if run_mode == 'train':
                    if log['valid']['acc'][-1] >= np.max(log['valid']['acc']):
                        bestModels_id.append(current_id)
                        shutil.copy('models/model.ckpt-'+str(current_id), 'nice_models/model.ckpt-'+str(current_id))
                        shutil.copy('models/model.ckpt-'+str(current_id) + '.meta', 'nice_models/model.ckpt-'+str(current_id) + '.meta')
                if needSummary:
                    valid_writer.add_summary(merge, i)
                logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5} validation ACC: {2:5}'.format(current_id, log['valid']['err'][-1], log['valid']['acc'][-1]))
            logging_io.LOG_COLLECTOR('LOGS')
    if run_mode == 'train':
        pickle.dump(bestModels_id, open('model_ids.npz', 'wb'))
    return log

def autoencoder_builder(data_conf_dir, model_conf_dir, data_provider, round_val=5, saveLOG=True):

    data_conf = json.load(open(data_conf_dir, 'r'))
    model_conf = json.load(open(model_conf_dir, 'r'))
    graph_assertion(model_conf['layers'])
    train_provider = data_provider(data_conf['training_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    valid_provider = data_provider(data_conf['validation_filename'], data_conf['batch_size'], isShuffle = bool(data_conf['shuffle']))
    assert(isinstance(train_provider, dataProvider))
    assert(isinstance(valid_provider, dataProvider))
    log = {
        'train':{
            'err':[]
        },
        'valid':{
            'err':[]
        },
        'test':{
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
        logging_io.DEBUG_INFO('Runing in autoencoder mode!')
        outputs['inputs_placeholder'] = inputs_placeholder

        for key in model_conf['layers'].keys():
            layers[key] = layers_mapping(key, model_conf['layers'][key])
            logging_io.BUILD_INFO(layers[key])
        for layer_name, layer in layers.items():
            outputs[layer_name] = layer.outputs(outputs[model_conf['layers'][layer_name]['inputs']])

        loss = loss_mean(outputs['outputs'], inputs_placeholder, model_conf)
        logging_io.BUILD_INFO(loss)
        optimizer = optimizer_mapping(model_conf['optimizer']).minimize(loss)
        logging_io.BUILD_INFO(optimizer)

        if run_mode == 'train':
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                errs = 0
                # accs = 0
                for batch_train_inputs, batch_train_targets in train_provider:
                    _, err = sess.run([optimizer, loss], feed_dict = {inputs_placeholder:batch_train_inputs})
                    errs += err
                log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
                logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5}'.format(i, log['train']['err'][-1]))
                if (i+1) % interval == 0:
                    errs = 0
                    for batch_valid_inputs, batch_valid_targets in valid_provider:
                        err = sess.run([loss], feed_dict = {inputs_placeholder:batch_valid_inputs})
                        errs += err[0]
                    log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
                    logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5}'.format(i, log['valid']['err'][-1]))
        elif run_mode == 'valid':
            errs = 0
            for batch_train_inputs, batch_train_targets in train_provider:
                err = sess.run([loss], feed_dict = {inputs_placeholder:batch_train_inputs})
                errs += err
            log['train']['err'].append(round(errs/train_provider.n_batches(), round_val))
            logging_io.RESULT_INFO('Epoch {0:5} Training LOSS: {1:5}'.format(i, log['train']['err'][-1]))
            errs = 0
            for batch_valid_inputs, batch_valid_targets in valid_provider:
                err = sess.run([loss], feed_dict = {inputs_placeholder:batch_valid_inputs})
                errs += err
            log['valid']['err'].append(round(errs/valid_provider.n_batches(), round_val))
            logging_io.RESULT_INFO('Epoch {0:5} validation LOSS: {1:5}'.format(i, log['valid']['err'][-1]))

        if saveLOG == True:
            logging_io.LOG_COLLECTOR('LOGS')
    return log
