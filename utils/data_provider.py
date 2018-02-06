import numpy as np
import pickle
# from cmd_io import *
from logging_io import *

class dataProvider(object):

    def __init__(self):
        raise NotImplementedError

    def n_batches(self):
        raise NotImplementedError

    def n_samples(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def _new_epoch(self):
        raise NotImplementedError

    def shuffle(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def next():
        return __next__(self)

    def __next__(self):
        raise NotImplementedError

    def one_hot(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MNISTProvider(dataProvider):

    def __init__(self, filename, batch_size, isShuffle = True, isOneHot = False, isAutoEncoder = False):
        self.inputs = np.load(open(filename, 'rb'))['inputs']
        self.targets = np.load(open(filename, 'rb'))['targets']
        self._filename = filename
        self._current_index = 0
        self._batch_size = batch_size
        self._n_samples = len(self.targets)
        self.isShuffle = isShuffle
        self.isOneHot = isOneHot
        self.isAutoEncoder = isAutoEncoder
        self._current_order = np.arange(self._n_samples)
        if self._n_samples % self._batch_size == 0:
            self._n_batches = int(self._n_samples/self._batch_size)
        else:
            self._n_batches = int(self._n_samples/self._batch_size)+1
        if self.isOneHot == False:
            self.one_hot()
        else:
            self._n_classes = len(self.targets[0])
        if self.isShuffle == True:
            self.shuffle()

    def n_batches(self):
        return self._n_batches

    def n_samples(self):
        return self._n_samples

    def reset(self):
        self._current_order = np.arange(self._n_samples)
        self._new_epoch()
        print('reset')

    def _new_epoch(self):
        self._current_index = 0
        if self.isShuffle == True:
            self.shuffle()
        # print('new_epoch')

    def shuffle(self):
        np.random.shuffle(self._current_order)

    def __iter__(self):
        return self

    def next():
        return __next__(self)

    def __next__(self):
        if self._current_index >= self._n_batches:
            self._new_epoch()
            raise StopIteration
        else:
            slide_index = self._current_order[self._current_index*self._batch_size:(self._current_index+1)*self._batch_size]
            batch_inputs = self.inputs[slide_index]
            batch_targets = self.targets[slide_index]
            self._current_index += 1
            if self.isAutoEncoder:
                return batch_inputs, batch_inputs
            return batch_inputs, batch_targets

    def one_hot(self):
        self._n_classes = len(list(set(self.targets)))
        one_hot_res = np.zeros((self._n_samples, self._n_classes))
        for target, i in zip(self.targets, range(self._n_samples)):
            one_hot_res[i][int(target)] = 1.0
        self.targets = one_hot_res

    def __str__(self):
        return logging_io.BUILD('\n{0:40}\n|{1:^38}|\n{0:40}\n'+
                                '| {2:17}| {3:17} |\n'+
                                '| {4:17}| {5:17} |\n'+
                                '| {6:17}| {7:17} |\n'+
                                '| {8:17}| {9:17} |\n'+
                                '| {10:17}| {11:17} |\n'+
                                '{0:40}\n'
                                ).format('+'+'-'*38+'+',
                                        'MNIST DATA PROVIDER',
                                        'n_samples',
                                        self._n_samples,
                                        'n_batch',
                                        self._n_batches,
                                        'batch_size',
                                        self._batch_size,
                                        'shuffle',
                                        self.isShuffle,
                                        'one_hot',
                                        self.isOneHot)


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


# if __name__ == '__main__':
#
#     provider = MNISTProvider('../data/mnist-train.npz', 50)
    # provider.shuffle()
    # print(provider.inputs.shape)
    # for train, valid in provider:
    # print(provider)
    #     print(train.shape, valid.shape)
