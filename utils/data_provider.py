import numpy as np
import pickle
from cmd_io import *

class dataProvider(object):

    def __init__(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def _shuffle(self):
        raise NotImplementedError

    def one_hot(self):
        raise NotImplementedError


class MNISTProvider(dataProvider):

    def __init__(self, filename, batch_size, isShuffle = True, isOneHot = False):
        self.inputs = np.load(open(filename, 'rb'))['inputs']
        self.targets = np.load(open(filename, 'rb'))['targets']
        self._filename = filename
        self._current_index = 0
        self._batch_size = batch_size
        self._n_samples = len(self.targets)
        self.isShuffle = isShuffle
        self.isOneHot = isOneHot
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
            return batch_inputs, batch_targets

    def one_hot(self):
        self._n_classes = len(list(set(self.targets)))
        one_hot_res = np.zeros((self._n_samples, self._n_classes))
        for target, i in zip(self.targets, range(self._n_samples)):
            one_hot_res[i][int(target)] = 1.0
        self.targets = one_hot_res

    def __str__(self):
        return cmd_print(0, ('\n{0:40}\n|{1:^38}|\n{0:40}\n'+
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
                                        self.isOneHot), False)

if __name__ == '__main__':

    provider = MNISTProvider('../data/mnist-train.npz', 50)
    # provider.shuffle()
    # print(provider.inputs.shape)
    # for train, valid in provider:
    #     print(train.shape, valid.shape)
    print(provider)
