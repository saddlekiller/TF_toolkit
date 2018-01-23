import numpy as np
import pickle

class dataProvider(object):
    
    def __init__(self):
        raise NotImplementedError
        
    def next(self):
        raise NotImplementedError
        
    def __str__(self):
        raise NotImplementedError
        
        
    def _shuffle(self):
        raise NotImplementedError
        
class MNISTProvider(dataProvider):
    
    def __init__(self, filename, batch_size):
        self.data = np.load(open(filename, 'rb'))
        self._current_index = 0
        self._batch_size = batch_size
        self._n_samples = len(self.data['targets'])
        if self._n_samples % self._batch_size == 0:
            self._n_batches = int(self._n_samples/self._batch_size)
        else:
            self._n_batches = int(self._n_samples/self._batch_size)+1
        
    def _new_epoch(self):
        self._current_index = 0
        
    def shuffle(self):
        
        
    def next(self):
        raise NotImplementedError
    
if __name__ == '__main__':
    
    MNISTProvider('../data/mnist-train.npz')
    
    
    