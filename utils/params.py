import tensorflow as tf
from cmd_io import *

class variables(object):
    
    def __init__(self, name, shape, initializer, dtype=tf.float32):
        self.param = tf.get_variable(name=name, initializer=initializer(shape), dtype=dtype)
    
    def get(self):
        return self.param
    
    def __str__(self, classname):
        return cmd_print(0, '<' + classname + '> ' + str(self.param))    
        
class truncated_normal(variables):

    def __init__(self, name, shape, dtype=tf.float32):
        super(truncated_normal, self).__init__(name, shape, tf.truncated_normal, dtype)
        
    def get(self):
        return super(truncated_normal, self).get()
    
    def __str__(self):
        return super(truncated_normal, self).__str__('variables.truncated_normal')

class random_normal(variables):

    def __init__(self, name, shape, dtype=tf.float32):
        super(random_normal, self).__init__(name, shape, tf.random_normal, dtype)
        
    def get(self):
        return super(random_normal, self).get()
    
    def __str__(self):
        return super(random_normal, self).__str__('variables.random_normal')
        
class random_uniform(variables):

    def __init__(self, name, shape, dtype=tf.float32):
        super(random_uniform, self).__init__(name, shape, tf.random_uniform, dtype)
        
    def get(self):
        return super(random_uniform, self).get()
    
    def __str__(self):
        return super(random_uniform, self).__str__('variables.random_uniform')
        

        







