import json

class config(object):
    
    def __init__(self):
        raise NotImplementedError
        
    def __str__(self):
        raise NotImplementedError
        
        
        
class dataConfig():
    
    def __init__(self, filename):
        self.filename = filename
        self.config = json.load(open(self.filename, 'r'))
        
    def __str__(self):
        print(self.config)
        
        
class modelConfig():
    
    def __init__(self):
        pass
    
    def __str__(self):
        pass
        
        