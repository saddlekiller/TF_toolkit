import json

class config(object):

    def __init__(self, filename):
        self.filename = filename
        self.config = json.load(open(self.filename, 'r'))

    def get(self):
        return self.config

    def __str__(self):
        print(self.config)
        
