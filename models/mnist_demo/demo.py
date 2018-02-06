import json
import tensorflow as tf
import sys
sys.path.append('../../utils')
from config_mapping import *
from tools import *
from data_provider import *
from cmd_io import *
from model_builder import *

data_conf_dir = 'data.conf'
model_conf_dir = 'model.conf'
data_provider = MNISTProvider
log = basic_builder(data_conf_dir, model_conf_dir, data_provider, needSummary=True)
