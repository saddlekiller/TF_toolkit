import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils')
from tools import *


dir_path = '/home/cheng/Downloads/images/'
all_data = []
max_n = 25
multiplier = 3
bias = 0
for i in range(0, max_n):
    path = dir_path + str(i*multiplier + bias) + '.png'
    data = plt.imread(path)[:,:,1]
    all_data.append(data)
print(path)
all_data = np.array(all_data).transpose([1, 2, 0])
images = build_image(all_data, int(np.sqrt(max_n)), 10)
plt.imshow(images, 'gray')
# plt.show()
plt.imsave('combine.png', images)
