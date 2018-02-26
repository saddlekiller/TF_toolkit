import matplotlib.pyplot as plt
import numpy as np
import pickle

data = pickle.load(open('features.npz', 'rb'))
f = plt.figure()
for i in range(5):
    for j in range(2):
        ax = f.add_subplot(5, 2, i*2+j+1)
        legends = []
        for key in data.keys():
            ax.hist(np.log(data[key][:, i*2+j]), bins=100)
            legends.append(str(key))
        plt.legend(legends)
plt.show()
