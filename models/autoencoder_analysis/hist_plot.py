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

# f = plt.figure()
# for key,i in zip(data.keys(), range(10)):
#     ax = f.add_subplot(5,2, i + 1)
#     temp = data[key].reshape(-1,)
#     ax.hist(temp, bins = 100, normed = True)
#     plt.title(str(key))
#     plt.ylim([0,4])
# plt.show()
