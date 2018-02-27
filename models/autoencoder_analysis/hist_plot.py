import matplotlib.pyplot as plt
import numpy as np
import pickle

features = pickle.load(open('features.npz', 'rb'))

# data = features
# f = plt.figure()
# for i in range(5):
#     for j in range(2):
#         ax = f.add_subplot(5, 2, i*2+j+1)
#         legends = []
#         for key in data.keys():
#             ax.hist(np.log(data[key][:, i*2+j]), bins=100)
#             legends.append(str(key))
#         plt.legend(legends)
# plt.show()

# f = plt.figure()
# for key,i in zip(data.keys(), range(10)):
#     ax = f.add_subplot(5,2, i + 1)
#     temp = data[key].reshape(-1,)
#     ax.hist(temp, bins = 100, normed = True)
#     plt.title(str(key))
#     plt.ylim([0,4])
# plt.show()

def Gau(x, mu, sigma):
    return 1./(np.sqrt(2*np.pi*sigma))*np.exp(-0.5*(x-mu)*(x-mu)/sigma)

k = 5
params = dict()
for l in range(10):
    params[l] = dict()
    params[l]['pis'] = []
    params[l]['mus'] = []
    params[l]['sigmas'] = []
    for j in range(len(features[l][0])):
        data = features[l][:, j]
        mu = np.linspace(0, 1, k)
        sigma = np.array([1]*k)
        pik = np.array([1]*k)/k
        N = len(data)
        for i in range(500):
            vs = []
            for ki in range(k):
                vs.append(pik[ki] * Gau(data, mu[ki], sigma[ki]))
            vs = np.array(vs)
            rkn = vs / np.sum(vs,0)
            rk = np.sum(rkn, 1)
            pik = rk / N
            mu = 1. / rk * rkn.dot(data)
            sigma = 1. / rk * rkn.dot(data*data) - mu*mu
        # print(params[i])
        params[l]['pis'].append(pik)
        params[l]['mus'].append(mu)
        params[l]['sigmas'].append(sigma)

f = plt.figure()
for i in range(len(features[0][0])):
    ax = f.add_subplot(4, 4, i+1)
    xs = np.linspace(0, 1, 5000)
    for j in range(10):
        ys = [0]*5000
        for ki in range(k):
            ys += params[j]['pis'][i][ki] * (1./(np.sqrt(2*np.pi*params[j]['sigmas'][i][ki]))*np.exp(-0.5*(xs-params[j]['mus'][i][ki])**2/params[j]['sigmas'][i][ki]))
        ax.plot(xs, ys)
pickle.dump(params, open('params.npz', 'wb'))
# f = plt.figure()
# ax = f.add_subplot(211)
# ax.hist(data, bins=100)
# plt.xlim([0,1])
# ax = f.add_subplot(212)
# xs = np.linspace(0, 1, 5000)
# ys = [0]*5000
# for i in range(k):
#     ys += pik[i] * (1./(np.sqrt(2*np.pi*sigma[i]))*np.exp(-0.5*(xs-mu[i])**2/sigma[i]))
# ax.plot(xs, ys)
# plt.xlim([0,1])
plt.show()
