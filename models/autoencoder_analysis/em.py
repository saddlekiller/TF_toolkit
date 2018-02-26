import numpy as np
import pickle
import matplotlib.pyplot as plt

def Gau(x, mu, sigma):
    # print(mu, sigma)
    # print(mu)
    # print(1./(np.sqrt(2*np.pi)*sigma))
    # print(np.exp(-0.5*(x-mu)*(x-mu)/sigma))
    # print(np.exp(-0.5*(x-mu)*(x-mu)/sigma))
    return 1./(np.sqrt(2*np.pi*sigma))*np.exp(-0.5*(x-mu)*(x-mu)/sigma)

# data = pickle.load(open('features.npz', 'rb'))
# temp = data[0][:,0]
# k = 3
# mus = np.random(k)
# sigmas = np.ones(k)
# pis = np.ones(k)/k
# N = len(temp)
# for i in range(10):
#     r = pis

k = 3
mu = np.array([-10, 0, 10])
sigma = np.array([9, 9 ,9])
pik = np.array([1/3, 1/3, 1/3])
n1 = 20000
n2 = 40000
n3 = 60000
real_mu = [-20, 0, 20]
real_sigma = [16, 4, 16]
real_pi = [1/6, 2/6, 3/6]
data = np.array(list(np.random.normal(real_mu[0], np.sqrt(real_sigma[0]), n1))+
        list(np.random.normal(real_mu[1], np.sqrt(real_sigma[1]), n2))+
        list(np.random.normal(real_mu[2], np.sqrt(real_sigma[2]), n3))).reshape(-1, )
N = len(data)

print(N)
mus = []
sigmas = []
for i in range(500):
    vs = []
    for ki in range(k):
        vs.append(pik[ki] * Gau(data, mu[ki], sigma[ki]))
    vs = np.array(vs)
    rkn = vs / np.sum(vs,0)
    # print(rkn.shape)
    rk = np.sum(rkn, 1)
    pik = rk / N

    mu = 1. / rk * rkn.dot(data)
    sigma = 1. / rk * rkn.dot(data*data) - mu*mu
    mus.append(mu)
    sigmas.append(sigma)

print(pik)
print(mu)
print(sigma)

plt.clf()
a,b,c = plt.hist(data, bins = 200)
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(b[1:], a/N, 'rx')
print(np.sum(a))

#
n = 50000
xs = np.linspace(-40, 50, n)
ysum = [0]*n
real_ysum = [0]*n

for i in range(k):
    ys = pik[i] * (1./(np.sqrt(2*np.pi*sigma[i]))*np.exp(-0.5*(xs-mu[i])**2/sigma[i]))
    ysum += ys

    real_ysum += real_pi[i]*(1./(np.sqrt(2*np.pi*real_sigma[i]))*np.exp(-0.5*(xs-real_mu[i])**2/real_sigma[i]))
    # ax.plot(xs, ys)
ax.plot(xs, ysum, 'g*')
ax.plot(xs, real_ysum, 'b')


# f1 = plt.figure()
# ax2 = f1.add_subplot(211)
# ax2.plot(np.array(mus)[:, 0])
# ax2 = f1.add_subplot(212)
# ax2.plot(np.array(sigmas)[:, 0])

plt.show()
