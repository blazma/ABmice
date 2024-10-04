import numpy as np
from matplotlib import pyplot as plt

states = [0, 1, 2]  # silent, fires, fires more

transition = np.array([[0.5,  0,    0.5],
                       [0,    0.3,  0.7],
                       [0.15, 0.4,  0.6]])

emission = np.array([[0.9, 0.1,   0],
                     [0.5, 0.4, 0.1],
                     [0.1, 0.4, 0.5]])

rng = np.random.default_rng(1234)
N = 100
data = np.zeros(N)
data[30:] = rng.integers(1, 3, size=N-30)

prior = np.array([1/3, 1/3, 1/3]) #emission[int(data[0])]
posterior = np.zeros((3,N))
posterior[:,0] = prior
for n in range(1,N):
    posterior[:,n] = np.multiply(emission[int(data[n])].T, transition @ posterior[:,n-1])
    posterior[:,n] = posterior[:,n] / np.sum(posterior[:,n])

plt.plot(data, c="red")
plt.imshow(posterior)
plt.show()