import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt

n = 1000
rng = np.random.default_rng(1234)
data = rng.normal(loc=0, scale=1.0, size=n)

pos = np.linspace(min(data), max(data), n)
kde = scipy.stats.gaussian_kde(data)

fig, ax = plt.subplots()
ax.hist(data, bins=np.arange(-5,5,0.1), density=True)
ax.plot(pos, kde(pos))
plt.show()
