import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
import tqdm

rng = np.random.default_rng(1234)

n_laps = 50
n_bins = 75


def generate_pf(rate_mean, lb, ub):
    #spike_matrix = np.zeros((n_bins, n_laps))
    #for lap in range(n_laps):
    #    spikes = scipy.stats.poisson.rvs(rate_mean, size=ub-lb, random_state=rng)
    #    spikes = scipy.ndimage.gaussian_filter1d(spikes, sigma=0.1)
    #    spike_matrix[lb:ub, lap] = spikes

    spike_vector = np.zeros(n_bins)
    spikes = scipy.stats.poisson.rvs(rate_mean, size=ub-lb, random_state=rng)
    spike_vector[lb:ub] = spikes
    return spike_vector

n_pfs = 1000
rate_mean = 2
formation_bins = []

for i_pf in tqdm.tqdm(range(n_pfs)):
    pf_width = rng.integers(3,15)
    lb = rng.integers(0,75-pf_width)
    ub = rng.integers(lb,lb+pf_width)
    if ub > 75:
        ub = 75
    spikes_fl = generate_pf(rate_mean, lb, ub)
    if spikes_fl[lb:ub].sum() == 0:
        continue

    bins_range = np.arange(0, ub-lb)
    formation_bin = lb + np.average(bins_range, weights=spikes_fl[lb:ub])
    formation_bins.append(formation_bin)
formation_bins = np.array(formation_bins)
sns.histplot(data=formation_bins, binwidth=1)
plt.xlim([0,n_bins])
plt.show()
#plt.imshow(spike_matrix.T)
#plt.show()