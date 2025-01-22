import numpy as np
import scipy
from matplotlib import pyplot as plt


class ModelSynthetic:
    def __init__(self, seed=1234, window_size=10):
        self.rng = np.random.default_rng(seed)
        self.data = np.nan
        self.window_size = window_size

    def generate_data(self, mean_spks, n_laps, big_event_lap):
        data = self.rng.poisson(mean_spks,n_laps)
        data_high = self.rng.poisson(mean_spks*5, n_laps)
        data[big_event_lap] = mean_spks*10
        data[big_event_lap+1:] = data_high[big_event_lap+1:]
        self.data = data

    def null_model(self):
        pass

    def plot_data(self):
        plt.plot(self.data)
        plt.show()

model = ModelSynthetic()
model.generate_data(4, 200, 100)
model.plot_data()
