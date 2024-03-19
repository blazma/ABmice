import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import matplotlib
from matplotlib import pyplot as plt


def preprocess(F, win_baseline, sig_baseline, fs):
## https://suite2p.readthedocs.io/en/latest/api/suite2p.extraction.html

    win = int(win_baseline*fs)
    Flow = gaussian_filter(F,    [0., sig_baseline])
    Flow = minimum_filter1d(Flow,    win)
    Flow = maximum_filter1d(Flow,    win)
    F = (F - Flow)/(Flow)
    return F

def preprocess_ImageAnal(F):
    hist = np.histogram(F, bins=100)
    max_index = np.where(hist[0] == max(hist[0]))[0][0]
    baseline = hist[1][max_index]
    return (F - baseline) / baseline

#area = "CA1"
#base_folder = rf"D:\special\all_{area}_tau0.8_gaspar\data\KS028_imaging\KS028_103121"
area = "CA3"
base_folder = "D:\\special\\all_CA3_tau0.8_gaspar\\data\\srb270_imaging\\srb270_230118\\"

output_folder = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\statistics\CA3_filterDisengaged"

fluo = np.load(f"{base_folder}/F.npy")
fpil = np.load(f"{base_folder}/Fneu.npy")
iscell = np.load(f"{base_folder}/iscell.npy")
#F = fluo-(0.7*fpil)  # subtract neuropil

F = fluo
neuron_index = np.nonzero(iscell[:, 0])[0]
F = F[neuron_index, :]
#Fc = preprocess(F=F, win_baseline=60, sig_baseline=10, fs=30)  # subtract baseline

# set(self.pfs_df[(self.pfs_df["category"] == "btsp") | (self.pfs_df["category"] == "early")]["cell id"].values)
#cellids = [0, 1, 2, 3, 4, 5, ]
#cellids2 = [6, 7, 8, 9, 10, 11, ]
#cellids3 = [14, 15, 16, 17, 18, 19, ]
#cellids4 = [20, 22, 24, 26, 28, 29, ] #32, 34, 35, 36, 37, 40, 44, 45, 47, 48, 49, 50, 53, 60, 62, 63, 64, 65, 68, 70, 72, 74, 75, 84, 85, 87, 88, 90, 93, 96, 99, 104, 107, 117, 129, 155, 165, 171]

cellids_selected = [14, 15, 17, 19]  # CA3
t = np.arange(len(F[0, :]))/30
T = 470  # s

scale = 2
#plt.figure(figsize=(scale*6, scale*1))
fig, ax = plt.subplots(1,1, figsize=(scale*6, scale*1), sharey=True)
colors = ["#4fc5cf", "#918f8f", "#db5db6", "#874a44", "#4fc5cf", "#918f8f"]

y_scale_factor = 1
for i, cellid in enumerate(cellids_selected):
    #plt.plot(t[:T*30], y_scale_factor*Fc[cellid,:T*30]-i*5, c=colors[i], linewidth=0.8)
    #Fc = preprocess(F=F, win_baseline=60, sig_baseline=10, fs=30)
    #axs[0].plot(t[:T*30], y_scale_factor*Fc[cellid,:T*30], c=colors[i], linewidth=0.8)
    #axs[0].set_ylabel("preprocess")

    Fc = preprocess_ImageAnal(F)
    ax.plot(t[:T*30], y_scale_factor*Fc[cellid,:T*30], c=colors[i], linewidth=0.8)
    ax.set_ylabel("ImageAnal")
    #plt.plot([t[T*30+200],t[T*30+200]], [-7, -7+y_scale_factor], c="k")  # scale bar
#plt.gca().set_axis_off()
plt.savefig(f"{output_folder}/{area}_calcium_selected.svg", transparent=True)
plt.savefig(f"{output_folder}/{area}_calcium_selected.pdf", transparent=True)
plt.show()
