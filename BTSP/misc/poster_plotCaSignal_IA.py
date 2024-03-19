import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import matplotlib
from matplotlib import pyplot as plt


def preprocess_ImageAnal(F):
    hist = np.histogram(F, bins=100)
    max_index = np.where(hist[0] == max(hist[0]))[0][0]
    baseline = hist[1][max_index]
    return (F - baseline) / baseline

area = "CA1"
base_folder = rf"D:\special\all_{area}_tau0.8_gaspar\data\KS028_imaging\KS028_103121"
#area = "CA3"
#base_folder = "D:\\special\\all_CA3_tau0.8_gaspar\\data\\srb270_imaging\\srb270_230118\\"

output_folder = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\statistics\CA3_filterDisengaged"

fluo = np.load(f"{base_folder}/F.npy")
fpil = np.load(f"{base_folder}/Fneu.npy")
iscell = np.load(f"{base_folder}/iscell.npy")

neuron_index = np.nonzero(iscell[:, 0])[0]
F = fluo[neuron_index, :]

#cellids_selected = [14, 15, 17, 19]  # CA3
cellids_selected = [62, 15, 47, 9]  # CA1

#k=11
#cellids_selected = range(k*4,(k+1)*4)
#print(list(cellids_selected))

t = np.arange(len(fluo[0, :]))/30
T = 470  # s

scale = 2
#plt.figure(figsize=(scale*6, scale*1))
fig, ax = plt.subplots(1,1, figsize=(scale*6, scale*1), sharey=True)
colors = ["#4fc5cf", "#918f8f", "#db5db6", "#874a44", "#4fc5cf", "#918f8f"]

y_scale_factor = 1
for i, cellid in enumerate(cellids_selected):
    Fc = preprocess_ImageAnal(fluo[cellid,:])
    ax.plot(t[:T*30], y_scale_factor*Fc[:T*30]-i*3, c=colors[i], linewidth=1.5)
    plt.plot([t[T*30+200],t[T*30+200]], [-5, -5+y_scale_factor], c="k")  # scale bar

plt.gca().set_axis_off()
plt.tight_layout()
plt.savefig(f"{output_folder}/{area}_calcium_selected.svg", transparent=True)
plt.savefig(f"{output_folder}/{area}_calcium_selected.pdf", transparent=True)
plt.show()
