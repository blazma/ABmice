import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from matplotlib import pyplot as plt
import scipy


def preprocess(F, win_baseline, sig_baseline, fs):
## https://suite2p.readthedocs.io/en/latest/api/suite2p.extraction.html

    win = int(win_baseline*fs)
    Flow = gaussian_filter(F,    [0., sig_baseline])
    Flow = minimum_filter1d(Flow,    win)
    Flow = maximum_filter1d(Flow,    win)
    F = F - Flow
    return F

def calc_chunky_sum(y, chunk_size):
    N = len(y)
    #n_chunks = 10
    y_sum = np.zeros(N)
    n_chunks = N // chunk_size
    for i_chunk in range(n_chunks-1):
        chunk_start, chunk_end = [i_chunk*chunk_size, (i_chunk+1)*chunk_size]
        y_sum[chunk_start:chunk_end] = y[chunk_start:chunk_end].sum()
    return y_sum

base_folder_all_CA3 = "D:\\special\\all_CA3_tau0.8_gaspar\\data"
base_folders_all_CA3 = [
    f"{base_folder_all_CA3}\\srb231_imaging\\srb231_220804\\",
]

output_folder = r"C:\Users\martin\home\phd\btsp_project\analyses\other"

for base_folder in base_folders_all_CA3:
    spks_path = f"{base_folder}/spks_oasis_08s.npy"
    fluo_path = f"{base_folder}/F.npy"
    fpil_path = f"{base_folder}/Fneu.npy"
    spks = np.load(spks_path)
    fluo = np.load(fluo_path)
    fpil = np.load(fpil_path)
    F = fluo-(0.7*fpil)  # subtract neuropil
    Fc = preprocess(F=F, win_baseline=60, sig_baseline=10, fs=30)  # subtract baseline

    for cellid in range(10,20):
        print(cellid)

        ### convolve spks with exponential kernel of tau=0.8 s
        tau = 0.8  # s
        fs = 30 # frame / s
        tau_f = tau * fs  # frame

        n_frames = spks[cellid,:].shape[0]
        kernel = np.exp(-np.arange(n_frames) / tau_f)
        conv = scipy.signal.convolve(spks[cellid,:], kernel)

        ### plotting
        pmws = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        fig, axs = plt.subplots(len(pmws)+2, sharex=True, height_ratios=[len(pmws)+1]+[1]*(len(pmws)+1), figsize=(13,9), dpi=250)
        t = np.arange(len(Fc[cellid, :]))/30
        axs[0].plot(t, Fc[cellid,:], color="gold")
        axs[0].plot(t, conv[:n_frames], color="skyblue")
        axs[0].axhline(0, color="k", linestyle="--")
        axs[0].set_ylim([-200,1200])
        #Fc_filt_zeroed = Fc_filt-np.mean(Fc_filt)
        #Fc_filt_zeroed[Fc_filt_zeroed < 0] = 0
        #axs[0].plot(Fc_filt_zeroed, color="red")

        chunky_spks = calc_chunky_sum(spks[cellid,:], chunk_size=10)
        axs[1].stairs(chunky_spks[:-1], edges=t, label="tau0.8", color="orange", fill=True)
        axs[1].stairs(spks[cellid,:][:-1], edges=t, fill=True, color="red")
        axs[1].spines[['right', 'top']].set_visible(False)
        axs[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axs[1].spines[['bottom']].set_visible(False)
        axs[1].set_ylim([0, 500])

        for i, pmw in enumerate(pmws):
            spks_gaspar_path = f"{base_folder}/spks_pmw={pmw}.npy"
            spks_gaspar = np.load(spks_gaspar_path)
            chunky_spks = calc_chunky_sum(spks_gaspar[cellid, :], chunk_size=10)
            axs[i+2].stairs(chunky_spks[:-1], edges=t, color="blue", fill=True, label=f"pmw={pmw}")
            axs[i+2].stairs(spks_gaspar[cellid,:][:-1], edges=t, color="cyan", fill=True)
            axs[i+2].sharey(axs[1])
            axs[i+2].legend(loc="center left", bbox_to_anchor=(1,0.5))
            axs[i+2].spines[['right', 'top']].set_visible(False)
            if i != len(pmws):
                axs[i+2].spines[['bottom']].set_visible(False)

        plt.subplots_adjust(wspace=0, hspace=0.15)
        plt.xlim(0, 250)
        plt.savefig(f"{output_folder}/{cellid}.png", bbox_inches='tight')
        plt.close()
