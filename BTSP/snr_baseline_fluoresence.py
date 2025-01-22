import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logging
from BTSP.TunedCellList import TunedCellList
from ImageAnal import *
from utils import grow_df
import seaborn as sns
import scipy
from copy import deepcopy


data_root = r"D:\\"
output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual"

def extract_SNR_and_baselines(area):
    data_path = rf"{data_root}\{area}"
    tcl = TunedCellList(area, data_path, output_path, extra_info="")

    SNR_and_Fs_df = None
    for animal in tcl.animals:
        sessions_all = tcl._read_meta(animal)

        for session in sessions_all:
            try:
                ISD = tcl._load_session(sessions_all, session)
                ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=12)
            except Exception:
                logging.exception(f"loading session {session} failed; skipping")
                continue
            if ISD == None:  # if failed to load ISD, skip
                continue

            n_cells = ISD.N_cells
            active_cells = ISD.active_cells
            tuned_cells = np.union1d(ISD.tuned_cells[0], ISD.tuned_cells[1])
            labels = np.array(["inactive"]*n_cells)
            labels[active_cells] = "untuned"
            labels[tuned_cells] = "tuned"

            len_recording = ISD.frame_times[-1] - ISD.frame_times[0]
            SNR_and_Fs_dict = {
                "area": [area]*n_cells,
                "animal id": [animal]*n_cells,
                "session id": [session]*n_cells,
                "baseline (dF/F)": ISD.cell_baselines,
                "baseline (F)": ISD.cell_baselines_F,
                "SNR": ISD.cell_SNR,
                "baseline SD": ISD.cell_baselines_F_SDs,
                "event rate": 60 * ISD.N_events / len_recording,
                "label": labels
            }
            SNR_and_Fs_df_sesh = pd.DataFrame.from_dict(SNR_and_Fs_dict)
            SNR_and_Fs_df = grow_df(SNR_and_Fs_df, SNR_and_Fs_df_sesh)
    SNR_and_Fs_df.to_pickle(f"{output_path}/statistics/misc/SNRs_baselines_df_{area}.pickle")

def plot_SNR_and_baseline_histograms():
    SNR_baselines_df_CA1 = pd.read_pickle(f"{output_path}/statistics/misc/SNRs_baselines_df_CA1.pickle")
    SNR_baselines_df_CA3 = pd.read_pickle(f"{output_path}/statistics/misc/SNRs_baselines_df_CA3.pickle")
    SNR_baselines_df = pd.concat([SNR_baselines_df_CA1, SNR_baselines_df_CA3]).reset_index(drop=True)
    SNR_baselines_df["log10(baseline dF/F)"] = np.log10(SNR_baselines_df["baseline (dF/F)"])
    #SNR_baselines_df["event rate"] = 60 * SNR_baselines_df["event rate"]  # convert to event / min

    df = deepcopy(SNR_baselines_df)

    ca1 = df[df["area"] == "CA1"]
    #ca1_filt = ca1.loc[(ca1["baseline (F)"] > ca1["baseline (F)"].quantile(0.1)) &
    #                   (ca1["baseline (F)"] < ca1["baseline (F)"].quantile(0.9))]

    ca3 = df[df["area"] == "CA3"]
    #ca3_filt = ca3.loc[(ca3["baseline (F)"] > ca3["baseline (F)"].quantile(0.1)) &
    #                   (ca3["baseline (F)"] < ca3["baseline (F)"].quantile(0.9))]

    res = scipy.stats.pearsonr(ca1["baseline (F)"],
                               ca1["event rate"])


    scipy.stats.pearsonr(ca3[~ca3["SNR"].isna()]["SNR"], ca3[~ca3["SNR"].isna()]["event rate"])


    plt.figure()
    sns.histplot(data=SNR_baselines_df, x="baseline (F)", hue="area")

    sns.jointplot(data=SNR_baselines_df, x="baseline (F)", y="event rate", hue="area",
                  marginal_kws={"cut": 0, "common_norm": False}, xlim=[0, 800], ylim=[0, 10])

    plt.figure()
    sns.histplot(data=SNR_baselines_df, x="baseline (dF/F)", hue="area", binrange=[-0.1, 0.2], binwidth=0.01)

    plt.figure()
    sns.kdeplot(data=SNR_baselines_df.reset_index(), x="log10(baseline dF/F)", hue="area", common_norm=False)
    #SNR_baselines_df.groupby("area")["baseline"].median()

    plt.figure()
    sns.histplot(data=SNR_baselines_df, x="SNR", hue="area")

    plt.figure()
    sns.kdeplot(data=SNR_baselines_df, x="SNR", hue="area", common_norm=False)

    sns.jointplot(data=SNR_baselines_df, x="log10(baseline dF/F)", y="SNR", hue="area", xlim=[-5, 1], ylim=[0, 200],
                  marginal_kws={"common_norm": False})

    ###########################################################
    ##### PLOT KDEs FOR DIFFERENT CELL ACTIVITY TYPES AND AREAS
    scale = 3
    fig, axs = plt.subplots(3, 2, sharex="col", sharey="col", figsize=(3 * scale, 2 * scale))
    sns.kdeplot(data=SNR_baselines_df[SNR_baselines_df["label"] == "inactive"], x="SNR", hue="area", common_norm=False, ax=axs[0, 0], cut=0)
    sns.kdeplot(data=SNR_baselines_df[SNR_baselines_df["label"] == "untuned"], x="SNR", hue="area", common_norm=False, ax=axs[1, 0], cut=0, legend=False)
    sns.kdeplot(data=SNR_baselines_df[SNR_baselines_df["label"] == "tuned"], x="SNR", hue="area", common_norm=False, ax=axs[2, 0], cut=0, legend=False)

    sns.kdeplot(data=SNR_baselines_df[SNR_baselines_df["label"] == "inactive"], x="log10(baseline dF/F)", hue="area", common_norm=False, ax=axs[0, 1], cut=0, legend=False)
    sns.kdeplot(data=SNR_baselines_df[SNR_baselines_df["label"] == "untuned"], x="log10(baseline dF/F)", hue="area", common_norm=False, ax=axs[1, 1], cut=0, legend=False)
    sns.kdeplot(data=SNR_baselines_df[SNR_baselines_df["label"] == "tuned"], x="log10(baseline dF/F)", hue="area", common_norm=False, ax=axs[2, 1], cut=0, legend=False)

    # set titles
    titles = ["inactive cells", "active, untuned cells", "active, tuned cells"]
    [[axs[i,j].set_title(titles[i]) for i in range(3)] for j in range(2)]

    # disable all ylabels
    [[axs[i, j].set_ylabel("") for i in range(3)] for j in range(2)]
    plt.tight_layout()
    plt.show()

    #############################################
    ############## TODO REFACTOR ALLLLLL OF THIS
    rng = np.random.default_rng(1234)

    ca1 = SNR_baselines_df[SNR_baselines_df["area"] == "CA1"]

    # fit gamma distribution on CA3 SNR values
    snr = ca3[~ca3["SNR"].isna()]["SNR"].values
    dist = scipy.stats.gamma
    bounds = {"a": [0, 100],
              "loc": [0, 100],
              "scale": [0, 100]}
    res = scipy.stats.fit(dist, snr, bounds)

    # sample from CA1 according to fitted distribution
    a, loc, scale = res.params.a, res.params.loc, res.params.scale
    snr_ca1 = ca1[~ca1["SNR"].isna()]["SNR"].values
    dist_fitted = scipy.stats.gamma.pdf(snr_ca1, a, loc, scale)
    p = dist_fitted / np.sum(dist_fitted)
    samples = rng.choice(snr_ca1, size=len(snr), p=p, replace=False)

    ca1_ss = ca1[ca1["SNR"].isin(samples)]

    """
    plt.figure()
    range = np.arange(0,100,1)
    plt.hist(ca1["SNR"].values, label="ca1_all", bins=range)
    plt.hist(snr, label="ca3", bins=range)
    plt.hist(ca1_ss["SNR"].values, label="ca1", bins=range)
    plt.scatter(snr, scipy.stats.gamma.pdf(snr, a, loc, scale)*len(snr), label="fit", color="k", s=2)
    plt.scatter(snr_ca1, scipy.stats.gamma.pdf(snr_ca1, a, loc, scale)*len(snr_ca1)/4, label="fit_ca1", color="r", s=2)
    plt.scatter(samples, scipy.stats.gamma.pdf(samples, a, loc, scale)*len(samples), label="fit_ca1_samples", color="cyan", s=2)
    plt.legend()
    """

    ca1_ss_ca3 = pd.concat([ca1_ss, ca3])
    # sns.jointplot(data=ca1_ss_ca3, x="SNR", y="event rate", hue="area", marginal_kws={"common_norm": False, "cut": 0}, xlim=[0,230], ylim=[0,10])

    fig, axs = plt.subplots()
    sns.kdeplot(data=ca1["event rate"], cut=0, label="ca1")
    sns.kdeplot(data=ca1_ss["event rate"], cut=0, label="ca1 subsampled")
    sns.kdeplot(data=ca1[ca1["SNR"] < 50]["event rate"], cut=0, label="ca1 thresholded")
    sns.kdeplot(data=ca3["event rate"], cut=0, label="ca3")
    plt.legend()



    ###################################
    #######################################
    fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    ca1 = SNR_baselines_df[SNR_baselines_df["area"] == "CA1"]
    ca3 = SNR_baselines_df[SNR_baselines_df["area"] == "CA3"]

    palette = {
        "KS028": "red",
        "KS029": "red",
        "KS030": "red",
        "srb131": "red",
        "srb231": "red",
        "srb251": "red",
        "srb402": "blue",
        "srb410": "red",
        "srb410a": "red",
        "srb363": "red",
        "srb377": "red",
        "srb270": "red",
        "srb269": "red",
    }

    sns.kdeplot(data=ca1, x="event rate", hue="animal id", common_norm=False, ax=axs[0])
    sns.kdeplot(data=ca3, x="event rate", hue="animal id", common_norm=False, ax=axs[1])

    # sns.kdeplot(data=ca1[(ca1["animal id"] != "srb402")], x="SNR", hue="animal id", common_norm=False, ax=axs[0], palette=palette, alpha=0.2)
    # sns.kdeplot(data=ca1[(ca1["animal id"] == "srb402")], x="SNR", hue="animal id", common_norm=False, ax=axs[0], palette=palette)
    # sns.kdeplot(data=ca3[(ca3["animal id"] != "srb402")], x="SNR", hue="animal id", common_norm=False, ax=axs[1], palette=palette, alpha=0.2)
    # sns.kdeplot(data=ca3[(ca3["animal id"] == "srb402")], x="SNR", hue="animal id", common_norm=False, ax=axs[1], palette=palette)
    # axs[0].set_xlim([0,150])

#extract_SNR_and_baselines("CA1")
#extract_SNR_and_baselines("CA3")
plot_SNR_and_baseline_histograms()
