import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


imaging_rootDir = r"C:\Users\martin\home\phd\btsp_project\analyses\other\selected_CA3_for_FOV"
#imaging_rootDir = r"C:\Users\martin\home\phd\btsp_project\analyses\other\for_reshuffle\data"
pfData_rootDir = r""

sessions = ["srb231_220808",
            "srb251_221028_T1",
            "srb251_221122",
            "srb251_221125",
            "srb270_230117",
            "srb270_230118",
            "srb270_230120",
            "srb270_230127",
            "srb363_231207",
            "srb363_231211",
            "srb363_231215",]
animals = ["srb231",
           "srb251",
           "srb270",
           "srb363"]
for animal in animals:
    pfs = pd.read_pickle(rf"C:\Users\martin\home\phd\btsp_project\analyses\manual\place_fields\CA3\{animal}_place_fields_df.pickle")
    pfs["newly formed"] = np.where(pfs["category"].isin(["transient", "non-btsp", "btsp"]), True, False)

    # if shift criterion is not considered: reset all BTSP to non-BTSP PFs
    pfs.loc[pfs["category"] == "btsp", "category"] = "non-btsp"
    # then set those non-BTSP to BTSP who satisfy gain and drift only
    pfs.loc[(pfs["has high gain"] == True) & (pfs["has no drift"] == True), "category"] = "btsp"

    sessions_animal = [session for session in sessions if animal in session]
    for session in sessions_animal:
        ops = np.load(f"{imaging_rootDir}/{session}/ops.npy", allow_pickle=True).item()
        stat = np.load(f"{imaging_rootDir}/{session}/stat.npy", allow_pickle=True)
        iscell = np.load(f"{imaging_rootDir}/{session}/iscell.npy", allow_pickle=True)
        suite2p_cellids = np.nonzero(iscell[:,0])[0]

        pfs_session = pfs[pfs["session id"] == session]
        cellids_tuned = pfs_session["cell id"].unique()
        cellids_newly_formed = pfs_session[(pfs_session["category"] == "btsp") | (pfs_session["category"] == "non-btsp")]["cell id"].unique()
        cellids_btsp = pfs_session[pfs_session["category"] == "btsp"]["cell id"].unique()

        scale = 2
        fig, axs = plt.subplots(1,3, figsize=(scale*8,scale*4))
        axs[0].matshow(np.log(ops['meanImg']), cmap='gray')
        axs[1].matshow(np.log(ops['meanImg']), cmap='gray')
        axs[2].matshow(np.log(ops['meanImg']), cmap='gray')

        for cellid in cellids_tuned:
            im = np.ones((ops['Ly'], ops['Lx']))
            im[:] = np.nan

            suite2p_cellid = suite2p_cellids[cellid]
            ypix = stat[suite2p_cellid]["ypix"]#[~stat[suite2p_cellid]['overlap']]
            xpix = stat[suite2p_cellid]["xpix"]#[~stat[suite2p_cellid]['overlap']]
            im[ypix, xpix] = 0.43  # color
            axs[0].matshow(im, cmap="hsv", alpha=0.66, vmin=0, vmax=1)
            axs[0].set_axis_off()
            axs[0].set_title("tuned cells")

        for cellid in cellids_newly_formed:
            im = np.ones((ops['Ly'], ops['Lx']))
            im[:] = np.nan

            suite2p_cellid = suite2p_cellids[cellid]
            ypix = stat[suite2p_cellid]["ypix"]#[~stat[suite2p_cellid]['overlap']]
            xpix = stat[suite2p_cellid]["xpix"]#[~stat[suite2p_cellid]['overlap']]
            im[ypix, xpix] = 0.14  # color
            axs[1].matshow(im, cmap="hsv", alpha=0.66, vmin=0, vmax=1)
            axs[1].set_axis_off()
            axs[1].set_title("cells with newly formed, stable PFs")

        for cellid in cellids_btsp:
            im = np.ones((ops['Ly'], ops['Lx']))
            im[:] = np.nan

            suite2p_cellid = suite2p_cellids[cellid]
            ypix = stat[suite2p_cellid]["ypix"]#[~stat[suite2p_cellid]['overlap']]
            xpix = stat[suite2p_cellid]["xpix"]#[~stat[suite2p_cellid]['overlap']]
            im[ypix, xpix] = 0.85  # color
            axs[2].matshow(im, cmap="hsv", alpha=0.7, vmin=0, vmax=1)
            axs[2].set_axis_off()
            axs[2].set_title("cells with BTSP PFs")

        plt.tight_layout()
        plt.savefig(f"{imaging_rootDir}/{session}_FOV_masks_withoutShift.pdf")
        plt.close()
