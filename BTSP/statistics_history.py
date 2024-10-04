import numpy as np
import numpy.random
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from BtspStatistics import BtspStatistics
from constants import AREA_PALETTE
import scipy

data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
extra_info = "reboot_withoutPost_historyDependent"
extra_info_all = "reboot_withoutPost"

def calc_diff():
    stat_change = BtspStatistics("CA1", data_root, output_root, extra_info, is_shift_criterion_on=True, is_notebook=False, history="CHANGE")
    stat_change.load_data()
    stat_change.load_cell_data()
    #stat_change.filter_low_behavior_score()
    stat_change.calc_place_field_quality()
    df_change = stat_change.pfs_df
    pf_counts_change = df_change.groupby(by=["session id", "cell id", "corridor"]).count().rename({"animal id": "PF count"}, axis=1)["PF count"]
    df_change = df_change.set_index(keys=["session id", "cell id", "corridor"])
    df_change = df_change.join(pf_counts_change)

    stat_stay = BtspStatistics("CA1", data_root, output_root, extra_info, is_shift_criterion_on=True, is_notebook=False, history="STAY")
    stat_stay.load_data()
    stat_stay.load_cell_data()
    #stat_stay.filter_low_behavior_score()
    stat_stay.calc_place_field_quality()
    df_stay = stat_stay.pfs_df
    pf_counts_stay = df_stay.groupby(by=["session id", "cell id", "corridor"]).count().rename({"animal id": "PF count"}, axis=1)["PF count"]
    df_stay = df_stay.set_index(keys=["session id", "cell id", "corridor"])
    df_stay = df_stay.join(pf_counts_stay)

    unique_pfs_idxs = pf_counts_change[pf_counts_change == 1].index.intersection(pf_counts_stay[pf_counts_stay == 1].index)
    df_change = df_change.loc[unique_pfs_idxs]
    df_stay = df_stay.loc[unique_pfs_idxs]

    mean_com_diff = np.abs(df_change["Mean COM"] - df_stay["Mean COM"])
    mean_rate_diff = np.abs(df_change["Mean rate"] - df_stay["Mean rate"])
    max_rate_diff = np.abs(df_change["Max rate"] - df_stay["Max rate"])

    return mean_com_diff, mean_rate_diff, max_rate_diff

def calc_diff_shuffled():

    def calc_means(rate_matrix):
        rates_normalized_within_bounds = np.nanmax(rate_matrix, axis=0) / np.nanmax(rate_matrix)
        act_laps_idxs = np.where(rates_normalized_within_bounds >= 0.1)[0]
        RM_active_laps = rate_matrix[:, act_laps_idxs]
        COM = lambda arr: np.average(np.array(list(range(len(arr)))), weights=arr)  # weighted avg. of indices of input array, weights = input array itself
        COMs_active_laps = np.apply_along_axis(COM, axis=0, arr=RM_active_laps)
        mean_COM = COMs_active_laps.mean()
        mean_rate = RM_active_laps.mean()
        max_rate = RM_active_laps.max()
        return mean_COM, mean_rate, max_rate

    stat_all = BtspStatistics("CA1", data_root, output_root, extra_info_all, is_shift_criterion_on=True,
                                 is_notebook=False, history="ALL")
    stat_all.load_data()
    stat_all.load_cell_data()
    # stat_change.filter_low_behavior_score()
    df_all = stat_all.pfs_df
    pf_counts_all = df_all.groupby(by=["session id", "cell id", "corridor"]).count().rename({"animal id": "PF count"}, axis=1)["PF count"]
    df_all = df_all.set_index(keys=["session id", "cell id", "corridor"])
    df_all = df_all.join(pf_counts_all)

    stat_change = BtspStatistics("CA1", data_root, output_root, extra_info, is_shift_criterion_on=True,
                                 is_notebook=False, history="CHANGE")
    stat_change.load_data()
    stat_change.load_cell_data()
    # stat_change.filter_low_behavior_score()
    df_change = stat_change.pfs_df
    pf_counts_change = df_change.groupby(by=["session id", "cell id", "corridor"]).count().rename({"animal id": "PF count"}, axis=1)["PF count"]
    df_change = df_change.set_index(keys=["session id", "cell id", "corridor"])
    df_change = df_change.join(pf_counts_change)

    stat_stay = BtspStatistics("CA1", data_root, output_root, extra_info, is_shift_criterion_on=True, is_notebook=False,
                               history="STAY")
    stat_stay.load_data()
    stat_stay.load_cell_data()
    # stat_stay.filter_low_behavior_score()
    df_stay = stat_stay.pfs_df
    pf_counts_stay = df_stay.groupby(by=["session id", "cell id", "corridor"]).count().rename({"animal id": "PF count"}, axis=1)["PF count"]
    df_stay = df_stay.set_index(keys=["session id", "cell id", "corridor"])
    df_stay = df_stay.join(pf_counts_stay)

    # the following line finds the (session id, cell id, corridor) tuples where there was only one place field but which appeared in both CHANGE and STAY
    unique_pfs_idxs = pf_counts_change[pf_counts_change == 1].index.intersection(pf_counts_stay[pf_counts_stay == 1].index)
    unique_pfs_idxs = unique_pfs_idxs.intersection(pf_counts_all[pf_counts_all == 1].index)  # we further restrict analysis to when the PF exists in ALL too

    rng = numpy.random.default_rng(seed=1234)
    df_shuffled_dicts = []
    skipped_pfs = 0
    for pf_counter, pf_idx in enumerate(unique_pfs_idxs):
        if pf_counter % 100 == 0:
            print(f"{pf_counter} / {len(unique_pfs_idxs)}")

        tcs = stat_stay.tc_df.set_index(keys=["sessionID", "cellid", "corridor"]).loc[pf_idx]
        tc_stay = tcs[tcs["history"] == "STAY"]["tc"].iloc[0]
        tc_change = tcs[tcs["history"] == "CHANGE"]["tc"].iloc[0]
        n_laps = min([tc_change.rate_matrix.shape[1], tc_stay.rate_matrix.shape[1]])

        choices = ["STAY", "CHANGE"]
        choice_list1 = rng.choice([0, 1], size=n_laps)
        choice_list2 = rng.choice([0, 1], size=n_laps)

        RM_shuffled1 = np.zeros((tc_stay.rate_matrix.shape[0], n_laps))
        RM_shuffled2 = np.zeros((tc_stay.rate_matrix.shape[0], n_laps))
        for i_lap in range(n_laps):
            choice1 = choices[choice_list1[i_lap]]
            if choice1 == "STAY":
                rates_lap1 = tc_stay.rate_matrix[:,i_lap]
            else:
                rates_lap1 = tc_change.rate_matrix[:,i_lap]
            RM_shuffled1[:,i_lap] = rates_lap1

            choice2 = choices[choice_list2[i_lap]]
            if choice2 == "STAY":
                rates_lap2 = tc_stay.rate_matrix[:,i_lap]
            else:
                rates_lap2 = tc_change.rate_matrix[:,i_lap]
            RM_shuffled2[:, i_lap] = rates_lap2

        ### TODO: confirm that this is fair to do; i borrow code from BtspStatistics and keep the format to stay consistent
        ### TODO: it's not obvious which PF should be considered (STAY vs CHANGE), so i ignore PF bounds altogether in the shuffled RMs
        pf_original = df_all.loc[pf_idx]
        lb, ub, fl, el = df_all.loc[pf_idx]["lower bound"][0], df_all.loc[pf_idx]["upper bound"][0], df_all.loc[pf_idx]["formation lap"][0], df_all.loc[pf_idx]["end lap"][0]
        if RM_shuffled1[lb:ub, fl:el].sum() == 0 or RM_shuffled2[lb:ub, fl:el].sum() == 0:
            skipped_pfs += 1
            continue
        mean_COM1, mean_rate1, max_rate1 = calc_means(RM_shuffled1[lb:ub, fl:el])
        mean_COM2, mean_rate2, max_rate2 = calc_means(RM_shuffled2[lb:ub, fl:el])

        df_shuffled_dict = {
            "session id": pf_idx[0],
            "cell id": pf_idx[1],
            "corridor": pf_idx[2],
            "shuffled |mean COM diff|": np.abs(mean_COM1 - mean_COM2),
            "shuffled |mean rate diff|": np.abs(mean_rate1 - mean_rate2),
            "shuffled |max rate diff|": np.abs(max_rate1 - max_rate2),
        }
        df_shuffled_dicts.append(df_shuffled_dict)
    df_shuffled = pd.DataFrame.from_dict(df_shuffled_dicts)
    #df_shuffled["shuffled |mean COM diff|"].hist(range=[0,75], bins=75, histtype="step", label="shuffled")

    mean_com_diff = df_shuffled["shuffled |mean COM diff|"]
    mean_rate_diff = df_shuffled["shuffled |mean rate diff|"]
    max_rate_diff = df_shuffled["shuffled |max rate diff|"]

    return mean_com_diff, mean_rate_diff, max_rate_diff

mean_com_diff, mean_rate_diff, max_rate_diff = calc_diff()
mean_com_diff_sh, mean_rate_diff_sh, max_rate_diff_sh = calc_diff_shuffled()

fig, axs = plt.subplots(1,3)
mean_com_diff.hist(range=[0, 75], bins=75, histtype="step", label="normal order", ax=axs[0])
mean_com_diff_sh.hist(range=[0,75], bins=75, histtype="step", label="shuffled", ax=axs[0])
axs[0].set_title("|mean COM diff|")

mean_rate_diff.hist(range=[0, 80], bins=80, histtype="step", label="normal order", ax=axs[1])
mean_rate_diff_sh.hist(range=[0, 80], bins=80, histtype="step", label="shuffled", ax=axs[1])
axs[1].set_title("|mean rate diff|")

max_rate_diff.hist(range=[0, 300], bins=60, histtype="step", label="normal order", ax=axs[2])
max_rate_diff_sh.hist(range=[0, 300], bins=60, histtype="step", label="shuffled", ax=axs[2])
axs[2].set_title("|max rate diff|")
plt.legend()

######## LOGARITHMIC
fig, axs = plt.subplots(1,3)
mean_com_diff.hist(range=[0, 75], bins=75, histtype="step", label="normal order", ax=axs[0], log=True)
mean_com_diff_sh.hist(range=[0,75], bins=75, histtype="step", label="shuffled", ax=axs[0], log=True)
axs[0].set_title("|mean COM diff|")

mean_rate_diff.hist(range=[0, 80], bins=80, histtype="step", label="normal order", ax=axs[1], log=True)
mean_rate_diff_sh.hist(range=[0, 80], bins=80, histtype="step", label="shuffled", ax=axs[1], log=True)
axs[1].set_title("|mean rate diff|")

max_rate_diff.hist(range=[0, 300], bins=60, histtype="step", label="normal order", ax=axs[2], log=True)
max_rate_diff_sh.hist(range=[0, 300], bins=60, histtype="step", label="shuffled", ax=axs[2], log=True)
axs[2].set_title("|max rate diff|")
plt.legend()

#### CUMULATIVE
fig, axs = plt.subplots(1,3)
mean_com_diff.hist(range=[0, 75], bins=75, histtype="step", label="normal order", ax=axs[0], cumulative=True, density=1)
mean_com_diff_sh.hist(range=[0,75], bins=75, histtype="step", label="shuffled", ax=axs[0], cumulative=True, density=1)
axs[0].set_title("|mean COM diff|")

mean_rate_diff.hist(range=[0, 80], bins=80, histtype="step", label="normal order", ax=axs[1], cumulative=True, density=1)
mean_rate_diff_sh.hist(range=[0, 80], bins=80, histtype="step", label="shuffled", ax=axs[1], cumulative=True, density=1)
axs[1].set_title("|mean rate diff|")

max_rate_diff.hist(range=[0, 300], bins=60, histtype="step", label="normal order", ax=axs[2], cumulative=True, density=1)
max_rate_diff_sh.hist(range=[0, 300], bins=60, histtype="step", label="shuffled", ax=axs[2], cumulative=True, density=1)
axs[2].set_title("|max rate diff|")
plt.legend()
plt.show()

