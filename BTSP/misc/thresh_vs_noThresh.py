import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from BTSP.BtspStatistics import BtspStatistics
from BTSP.constants import AREA_PALETTE, CATEGORIES
import scipy
import tqdm
from utils import grow_df
import pickle


data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
depth_folder = f"{data_root}/depths"

extra_info_thresh = ""
extra_info_no_thresh = "noCaThreshold"

thresh_stat = BtspStatistics("CA3", data_root, output_root, extra_info_thresh, is_shift_criterion_on=True, is_notebook=False)
thresh_stat.load_data()
thresh_stat.load_cell_data()
thresh_stat.filter_low_behavior_score()
thresh_stat.calc_shift_gain_distribution(unit="cm")
thresh_stat.pfs_df["threshold"] = True
thresh_stat.shift_gain_df["threshold"] = True

no_thresh_stat = BtspStatistics("CA3", data_root, output_root, extra_info_no_thresh, is_shift_criterion_on=True, is_notebook=False)
no_thresh_stat.load_data()
no_thresh_stat.filter_low_behavior_score()
no_thresh_stat.calc_shift_gain_distribution(unit="cm")
no_thresh_stat.pfs_df["threshold"] = False
no_thresh_stat.shift_gain_df["threshold"] = False

pfs_df = pd.concat([thresh_stat.pfs_df, no_thresh_stat.pfs_df]).reset_index(drop=True)
shift_gain_df = pd.concat([thresh_stat.shift_gain_df, no_thresh_stat.shift_gain_df]).reset_index(drop=True)


def compare_depth_analysis():
    global pfs_df
    categories_colors_RGB = [category.color for _, category in CATEGORIES.items()]

    sessions = np.unique(pfs_df["session id"].values)
    depths_df_all = None
    for session in sessions:
        try:
            depths_df = pd.read_excel(f"{depth_folder}/cellDepth_depths_{session}.xlsx")
        except FileNotFoundError:
            print(f"failed to load depths for {session}; skippping")
            continue
        depths_df = depths_df.reset_index().rename(
            columns={"index": "cell id"})  # create cell ids from suite2p roi indices
        animal = session.partition("_")[0]
        depths_df["animal id"] = animal
        depths_df["session id"] = session
        depths_df_all = grow_df(depths_df_all, depths_df)
    pfs_df = pfs_df.set_index(["animal id", "session id", "cell id"])
    depths_df_all = depths_df_all.set_index(["animal id", "session id", "cell id"])
    pfs_df = pfs_df.join(depths_df_all, on=["animal id", "session id", "cell id"]).reset_index()
    depths_df_all = depths_df_all.reset_index()

    tc_df = thresh_stat.tc_df.rename(columns={"sessionID": "session id", "cellid": "cell id"})
    tc_df["Ca2+ event rate"] = tc_df.apply(lambda row: 60 * row["tc"].n_events / row["tc"].total_time, axis=1)
    tc_df = tc_df[tc_df["corridor"] == 14].set_index(
        ["session id", "cell id"])  # we can select either corridor bc. event rate is a cell-level property

    # load all cells (i.e not just tuned ones), calculate event rates for all of them
    sessions = np.unique(pfs_df["session id"].values)
    cells_all_sessions = []
    for session in sessions:
        print(session)
        cell_list_path = f"{data_root}/tuned_cells/CA3{extra_info_thresh}/all_cells_{session}.pickle"
        with open(cell_list_path, "rb") as cell_list_file:
            cell_list = pickle.load(cell_list_file)
            cells_all_sessions += cell_list
    cells_df = pd.DataFrame([{"sessionID": c.sessionID, "cellid": c.cellid, "cell": c} for c in cells_all_sessions])
    cells_df = cells_df.rename(columns={"sessionID": "session id", "cellid": "cell id"})
    cells_df["Ca2+ event rate"] = cells_df.apply(lambda row: 60 * row["cell"].n_events / row["cell"].total_time, axis=1)
    cells_df = cells_df.set_index(["session id", "cell id"])

    # join cell tables with depth tables
    depths_df_tuned = depths_df_all.set_index(["session id", "cell id"])
    depths_df_tuned = depths_df_all.join(tc_df, on=["session id", "cell id"]).reset_index()
    depths_df_all = depths_df_all.set_index(["session id", "cell id"])
    depths_df_all = depths_df_all.join(cells_df, on=["session id", "cell id"]).reset_index()

    # discard ROIs that are clearly out of the SP
    depths_df_tuned = depths_df_tuned[(depths_df_tuned["depth"] >= -1) & (depths_df_tuned["depth"] < 1.5)]
    depths_df_all = depths_df_all[(depths_df_all["depth"] >= -1) & (depths_df_all["depth"] < 1.5)]
    pass

    def create_depth_bins(bounds, step):
        lb, ub = bounds
        depth_range = np.arange(lb, ub + step, step)  # 1+step is necessary to include 1
        find_nearest = lambda row: np.round(depth_range[np.abs(depth_range - row["depth"]).argmin()], 3)
        depths_df_tuned["depth range"] = depths_df_tuned.apply(find_nearest, axis=1)

    pfs_thresh = pfs_df[pfs_df["threshold"] == True]
    pfs_no_thresh = pfs_df[pfs_df["threshold"] == False]

    ######## PLOT DISTRIBUTIONS
    fig, axs = plt.subplots(2,2, sharey="row")
    sns.histplot(data=pfs_thresh.sort_values(by="category_order"), x="depth", hue="category", multiple="stack",
                 binrange=(-1, 1.5), binwidth=0.1, palette=categories_colors_RGB, ax=axs[0,0], legend=False)
    sns.histplot(data=pfs_no_thresh.sort_values(by="category_order"), x="depth", hue="category", multiple="stack",
                 binrange=(-1, 1.5), binwidth=0.1, palette=categories_colors_RGB, ax=axs[0,1])

    sns.histplot(data=pfs_thresh.sort_values(by="category_order"), x="depth", hue="category", multiple="fill",
                 binrange=(0, 1), binwidth=0.1, palette=categories_colors_RGB, ax=axs[1,0], legend=False)
    sns.histplot(data=pfs_no_thresh.sort_values(by="category_order"), x="depth", hue="category", multiple="fill",
                 binrange=(0, 1), binwidth=0.1, palette=categories_colors_RGB, ax=axs[1,1], legend=False)

    ### COMPARE PF CATEGORY-WISE
    step = 0.1
    depth_range = np.arange(-0.5, 1+step, step)  # 1+step is necessary to include 1
    find_nearest_depth_bin = lambda row: np.round(depth_range[np.abs(depth_range - row["depth"]).argmin()],3)

    # filter for depth range
    pfs_thresh = pfs_thresh[(pfs_thresh["depth"] >= -0.5) & (pfs_thresh["depth"] <= 1.0)].sort_values(by="category_order")
    pfs_no_thresh = pfs_no_thresh[(pfs_no_thresh["depth"] >= -0.5) & (pfs_no_thresh["depth"] <= 1.0)].sort_values(by="category_order")

    pfs_thresh["depth range"] = pfs_thresh.apply(find_nearest_depth_bin, axis=1)
    pfs_no_thresh["depth range"] = pfs_no_thresh.apply(find_nearest_depth_bin, axis=1)

    pf_category_proportional_change = pfs_no_thresh.groupby(["category", "depth range"]).count() / pfs_thresh.groupby(["category", "depth range"]).count()
    pf_category_proportional_change = pf_category_proportional_change['animal id'].reset_index().rename({"animal id": "prop. change"}, axis=1)

    plt.figure()
    new_color_order = ["transient", "early", "unreliable", "non-btsp", "btsp"]
    new_color_RGB = [CATEGORIES[cat].color for cat in new_color_order]
    sns.barplot(data=pf_category_proportional_change, x="depth range", y="prop. change",
                hue="category", palette=new_color_RGB, width=0.6)
    plt.axhline(1, c="black", linewidth=2, linestyle="--")
    plt.show()

compare_depth_analysis()