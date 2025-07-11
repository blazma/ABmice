import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from utils import grow_df
from BTSP.BtspStatistics import BtspStatistics

def use_font():
    from matplotlib import font_manager
    font_dirs = ['C:\\Users\\martin\\home\\phd\\misc']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Roboto'
    plt.rcParams['mathtext.it'] = 'Roboto'

data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = "NFafter5Laps"
extra_info_CA3 = "NFafter5Laps"

CA1_stat = BtspStatistics("CA1", data_root, output_root, extra_info_CA1, is_shift_criterion_on=True, is_notebook=False)
CA1_stat.load_data()
CA1_stat.filter_low_behavior_score()
CA1_stat.calc_shift_gain_distribution(unit="cm")

CA3_stat = BtspStatistics("CA3", data_root, output_root, extra_info_CA3, is_shift_criterion_on=True, is_notebook=False)
CA3_stat.load_data()
CA3_stat.filter_low_behavior_score()
CA3_stat.calc_shift_gain_distribution(unit="cm")

base_folders = {
    "CA1": fr"C:\Users\martin\home\phd\btsp_project\analyses\manual\tuned_cells\CA1_{extra_info_CA1}",
    "CA3": fr"C:\Users\martin\home\phd\btsp_project\analyses\manual\tuned_cells\CA3_{extra_info_CA3}"
}

df = pd.concat((CA1_stat.cell_stats_df, CA3_stat.cell_stats_df))
filtered_sessions_list = list(df["sessionID"].values)

tcs_df = None
for area in ["CA1", "CA3"]:
    for _, _, files in os.walk(rf"{base_folders[area]}"):
        tcl_filenames = [file for file in files if "tuned_cells_" in file]
        for tcl_filename in tcl_filenames:
            with open(rf"{base_folders[area]}\{tcl_filename}", "rb") as tcl_file:
                print(rf"{base_folders[area]}\{tcl_filename}")
                tcl = pickle.load(tcl_file)
                for tc in tcl:
                    if tc.sessionID not in filtered_sessions_list:
                        continue
                    tc_dict = {
                        "area": area,
                        "animalID": tc.sessionID.partition("_")[0],  # TODO: store animalID in TunedCell directly
                        "sessionID": tc.sessionID,
                        "cellid": tc.cellid,
                        "corridor": tc.corridor,
                        "Ca2+ events": tc.n_events,
                        "total time": tc.total_time,
                        "event rate": 60 * tc.n_events / tc.total_time
                    }
                    tc_df = pd.DataFrame.from_dict([tc_dict])
                    tcs_df = grow_df(tcs_df, tc_df)

use_font()
tcs_df_c14 = tcs_df[tcs_df["corridor"] == 14]
tcs_df_long = tcs_df_c14.melt(id_vars=["area", "animalID", "sessionID"],
                              value_vars=["event rate"], var_name="event rate", value_name="events / minute")
fig, ax = plt.subplots(figsize=(2,3))
palette= ["#C0BAFF", "#FFB0BA"]
sns.boxplot(data=tcs_df_long, x="event rate", y="events / minute", hue="area", showfliers=False, ax=ax, palette=palette,
            linecolor="black", linewidth=1.5, saturation=1, gap=0.2, legend=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([], [])
ax.set_ylim([0,7])
ax.set_xlabel("")
output_root = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\statistics"
plt.tight_layout()
plt.savefig(f"{output_root}/event_rate.pdf")
plt.savefig(f"{output_root}/event_rate.svg")
plt.close()

mean_eventrates = tcs_df[["area", "animalID", "sessionID", "event rate"]].groupby(["area", "animalID", "sessionID"]).mean().reset_index()
mean_eventrates.to_pickle(f"{output_root}/mean_event_rates.pickle")
ax = sns.swarmplot(data=mean_eventrates, x="area", hue="animalID", y="event rate", dodge=True, palette="tab10")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"{output_root}/event_rates_by_session.pdf")
plt.close()

# statistics
import scipy
test = scipy.stats.mannwhitneyu(tcs_df[tcs_df["area"] == "CA1"]["event rate"].values, tcs_df[tcs_df["area"] == "CA3"]["event rate"].values)
print(test.pvalue)
