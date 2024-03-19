import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from utils import grow_df

base_folder = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\tuned_cells"
tcs_df = None
for area in ["CA1", "CA3"]:
    for _, _, files in os.walk(rf"{base_folder}\{area}"):
        tcl_filenames = [file for file in files if "tuned_cells_" in file]
        for tcl_filename in tcl_filenames:
            with open(rf"{base_folder}\{area}\{tcl_filename}", "rb") as tcl_file:
                print(rf"{base_folder}\{area}\{tcl_filename}")
                tcl = pickle.load(tcl_file)
                for tc in tcl:
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

tcs_df_c14 = tcs_df[tcs_df["corridor"] == 14]
tcs_df_long = tcs_df_c14.melt(id_vars=["area", "animalID", "sessionID"],
                              value_vars=["event rate"], var_name="event rate", value_name="events / minute")
fig, ax = plt.subplots(figsize=(2.6,3))
palette= ["#C0BAFF", "#FFB0BA"]
sns.boxplot(data=tcs_df_long, x="event rate", y="events / minute", hue="area", showfliers=False, ax=ax, palette=palette,
            linecolor="black", linewidth=1.5, saturation=1, gap=0.2)
ax.set_xlabel("")
ax.set_ylim([0,8])
output_root = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\statistics"
plt.tight_layout()
plt.savefig(f"{output_root}/event_rate.pdf")
plt.savefig(f"{output_root}/event_rate.svg")
plt.close()

mean_eventrates = tcs_df[["area", "animalID", "sessionID", "event rate"]].groupby(["area", "animalID", "sessionID"]).mean().reset_index()
mean_eventrates.to_pickle(f"{output_root}/mean_event_rates.pickle")
ax = sns.swarmplot(data=mean_eventrates, x="area", hue="animalID", y="event rate", dodge=True)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"{output_root}/event_rates_by_session.pdf")
plt.close()
