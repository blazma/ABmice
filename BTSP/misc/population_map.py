import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import grow_df

base_folder = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\tuned_cells"
area = "CA1"

tuning_curves_df = None
for _, _, files in os.walk(rf"{base_folder}\{area}"):
    tuncurv_filenames = [file for file in files if "tuning_curves" in file]
    for tuncurv_filename in tuncurv_filenames:
        with open(rf"{base_folder}\{area}\{tuncurv_filename}", "rb") as tuncurv_file:
            tuning_curve_df = pd.read_pickle(tuncurv_file)
            tuning_curves_df = grow_df(tuning_curves_df, tuning_curve_df)

# filter out all zero tuning curves
tuning_curves_df = tuning_curves_df.sort_values(by=["area", "animalID", "sessionID", "cellid"]).reset_index(drop=True)
tc_14 = tuning_curves_df[tuning_curves_df["corridor"] == 14].reset_index(drop=True)
tc_15 = tuning_curves_df[tuning_curves_df["corridor"] == 15].reset_index(drop=True)

for i_row, row in tc_14.iterrows():
    if np.sum(row["tuning curve"]) == 0:
        tc_14.drop(i_row, inplace=True)
        tc_15.drop(i_row, inplace=True)
for i_row, row in tc_15.iterrows():
    if np.sum(row["tuning curve"]) == 0:
        tc_14.drop(i_row, inplace=True)
        tc_15.drop(i_row, inplace=True)
tc_14 = tc_14.reset_index(drop=True)
tc_15 = tc_15.reset_index(drop=True)

# calculate center of mass for each tuning curve -- serves as basis for ordering
tc_14["com"] = tc_14.apply(lambda row: np.average(np.arange(75), weights=row["tuning curve"]), axis=1)
tc_15["com"] = tc_15.apply(lambda row: np.average(np.arange(75), weights=row["tuning curve"]), axis=1)
tc_14["max"] = tc_14.apply(lambda row: np.argmax(row["tuning curve"]), axis=1)
tc_15["max"] = tc_15.apply(lambda row: np.argmax(row["tuning curve"]), axis=1)

# [CA1 only] subsample bc there are just too many cells
if area == "CA1":
    sample = 630
    rng = np.random.default_rng(1234)
    random_idx = rng.integers(0,len(tc_14), size=sample)
    tc_14 = tc_14.iloc[random_idx]
    tc_15 = tc_15.iloc[random_idx]

# re-order and store indices for each corridor
by = "max"
tc_14 = tc_14.sort_values(by=by, ignore_index=True).reset_index(names="order14")
tc_14 = tc_14.sort_values(by=["area", "animalID", "sessionID", "cellid"]).reset_index(drop=True)
tc_15 = tc_15.sort_values(by=["area", "animalID", "sessionID", "cellid"]).reset_index(drop=True)
tc_15["order14"] = tc_14["order14"].values

tc_15 = tc_15.sort_values(by=by, ignore_index=True).reset_index(names="order15")
tc_15 = tc_15.sort_values(by=["area", "animalID", "sessionID", "cellid"]).reset_index(drop=True)
tc_14 = tc_14.sort_values(by=["area", "animalID", "sessionID", "cellid"]).reset_index(drop=True)
tc_14["order15"] = tc_15["order15"].values

# sanitizing - getting rid of artifact-like tiny negative values that would f*** up normalization
tc_14["tuning curve"] = np.abs(tc_14["tuning curve"])
tc_15["tuning curve"] = np.abs(tc_15["tuning curve"])

n = len(tc_14)  # number of cells
def sort_popmap(df, col, normalized=False):
    if normalized:
        return np.array([df.sort_values(by=col)["tuning curve"].iloc[i] / np.nanmax(df.sort_values(by=col)["tuning curve"].iloc[i]) for i in range(n)])
    else:
        return np.array([df.sort_values(by=col)["tuning curve"].iloc[i] for i in range(n)])

scale = 2
normalized = True
cmap = "gray_r"
fig, axs = plt.subplots(2,2, figsize=(scale*3,scale*4))
ax = sns.heatmap(sort_popmap(tc_14, "order14", normalized=normalized), ax=axs[0,0], cmap=cmap, xticklabels=10, yticklabels=100)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels([])
ax = sns.heatmap(sort_popmap(tc_14, "order15", normalized=normalized), ax=axs[1,0], cmap=cmap, xticklabels=10, yticklabels=100)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax = sns.heatmap(sort_popmap(tc_15, "order14", normalized=normalized), ax=axs[0,1], cmap=cmap, xticklabels=10, yticklabels=100)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax = sns.heatmap(sort_popmap(tc_15, "order15", normalized=normalized), ax=axs[1,1], cmap=cmap, xticklabels=10, yticklabels=100)
ax.set_yticklabels([])

output_folder = rf"C:\Users\martin\home\phd\btsp_project\analyses\manual\statistics\{area}"
plt.tight_layout()
plt.savefig(f"{output_folder}\POPMAP_{by}.pdf")
#plt.savefig(f"{output_folder}\POPMAP_{by}.svg")
plt.close()