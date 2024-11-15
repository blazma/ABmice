import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

area = "CA1"
place_fields_path = "BTSP_analysis_CA1_230912/place_fields_df.pickle"
place_fields_df = pd.read_pickle(place_fields_path)

n_bins = 75
n_laps = 50

background = np.ones((n_laps, n_bins))

fig, axs = plt.subplots(4,1)

categories = ["early", "transient", "non-btsp", "btsp"]
colors = ["cyan", "yellow", "red", "purple"]
for i_category, category in enumerate(categories):
	ax = axs[i_category]
	color = colors[i_category]
	pfs = place_fields_df.loc[place_fields_df["category"] == category]
	print(category, 1/len(pfs.index))
	for _, pf in pfs.iterrows():
		lb = pf["lower bound"]
		ub = pf["upper bound"]
		fl = pf["formation lap"]
		el = pf["end lap"]
		ax.axvspan(lb, ub, ymin=fl/n_laps, ymax=el/n_laps, color=color, alpha=0.005, edgecolor=None)
	ax.imshow(background, cmap="binary", origin="lower")
plt.show()