import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from BTSP.Statistics_BothAreas import Statistics_BothAreas

data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = "NFafter5Laps"
extra_info_CA3 = "NFafter5Laps"
extra_info = "NFafter5Laps"

filter_overextended = True

stats = Statistics_BothAreas(data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info,
                             filter_overextended=filter_overextended, create_output_folder=False)
stats.load_data()

cols = ["area", "animal id", "session id", "cell id", "corridor", "newly formed", "category", "lower bound", "upper bound", "active laps", "formation lap", "end lap", "initial shift", "log10(formation gain)", "spike counts"]
pfs = stats.pfs_df[cols]
pfs.to_pickle(f"{output_root}/pfs_df.pickle")
