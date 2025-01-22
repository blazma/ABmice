import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from BtspStatistics import BtspStatistics
from constants import AREA_PALETTE
import scipy
import tqdm
from copy import deepcopy
from utils import makedir_if_needed
import warnings
warnings.filterwarnings("ignore")


data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = ""
extra_info_CA3 = ""

def analyse_by_animals(area, extra_info):
    print(area)
    stat = BtspStatistics(area, data_root, output_root, extra_info, is_shift_criterion_on=True, is_notebook=False)
    stat.load_data()
    stat.filter_low_behavior_score()
    stat.calc_place_field_proportions()
    stat.calc_shift_gain_distribution(unit="cm")
    stat.pfs_df["area"] = area
    stat.shift_gain_df["area"] = area

    animals = stat.pfs_df[f"animal id"].unique()
    for animal in tqdm.tqdm(animals):
        mpl.rcParams.update(mpl.rcParamsDefault)

        stat_animal = deepcopy(stat)
        stat_animal._use_font()
        stat_animal.pfs_df = stat.pfs_df[stat.pfs_df["animal id"] == animal]
        stat_animal.shift_gain_df = stat.shift_gain_df[stat.shift_gain_df["animal id"] == animal]
        stat_animal.output_root = f"{stat.output_root}/{animal}/"
        makedir_if_needed(stat_animal.output_root)

        stat_animal.plot_place_fields()
        stat_animal.plot_place_fields_by_session()
        stat_animal.plot_place_field_proportions()
        try:
            stat_animal.plot_shift_gain_distribution(without_transient=False)
        except ValueError: # happens during tests if too few pfs
            print(f"error during animal {animal}; skipping")
            continue

        if area == "CA1":
            sessions = stat_animal.pfs_df["session id"].unique()
            for session in sessions:
                mpl.rcParams.update(mpl.rcParamsDefault)

                stat_session = deepcopy(stat)
                stat_session._use_font()
                stat_session.pfs_df = stat_animal.pfs_df[stat_animal.pfs_df["session id"] == session]
                stat_session.shift_gain_df = stat_animal.shift_gain_df[stat_animal.shift_gain_df["session id"] == session]
                stat_session.output_root = f"{stat.output_root}/{animal}/{session}"
                makedir_if_needed(stat_session.output_root)

                stat_session.plot_place_fields()
                stat_session.plot_place_fields_by_session()
                stat_session.plot_place_field_proportions()
                try:
                    stat_session.plot_shift_gain_distribution(without_transient=False)
                except ValueError:  # happens during tests if too few pfs
                    print(f"error during session {session}; skipping")
                    continue

analyse_by_animals("CA1", extra_info_CA1)
#analyse_by_animals("CA3", extra_info_CA3)
