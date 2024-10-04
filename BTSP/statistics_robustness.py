import numpy as np
import seaborn as sns
import pandas as pd
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
from BtspStatistics import BtspStatistics
from constants import AREA_PALETTE
import scipy
import tqdm
from copy import deepcopy
from utils import makedir_if_needed, grow_df
import itertools
import warnings
warnings.filterwarnings("ignore")


data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = ""
extra_info_CA3 = ""


def normalize_session_numbers(area, extra_info, selection_size):
    stat = BtspStatistics(area, data_root, output_root, extra_info, is_shift_criterion_on=True, is_notebook=False)
    stat.load_data()
    stat.filter_low_behavior_score()
    stat.calc_place_field_proportions()
    stat.calc_shift_gain_distribution(unit="cm")
    stat.pfs_df["area"] = area
    stat.shift_gain_df["area"] = area

    animals = stat.pfs_df[f"animal id"].unique()
    animals_sessions = stat.pfs_df[["animal id", "session id"]].groupby(["animal id", "session id"]).count().reset_index()
    animals_sessions_counts = animals_sessions.groupby(["animal id"]).count()

    ##########################
    ###### SESSION SUBSAMPLING
    # step 1: create combinations of sessions with a fixed size of session subsample
    # step 2: create permutations of these session-combinations which can then be used to run btsp analysis

    ###### STEP 1: combinations
    print("COMBINATIONS")
    dict_animals_sessions_counts = animals_sessions_counts.to_dict()["session id"]  # number of sessions per animal

    # store session combinations within one animal
    dict_combinations = {}
    for animal, count in tqdm.tqdm(dict_animals_sessions_counts.items()):
        # creates combinations of size 3 out of index lists of sessions within one animal:
        comb = list(itertools.combinations(range(count), selection_size))
        if comb:
            dict_combinations[animal] = comb
        else:
            # if number of sessions < selection_size, itertools.combinations returns empty list, so we store just
            # a list of the session indices within one animal:
            dict_combinations[animal] = (list(range(count)),)
    # creates a cartesian product of all possible combinations of sessions *across* animals using previously
    # created possible combinations *within* animals:
    sessions_cartesian = list(itertools.product(*list(dict_combinations.values())))

    ###### STEP 2: permutations
    print("PERMUTATIONS")
    permutation_dfs = []
    for permutation in tqdm.tqdm(sessions_cartesian):
        # assign animal ids to each session combination within each permutation of combinations
        permutation_dict = dict(zip(dict_combinations.keys(), permutation))
        permutation_df = None
        for animal in list(dict_combinations.keys()):
            animal_df = animals_sessions.set_index("animal id")  # create index out of animal id for easier indexing
            sessions_df = animal_df.loc[[animal]]  # filter sessions only for current animal
            permutation_sessions_df = sessions_df.iloc[list(permutation_dict[animal])]  # select only current permutation of sessions within animal
            permutation_df = grow_df(permutation_df, permutation_sessions_df)  # append this selection to a dataframe
        permutation_dfs.append(permutation_df)

    ################
    ### RUN ANALYSES
    print("ANALYSES")
    tests_df = None
    for i_perm in tqdm.tqdm(range(len(permutation_dfs))):
        permutation_df = permutation_dfs[i_perm]
        mpl.rcParams.update(mpl.rcParamsDefault)

        pfs_perm = stat.pfs_df.merge(permutation_df, on="session id")
        shift_gain_df_perm = stat.shift_gain_df.merge(permutation_df, on="session id")
        stat_perm = deepcopy(stat)
        stat_perm.pfs_df = pfs_perm
        stat_perm.shift_gain_df = shift_gain_df_perm

        stat_perm._use_font()
        stat_perm.output_root = f"{stat.output_root}/permutations/{i_perm}/"
        makedir_if_needed(f"{stat.output_root}/permutations/")
        makedir_if_needed(f"{stat.output_root}/permutations/{i_perm}/")

        #stat_perm.plot_place_fields()
        #stat_perm.plot_place_fields_by_session()
        #stat_perm.plot_place_field_proportions()
        try:
            stat_perm.plot_shift_gain_distribution(without_transient=False)
        except ValueError: # happens during tests if too few pfs
            print(f"error during permutation {i_perm}; skipping")
            continue
        stat_perm.tests_df["permutation"] = i_perm
        tests_df = grow_df(tests_df, stat_perm.tests_df)

    pass
    #tests_df[tests_df["feature"] == "log10(formation gain)"][["permutation", "population", "test", "p-value"]].hist(
    #    by="test", column="p-value")
    features = ["initial shift", "log10(formation gain)"]
    m = sys.float_info.min  # smallest representable float in python -- for logx lim
    for feature in features:
        mpl.rcParams.update(mpl.rcParamsDefault)
        tests_df_feature = tests_df[tests_df["feature"] == feature][["permutation", "population", "test", "p-value"]]
        fig, axs = plt.subplots(3,3)

        # reliables
        tests_pop = tests_df_feature[tests_df_feature["population"] == "reliables"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "mann-whitney u"], x="p-value", ax=axs[0,0], binwidth=0.05, binrange=(0,1)); axs[0,0].set_title("mann-whitney u")
        #tests_pop[tests_pop["test"] == "kolmogorov-smirnov"].plot(kind="hist", column="p-value", ax=axs[0,1]); axs[0,1].set_title("kolmogorov-smirnov")
        sns.histplot(data=tests_pop[tests_pop["test"] == "kolmogorov-smirnov"], x="p-value", ax=axs[0,1], binwidth=0.05, binrange=(0,1)); axs[0,1].set_title("kolmogorov-smirnov")
        sns.histplot(data=tests_pop[tests_pop["test"] == "t-test"], x="p-value", ax=axs[0,2], binwidth=0.05, binrange=(0,1)); axs[0,2].set_title("t-test")
        axs[0,0].set_ylabel("reliable PFs")

        # newly formed
        tests_pop = tests_df_feature[tests_df_feature["population"] == "newly formed"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "wilcoxon"], x="p-value", ax=axs[1,0], binwidth=0.05, binrange=(0,1)); axs[1,0].set_title("wilcoxon")
        sns.histplot(data=tests_pop[tests_pop["test"] == "1-sample t-test"], x="p-value", ax=axs[1,1], binwidth=0.05, binrange=(0,1)); axs[1,1].set_title("1-sample t-test")
        axs[1,0].set_ylabel("newly formed PFs")

        # established
        tests_pop = tests_df_feature[tests_df_feature["population"] == "established"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "wilcoxon"], x="p-value", ax=axs[2,0], binwidth=0.05, binrange=(0,1)); axs[2,0].set_title("wilcoxon")
        sns.histplot(data=tests_pop[tests_pop["test"] == "1-sample t-test"], x="p-value", ax=axs[2,1], binwidth=0.05, binrange=(0,1)); axs[2,1].set_title("1-sample t-test")
        axs[2,0].set_ylabel("established PFs")

        for x in range(3):
            for y in range(3):
               axs[x, y].axvline(0.05, color="red", linewidth=1.5)

        axs[1,2].set_axis_off()
        axs[2,2].set_axis_off()
        plt.suptitle(feature)
        plt.tight_layout()
        plt.savefig(f"{stat.output_root}/permutations/pvalues_{feature}.pdf")
        plt.close()

#normalize_session_numbers("CA1", extra_info_CA1, selection_size=3)
normalize_session_numbers("CA3", extra_info_CA3, selection_size=5)
