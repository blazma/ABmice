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
        fig, axs = plt.subplots(3,3,sharex=True)

        # reliables
        tests_pop = tests_df_feature[tests_df_feature["population"] == "reliables"]
        #sns.histplot(data=tests_pop[tests_pop["test"] == "mann-whitney u"], x="p-value", ax=axs[0,0], binwidth=0.05, binrange=(0,1)); axs[0,0].set_title("mann-whitney u")
        sns.histplot(data=tests_pop[tests_pop["test"] == "mann-whitney u"], x="p-value", ax=axs[0, 0]); axs[0, 0].set_title("mann-whitney u")
        #tests_pop[tests_pop["test"] == "kolmogorov-smirnov"].plot(kind="hist", column="p-value", ax=axs[0,1]); axs[0,1].set_title("kolmogorov-smirnov")
        sns.histplot(data=tests_pop[tests_pop["test"] == "kolmogorov-smirnov"], x="p-value", ax=axs[0,1]); axs[0,1].set_title("kolmogorov-smirnov")
        sns.histplot(data=tests_pop[tests_pop["test"] == "t-test"], x="p-value", ax=axs[0,2]); axs[0,2].set_title("t-test")
        axs[0,0].set_ylabel("reliable PFs")

        # newly formed
        tests_pop = tests_df_feature[tests_df_feature["population"] == "newly formed"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "wilcoxon"], x="p-value", ax=axs[1,0]); axs[1,0].set_title("wilcoxon")
        sns.histplot(data=tests_pop[tests_pop["test"] == "1-sample t-test"], x="p-value", ax=axs[1,1]); axs[1,1].set_title("1-sample t-test")
        axs[1,0].set_ylabel("newly formed PFs")

        # established
        tests_pop = tests_df_feature[tests_df_feature["population"] == "established"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "wilcoxon"], x="p-value", ax=axs[2,0]); axs[2,0].set_title("wilcoxon")
        sns.histplot(data=tests_pop[tests_pop["test"] == "1-sample t-test"], x="p-value", ax=axs[2,1]); axs[2,1].set_title("1-sample t-test")
        axs[2,0].set_ylabel("established PFs")

        for x in range(3):
            for y in range(3):
               axs[x, y].axvline(np.log10(0.05), color="red", linewidth=1.5)

        axs[1,2].set_axis_off()
        axs[2,2].set_axis_off()
        plt.suptitle(feature)
        plt.tight_layout()
        plt.savefig(f"{stat.output_root}/permutations/pvalues_{feature}_log.pdf")
        plt.close()

def normalize_cell_numbers(area, extra_info, selection_size, n_runs):
    stat = BtspStatistics(area, data_root, output_root, extra_info, is_shift_criterion_on=True, is_notebook=False)
    stat.load_data()
    stat.filter_low_behavior_score()
    stat.calc_place_field_proportions()
    stat.calc_shift_gain_distribution(unit="cm")
    stat.pfs_df["area"] = area
    stat.shift_gain_df["area"] = area

    sessions = stat.pfs_df["session id"].unique()

    tests_df = None
    rng = np.random.default_rng(1234)
    for i_subset in tqdm.tqdm(range(n_runs)):
        stat_subset = deepcopy(stat)
        stat_subset.pfs_df = None

        for session in sessions:
            pfs = stat.pfs_df
            pfs = pfs[pfs["session id"] == session]

            cells = pfs["cell id"].unique()
            if selection_size > len(cells):
                cells_subset = cells
            else:
                cells_subset = rng.choice(cells, size=selection_size, replace=False)

            pfs_subset = pfs[pfs["cell id"].isin(cells_subset)]
            stat_subset.pfs_df = grow_df(stat_subset.pfs_df, pfs_subset)

        stat_subset.calc_shift_gain_distribution(unit="cm")
        stat_subset.output_root = f"{stat.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/test_results/{i_subset}/"
        makedir_if_needed(f"{stat.output_root}/cell_subsets/")
        makedir_if_needed(f"{stat.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}")
        makedir_if_needed(f"{stat.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/test_results")
        makedir_if_needed(f"{stat.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/test_results/{i_subset}")

        try:
            sg_df = stat_subset.shift_gain_df
            stat_subset.run_tests(sg_df, params=["initial shift", "log10(formation gain)"])
        except ValueError: # happens during tests if too few pfs
            print(f"error during permutation {i_subset}; skipping")
            continue
        stat_subset.tests_df["subset"] = i_subset
        tests_df = grow_df(tests_df, stat_subset.tests_df)

    features = ["initial shift", "log10(formation gain)"]
    for feature in features:
        mpl.rcParams.update(mpl.rcParamsDefault)
        tests_df_feature = tests_df[tests_df["feature"] == feature][["subset", "population", "test", "p-value"]]
        fig, axs = plt.subplots(3,3,sharex=True)
        fig_cum, axs_cum = plt.subplots(3,3,sharex=True, sharey=True)

        ### RELIABLES
        tests_pop = tests_df_feature[tests_df_feature["population"] == "reliables"]
        ### log bs:
        #sns.histplot(data=tests_pop[tests_pop["test"] == "mann-whitney u"], x="p-value", ax=axs[0, 0]); axs[0, 0].set_title("mann-whitney u")
        sns.histplot(data=tests_pop[tests_pop["test"] == "mann-whitney u"], x="p-value", ax=axs[0,0], binwidth=0.05, binrange=(0,1))
        sns.histplot(data=tests_pop[tests_pop["test"] == "kolmogorov-smirnov"], x="p-value", ax=axs[0,1], binwidth=0.05, binrange=(0,1))
        sns.histplot(data=tests_pop[tests_pop["test"] == "t-test"], x="p-value", ax=axs[0,2], binwidth=0.05, binrange=(0,1))

        axs[0,0].set_title("mann-whitney u")
        axs[0,1].set_title("kolmogorov-smirnov")
        axs[0,2].set_title("t-test")
        axs[0,0].set_ylabel("reliable PFs")

        # cumulative plots
        axs_cum[0,0].hist(tests_pop[tests_pop["test"] == "mann-whitney u"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")
        axs_cum[0,1].hist(tests_pop[tests_pop["test"] == "kolmogorov-smirnov"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")
        axs_cum[0,2].hist(tests_pop[tests_pop["test"] == "t-test"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")

        axs_cum[0,0].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))
        axs_cum[0,1].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))
        axs_cum[0,2].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))

        axs_cum[0,0].set_title("mann-whitney u")
        axs_cum[0,1].set_title("kolmogorov-smirnov")
        axs_cum[0,2].set_title("t-test")
        axs_cum[0,0].set_ylabel("reliable PFs")

        ### NEWLY FORMED
        tests_pop = tests_df_feature[tests_df_feature["population"] == "newly formed"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "wilcoxon"], x="p-value", ax=axs[1,0], binwidth=0.05, binrange=(0,1))
        sns.histplot(data=tests_pop[tests_pop["test"] == "1-sample t-test"], x="p-value", ax=axs[1,1], binwidth=0.05, binrange=(0,1))

        axs[1,0].set_title("wilcoxon")
        axs[1,1].set_title("1-sample t-test")
        axs[1,0].set_ylabel("newly formed PFs")

        # cumulative plots
        axs_cum[1,0].hist(tests_pop[tests_pop["test"] == "wilcoxon"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")
        axs_cum[1,1].hist(tests_pop[tests_pop["test"] == "1-sample t-test"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")

        axs_cum[1,0].set_title("wilcoxon")
        axs_cum[1,1].set_title("1-sample t-test")
        axs_cum[1,0].set_ylabel("newly formed PFs")

        axs_cum[1,0].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))
        axs_cum[1,1].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))

        # ESTABLISHED
        tests_pop = tests_df_feature[tests_df_feature["population"] == "established"]
        sns.histplot(data=tests_pop[tests_pop["test"] == "wilcoxon"], x="p-value", ax=axs[2,0], binwidth=0.05, binrange=(0,1)); axs[2,0].set_title("wilcoxon")
        sns.histplot(data=tests_pop[tests_pop["test"] == "1-sample t-test"], x="p-value", ax=axs[2,1], binwidth=0.05, binrange=(0,1)); axs[2,1].set_title("1-sample t-test")
        axs[2,0].set_ylabel("established PFs")

        # cumulative plots
        axs_cum[2,0].hist(tests_pop[tests_pop["test"] == "wilcoxon"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")
        axs_cum[2,1].hist(tests_pop[tests_pop["test"] == "1-sample t-test"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")

        axs_cum[2,0].set_title("wilcoxon")

        axs_cum[2,1].set_title("1-sample t-test")
        axs_cum[2,0].set_ylabel("newly formed PFs")

        axs_cum[2,0].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))
        axs_cum[2,1].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))

        for x in range(3):
            for y in range(3):
               #axs[x, y].axvline(np.log10(0.05), color="red", linewidth=1.5)
               axs[x, y].axvline(0.05, color="red", linewidth=1.5)
               axs_cum[x, y].axvline(0.05, color="red", linewidth=1.5)

        axs[1,2].set_axis_off()
        axs[2,2].set_axis_off()

        axs_cum[1,2].set_axis_off()
        axs_cum[2,2].set_axis_off()

        fig.suptitle(feature)
        fig.tight_layout()

        fig_cum.suptitle(feature)
        fig_cum.tight_layout()

        fig.savefig(f"{stat.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/pvalues_{feature}_runs_{n_runs}_cells_{selection_size}.pdf")
        fig_cum.savefig(f"{stat.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/pvalues_{feature}_runs_{n_runs}_cells_{selection_size}_CUMULATIVE.pdf")

        plt.close(fig)
        plt.close(fig_cum)

def normalize_cell_numbers_both_areas(extra_info_CA1, extra_info_CA3, selection_size, n_subsets):

    def create_subset(stat):
        stat_subset = deepcopy(stat)
        stat_subset.pfs_df = None

        sessions = stat.pfs_df["session id"].unique()
        for session in sessions:
            pfs = stat.pfs_df
            pfs = pfs[pfs["session id"] == session]

            cells = pfs["cell id"].unique()
            if selection_size > len(cells):
                cells_subset = cells
            else:
                cells_subset = rng.choice(cells, size=selection_size, replace=False)

            pfs_subset = pfs[pfs["cell id"].isin(cells_subset)]
            stat_subset.pfs_df = grow_df(stat_subset.pfs_df, pfs_subset)

        stat_subset.calc_shift_gain_distribution(unit="cm")
        return stat_subset

    CA1_stat = BtspStatistics("CA1", data_root, output_root, extra_info_CA1, is_shift_criterion_on=True, is_notebook=False)
    CA1_stat.load_data()
    CA1_stat.filter_low_behavior_score()
    CA1_stat.calc_shift_gain_distribution(unit="cm")
    CA1_stat.pfs_df["area"] = "CA1"
    CA1_stat.shift_gain_df["area"] = "CA1"

    CA3_stat = BtspStatistics("CA3", data_root, output_root, extra_info_CA3, is_shift_criterion_on=True, is_notebook=False)
    CA3_stat.load_data()
    CA3_stat.filter_low_behavior_score()
    CA3_stat.calc_shift_gain_distribution(unit="cm")
    CA3_stat.pfs_df["area"] = "CA3"
    CA3_stat.shift_gain_df["area"] = "CA3"

    makedir_if_needed(f"{CA1_stat.output_root}/cell_subsets_both_areas/")
    makedir_if_needed(f"{CA1_stat.output_root}/cell_subsets_both_areas/subsets_{n_subsets}_cells_{selection_size}")
    rng = np.random.default_rng(1234)

    CA1_pf_subsets = None
    CA1_sg_subsets = None
    CA3_pf_subsets = None
    CA3_sg_subsets = None
    for i_subset in tqdm.tqdm(range(n_subsets)):
        CA1_subset = create_subset(CA1_stat)
        CA1_subset.pfs_df["area"] = "CA1"
        CA1_subset.shift_gain_df["area"] = "CA1"
        CA1_subset.pfs_df["subset"] = i_subset
        CA1_pf_subsets = grow_df(CA1_pf_subsets, CA1_subset.pfs_df)
        CA1_subset.shift_gain_df["subset"] = i_subset
        CA1_sg_subsets = grow_df(CA1_sg_subsets, CA1_subset.shift_gain_df)

        CA3_subset = create_subset(CA3_stat)
        CA3_subset.pfs_df["area"] = "CA3"
        CA3_subset.shift_gain_df["area"] = "CA3"
        CA3_subset.pfs_df["subset"] = i_subset
        CA3_pf_subsets = grow_df(CA3_pf_subsets, CA3_subset.pfs_df)
        CA3_subset.shift_gain_df["subset"] = i_subset
        CA3_sg_subsets = grow_df(CA3_sg_subsets, CA3_subset.shift_gain_df)

    tests_df = None
    perm_idx = 0
    for i_CA1 in tqdm.tqdm(range(n_subsets)):
        for i_CA3 in range(n_subsets):
            CA1_pf_subset = CA1_pf_subsets[CA1_pf_subsets["subset"] == i_CA1]
            CA1_sg_subset = CA1_sg_subsets[CA1_sg_subsets["subset"] == i_CA1]
            CA3_pf_subset = CA3_pf_subsets[CA3_pf_subsets["subset"] == i_CA3]
            CA3_sg_subset = CA3_sg_subsets[CA3_sg_subsets["subset"] == i_CA3]

            pfs_df = pd.concat([CA1_pf_subset, CA3_pf_subset]).reset_index(drop=True)
            shift_gain_df = pd.concat([CA1_sg_subset, CA3_sg_subset]).reset_index(drop=True)

            nf_df = shift_gain_df[shift_gain_df["newly formed"] == True]

            shift_mw = scipy.stats.mannwhitneyu(nf_df[nf_df["area"] == "CA1"]["initial shift"].values, nf_df[nf_df["area"] == "CA3"]["initial shift"].values)
            shift_ks = scipy.stats.kstest(nf_df[nf_df["area"] == "CA1"]["initial shift"].values, cdf=nf_df[nf_df["area"] == "CA3"]["initial shift"].values)

            gain_mw = scipy.stats.mannwhitneyu(nf_df[nf_df["area"] == "CA1"]["log10(formation gain)"].values, nf_df[nf_df["area"] == "CA3"]["log10(formation gain)"].values)
            gain_ks = scipy.stats.kstest(nf_df[nf_df["area"] == "CA1"]["log10(formation gain)"].values, cdf=nf_df[nf_df["area"] == "CA3"]["log10(formation gain)"].values)

            test_dict = {
                "feature": [],
                "test": [],
                # "statistic": [],
                "p-value": [],
                # "log p-value": []
            }
            test_dict["feature"] = ["initial shift", "initial shift", "log10(formation gain)", "log10(formation gain)"]
            test_dict["test"] = ["mann-whitney u", "kolmogorov-smirnov", "mann-whitney u", "kolmogorov-smirnov"]
            test_dict["p-value"] = [shift_mw.pvalue, shift_ks.pvalue, gain_mw.pvalue, gain_ks.pvalue]
            test_df = pd.DataFrame.from_dict(test_dict)
            test_df["permutation"] = perm_idx
            tests_df = grow_df(tests_df, test_df)

            perm_idx += 1

    features = ["initial shift", "log10(formation gain)"]
    for feature in features:
        mpl.rcParams.update(mpl.rcParamsDefault)
        tests_df_feat = tests_df[tests_df["feature"] == feature][["permutation", "test", "p-value"]]
        fig, axs = plt.subplots(2,2,sharex=True)

        ### NEWLY FORMED CA1 vs CA3
        sns.histplot(data=tests_df_feat[tests_df_feat["test"] == "mann-whitney u"], x="p-value", ax=axs[0,0], binwidth=0.05, binrange=(0,1))
        sns.histplot(data=tests_df_feat[tests_df_feat["test"] == "kolmogorov-smirnov"], x="p-value", ax=axs[0,1], binwidth=0.05, binrange=(0,1))

        axs[0,0].set_title("mann-whitney u")
        axs[0,1].set_title("kolmogorov-smirnov")
        axs[0,0].set_ylabel("newly formed (CA1 vs CA3)")

        # cumulative plots
        axs[1,0].hist(tests_df_feat[tests_df_feat["test"] == "mann-whitney u"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")
        axs[1,1].hist(tests_df_feat[tests_df_feat["test"] == "kolmogorov-smirnov"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")

        axs[1,0].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))
        axs[1,1].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))

        for x in range(2):
            for y in range(2):
               #axs[x, y].axvline(np.log10(0.05), color="red", linewidth=1.5)
               axs[x, y].axvline(0.05, color="red", linewidth=1.5)

        fig.suptitle(feature)
        fig.tight_layout()
        fig.savefig(f"{CA1_stat.output_root}/cell_subsets_both_areas/subsets_{n_subsets}_cells_{selection_size}/pvalues_{feature}_subsets_{n_subsets}_cells_{selection_size}.pdf")
        plt.close(fig)

#normalize_cell_numbers("CA1", extra_info_CA1, selection_size=60, n_runs=100)
#normalize_cell_numbers("CA1", extra_info_CA1, selection_size=60, n_runs=1000)
#normalize_cell_numbers("CA1", extra_info_CA1, selection_size=15, n_runs=100)
#normalize_cell_numbers("CA1", extra_info_CA1, selection_size=15, n_runs=1000)

#normalize_cell_numbers("CA3", extra_info_CA3, selection_size=15, n_runs=100)
#normalize_cell_numbers("CA3", extra_info_CA3, selection_size=15, n_runs=1000)

normalize_cell_numbers_both_areas(extra_info_CA1, extra_info_CA3, selection_size=60, n_subsets=100)
