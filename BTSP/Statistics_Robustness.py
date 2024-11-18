import numpy as np
import openpyxl
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
extra_info = ""


class Statistics_Robustness:
    def __init__(self, data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info):
        self.extra_info = "" if not extra_info else f"_{extra_info}"

        self.CA1_stat = BtspStatistics("CA1", data_root, output_root, extra_info_CA1, is_shift_criterion_on=True, is_notebook=False)
        self.CA1_stat.load_data()
        self.CA1_stat.filter_low_behavior_score()
        self.CA1_stat.calc_shift_gain_distribution(unit="cm")
        self.CA1_stat.pfs_df["area"] = "CA1"
        self.CA1_stat.shift_gain_df["area"] = "CA1"

        self.CA3_stat = BtspStatistics("CA3", data_root, output_root, extra_info_CA3, is_shift_criterion_on=True, is_notebook=False)
        self.CA3_stat.load_data()
        self.CA3_stat.filter_low_behavior_score()
        self.CA3_stat.calc_shift_gain_distribution(unit="cm")
        self.CA3_stat.pfs_df["area"] = "CA3"
        self.CA3_stat.shift_gain_df["area"] = "CA3"

        self.output_root = f"{output_root}//statistics//Robustness{self.extra_info}"
        makedir_if_needed(self.output_root)

        self.pfs_df = pd.concat([self.CA1_stat.pfs_df, self.CA3_stat.pfs_df]).reset_index(drop=True)
        self.shift_gain_df = pd.concat([self.CA1_stat.shift_gain_df, self.CA3_stat.shift_gain_df]).reset_index(drop=True)
        self.tests_normalize_by_sessions = None
        self.tests_normalize_by_cells = None

        # run parameters
        self.selection_sizes_normalize_by_sessions = []  # (n_CA1, n_CA3)
        self.selection_sizes_normalize_by_cells = []
        self.n_runs_normalize_by_cells = []

    def __combine_sessions(self, stat_obj, selection_size):
        ###### STEP 1: combinations
        print("COMBINATIONS")

        animals_sessions = stat_obj.pfs_df[["animal id", "session id"]].groupby(["animal id", "session id"]).count().reset_index()
        animals_sessions_counts = animals_sessions.groupby(["animal id"]).count()
        dict_animals_sessions_counts = animals_sessions_counts.to_dict()["session id"]  # number of sessions per animal

        # store session combinations within one animal
        combinations_dict = {}
        for animal, count in tqdm.tqdm(dict_animals_sessions_counts.items()):
            # creates combinations of size 3 out of index lists of sessions within one animal:
            comb = list(itertools.combinations(range(count), selection_size))
            if comb:
                combinations_dict[animal] = comb
            else:
                # if number of sessions < selection_size, itertools.combinations returns empty list, so we store just
                # a list of the session indices within one animal:
                combinations_dict[animal] = (list(range(count)),)
        return combinations_dict

    def __permutate_combinations(self, stat_obj, combinations_dict):
        animals_sessions = stat_obj.pfs_df[["animal id", "session id"]].groupby(["animal id", "session id"]).count().reset_index()

        # creates a cartesian product of all possible combinations of sessions *across* animals using previously
        # created possible combinations *within* animals:
        combinations_cartesian = list(itertools.product(*list(combinations_dict.values())))

        ###### STEP 2: permutations
        print("PERMUTATIONS")
        permutation_dfs = []
        for permutation in tqdm.tqdm(combinations_cartesian):
            # assign animal ids to each session combination within each permutation of combinations
            permutation_dict = dict(zip(combinations_dict.keys(), permutation))
            permutation_df = None
            for animal in list(combinations_dict.keys()):
                animal_df = animals_sessions.set_index("animal id")  # create index out of animal id for easier indexing
                sessions_df = animal_df.loc[[animal]]  # filter sessions only for current animal
                permutation_sessions_df = sessions_df.iloc[
                    list(permutation_dict[animal])]  # select only current permutation of sessions within animal
                permutation_df = grow_df(permutation_df,
                                         permutation_sessions_df)  # append this selection to a dataframe
            permutation_dfs.append(permutation_df)
        return permutation_dfs

    def __run_tests_normalize_by_sessions(self, stat_obj, permutation_dfs, generate_plots):
        tests_df = None
        for i_perm in tqdm.tqdm(range(len(permutation_dfs))):
            permutation_df = permutation_dfs[i_perm]
            mpl.rcParams.update(mpl.rcParamsDefault)

            pfs_perm = stat_obj.pfs_df.merge(permutation_df, on="session id")
            shift_gain_df_perm = stat_obj.shift_gain_df.merge(permutation_df, on="session id")
            stat_perm = deepcopy(stat_obj)
            stat_perm.pfs_df = pfs_perm
            stat_perm.shift_gain_df = shift_gain_df_perm

            stat_perm._use_font()
            stat_perm.output_root = f"{stat_obj.output_root}/permutations/{i_perm}/"
            makedir_if_needed(f"{stat_obj.output_root}/permutations/")
            makedir_if_needed(f"{stat_obj.output_root}/permutations/{i_perm}/")

            try:
                stat_perm.run_tests(stat_perm.shift_gain_df, params=["initial shift", "log10(formation gain)"])
                if generate_plots:
                    stat_perm.plot_shift_gain_distribution(without_transient=False)
            except ValueError:  # happens during tests if too few pfs
                print(f"error during permutation {i_perm}; skipping")
                continue
            stat_perm.tests_df["permutation"] = i_perm
            tests_df = grow_df(tests_df, stat_perm.tests_df)
        return tests_df

    def calc_normalize_by_sessions(self, selection_sizes, generate_plots=False):
        """
        selection_sizes: max. number of sessions for each area, e.g [5, 3]
        """
        self.selection_sizes_normalize_by_sessions = selection_sizes
        for i_area, stat_obj in enumerate([self.CA1_stat, self.CA3_stat]):
            # step 1: create combinations of sessions with a fixed size of session subsample
            combinations_dict = self.__combine_sessions(stat_obj, selection_sizes[i_area])

            # step 2: create permutations of these session-combinations which can then be used to run analysis on
            permutation_dfs = self.__permutate_combinations(stat_obj, combinations_dict)

            # step 3: run analyses, tests, plot shift gain distributions in needed
            tests_df = self.__run_tests_normalize_by_sessions(stat_obj, permutation_dfs, generate_plots)
            self.tests_normalize_by_sessions = grow_df(self.tests_normalize_by_sessions, tests_df)
        #self.tests_normalize_by_sessions.to_excel(f"{self.output_root}//tests_normalize_by_sessions.xlsx")

    def __run_tests_normalize_by_cells(self, rng, stat_obj, selection_size):
        sessions = stat_obj.pfs_df["session id"].unique()

        stat_subset = deepcopy(stat_obj)
        stat_subset.pfs_df = None
        for session in sessions:
            pfs = stat_obj.pfs_df
            pfs = pfs[pfs["session id"] == session]

            cells = pfs["cell id"].unique()
            if selection_size > len(cells):
                cells_subset = cells
            else:
                cells_subset = rng.choice(cells, size=selection_size, replace=False)

            pfs_subset = pfs[pfs["cell id"].isin(cells_subset)]
            stat_subset.pfs_df = grow_df(stat_subset.pfs_df, pfs_subset)

        stat_subset.calc_shift_gain_distribution(unit="cm")
        #stat_subset.output_root = f"{stat_subset.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/test_results/{i_subset}/"
        #makedir_if_needed(f"{stat_subset.output_root}/cell_subsets/")
        #makedir_if_needed(f"{stat_subset.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}")
        #makedir_if_needed(f"{stat_subset.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/test_results")
        #makedir_if_needed(f"{stat_subset.output_root}/cell_subsets/runs_{n_runs}_cells_{selection_size}/test_results/{i_subset}")

        sg_df = stat_subset.shift_gain_df
        stat_subset.run_tests(sg_df, params=["initial shift", "log10(formation gain)"])
        return stat_subset.tests_df

    def calc_normalize_by_cells(self, selection_sizes, n_runs):
        self.selection_sizes_normalize_by_cells = selection_sizes
        self.n_runs_normalize_by_cells = n_runs
        rng = np.random.default_rng(1234)

        tests_df = None
        for i_area, stat_area in enumerate([self.CA1_stat, self.CA3_stat]):
            for i_subset in tqdm.tqdm(range(n_runs)):
                tests_df_area = self.__run_tests_normalize_by_cells(rng, stat_area, selection_sizes[i_area])
                tests_df_area["subset"] = i_subset
                tests_df = grow_df(tests_df, tests_df_area)
        self.tests_normalize_by_cells = tests_df

    def plot_tests_within_area(self):
        features = ["initial shift", "log10(formation gain)"]
        available_tests_by_pop = {
            "reliables": ["mann-whitney u", "kolmogorov-smirnov", "t-test"],
            "newly formed": ["wilcoxon", "1-sample t-test"],
            "established": ["wilcoxon", "1-sample t-test"],
        }
        run_types_and_test_results = {
            "normalize_by_sessions": self.tests_normalize_by_sessions,
            "normalize_by_cells": self.tests_normalize_by_cells
        }
        for run_type, test_results in run_types_and_test_results.items():
            if type(test_results) == type(None):
                print(f"can't plot test results of '{run_type}' - run calc function first")
                continue

            for feature in features:
                # plot titles and filenames
                if run_type == "normalize_by_sessions":
                    selsize = self.selection_sizes_normalize_by_sessions
                    suptitle = f"{run_type}    {feature}\nn_sessions={selsize}"
                    filename = f"cells_{selsize}_{feature}"
                else:  # normalize_by_cells
                    selsize = self.selection_sizes_normalize_by_cells
                    n_runs = self.n_runs_normalize_by_cells
                    suptitle = f"{run_type}    {feature}\nn_cells={selsize}    n_runs={n_runs}"
                    filename = f"cells_{selsize}_runs_{n_runs}_{feature}"

                group_key = "permutation" if run_type == "normalize_by_sessions" else "subset"  # TODO: not a great solution
                tests_df_feature = test_results[test_results["feature"] == feature][["area", group_key, "population", "test", "p-value"]]
                for area in ["CA1", "CA3"]:
                    fig, axs = plt.subplots(3, 3, sharex=True)

                    mpl.rcParams.update(mpl.rcParamsDefault)
                    tests_df_feature_area = tests_df_feature[tests_df_feature["area"] == area]

                    i_row = 0
                    for pop, available_tests in available_tests_by_pop.items():
                        i_col = 0
                        tests_results_pop = tests_df_feature_area[tests_df_feature_area["population"] == pop]
                        for test in available_tests:
                            # histograms
                            ax = axs[i_row, i_col]
                            sns.histplot(data=tests_results_pop[tests_results_pop["test"] == test], x="p-value", ax=ax,
                                         binwidth=0.05, binrange=(0,1), hue="area", legend=False)
                            ax.set_title(test)
                            if not (i_col == 2 and i_row != 0):  # nf and est. don't have a 3rd test, so no 3rd column
                                ax.axvline(0.05, color="red", linewidth=1.5)
                            i_col += 1
                        axs[i_row, 0].set_ylabel(pop)
                        i_row += 1

                    axs[1, 2].set_axis_off()
                    axs[2, 2].set_axis_off()

                    plt.suptitle(f"{area}    {suptitle}")
                    plt.tight_layout()
                    makedir_if_needed(f"{self.output_root}/plots_{run_type}/")
                    plt.savefig(f"{self.output_root}/plots_{run_type}/pvalues_{area}_{filename}.pdf")
                    plt.close()

                # cumulative plots -- areas plotted on same graph now
                fig_cum, axs_cum = plt.subplots(3, 3, sharex=True, sharey=True)

                i_row = 0
                for pop, available_tests in available_tests_by_pop.items():
                    i_col = 0
                    tests_results_pop = tests_df_feature[tests_df_feature["population"] == pop]
                    for test in available_tests:
                        ax_cum = axs_cum[i_row, i_col]
                        test_results_CA1 = tests_results_pop[(tests_results_pop["test"] == test) & (tests_results_pop["area"] == "CA1")]["p-value"].values
                        test_results_CA3 = tests_results_pop[(tests_results_pop["test"] == test) & (tests_results_pop["area"] == "CA3")]["p-value"].values
                        ax_cum.hist(test_results_CA1, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step", label="CA1")
                        ax_cum.hist(test_results_CA3, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step", label="CA3")
                        ax_cum.set_title(test)
                        if not (i_col == 2 and i_row != 0):  # nf and est. don't have a 3rd test, so no 3rd column
                            ax_cum.axvline(0.05, color="red", linewidth=1.5)
                        i_col += 1
                    axs_cum[i_row, 0].set_ylabel(pop)
                    i_row += 1

                axs_cum[1, 2].set_axis_off()
                axs_cum[2, 2].set_axis_off()
                axs_cum[2, 1].legend()

                plt.suptitle(suptitle)
                plt.tight_layout()
                plt.savefig(f"{self.output_root}/plots_{run_type}/CUM_pvalues_{filename}.pdf")
                plt.close()

    def summarize_results(self):

        def cell_val(sp, pop):
            try:
                val = np.round(sp[(sp["population"] == pop) & (sp["is significant"] == True)]["prop"].values[0],2)
            except IndexError:
                val = 0.0
            return val

        wb = openpyxl.Workbook()

        run_types_and_test_results = {
            "normalize_by_sessions": self.tests_normalize_by_sessions,
            "normalize_by_cells": self.tests_normalize_by_cells
        }
        run_types_and_counts = {
            "normalize_by_sessions": [self.selection_sizes_normalize_by_sessions],
            "normalize_by_cells": [self.selection_sizes_normalize_by_cells, self.n_runs_normalize_by_cells]
        }
        for run_type, test_results in run_types_and_test_results.items():
            wb.create_sheet(run_type)
            ws = wb.get_sheet_by_name(run_type)
            ws.append(["", run_type])
            ws.append(["", "CA1", "CA3"])
            ws.append(["max num. cells/sessions", *run_types_and_counts[run_type][0]])
            if run_type == "normalize_by_cells":
                ws.append(["num. of runs", run_types_and_counts[run_type][1]])
            else:
                ws.append([""])
            ws.append([""])

            df = test_results
            df["is significant"] = df["p-value"] < 0.05
            df = df[(df["test"] == "mann-whitney u") | (df["test"] == "wilcoxon")]

            features = ["initial shift", "log10(formation gain)"]
            for feature in features:
                df_feat = df[df["feature"] == feature]
                summary_df = {
                    "newly formed vs. established": [],
                    "newly formed only": [],
                    "established only": []
                }
                for area in ["CA1", "CA3"]:
                    df_area = df_feat[df_feat["area"] == area]
                    cols = ["population", "feature", "test"]
                    significant_runs = df_area[[*cols, "is significant", "p-value"]].groupby([*cols, "is significant"]).count()
                    total_runs = df_area[[*cols, "p-value"]].groupby(cols).count()
                    significant_proportions = significant_runs / total_runs
                    sp = significant_proportions.reset_index().rename(columns={"p-value": "prop"})

                    summary_df["newly formed vs. established"].append(cell_val(sp, "reliables"))
                    summary_df["newly formed only"].append(cell_val(sp, "newly formed"))
                    summary_df["established only"].append(cell_val(sp, "established"))

                ws.append(["", feature])
                ws.append(["", "CA1", "CA3"])
                ws.append(["newly formed vs. established", *summary_df["newly formed vs. established"]])
                ws.append(["newly formed only", *summary_df["newly formed only"]])
                ws.append(["established only", *summary_df["established only"]])
                ws.append([""])
        wb.remove(wb.get_sheet_by_name("Sheet"))
        suffix = "_sessions_{s}_cells_{c}_runs_{r}".format(s=self.selection_sizes_normalize_by_sessions,
                                                           c=self.selection_sizes_normalize_by_cells,
                                                           r=self.n_runs_normalize_by_cells)
        wb.save(f"{self.output_root}/proportions_of_significant_tests{suffix}.xlsx")

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

#normalize_cell_numbers_both_areas(extra_info_CA1, extra_info_CA3, selection_size=60, n_subsets=100)

stats = Statistics_Robustness(data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info)
stats.calc_normalize_by_sessions(selection_sizes=[3,5])
stats.calc_normalize_by_cells(selection_sizes=[15,15], n_runs=1000)
stats.plot_tests_within_area()
stats.summarize_results()
