import numpy as np
import openpyxl
import seaborn as sns
import pandas as pd
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
from BtspStatistics import BtspStatistics
from Statistics_BothAreas import Statistics_BothAreas
from constants import AREA_PALETTE
import scipy
import tqdm
from copy import deepcopy
from utils import makedir_if_needed, grow_df
import itertools
import warnings
from robustness_params import PARAMS
warnings.filterwarnings("ignore")


class Statistics_Robustness:
    def __init__(self, data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info, save_results=True):
        self.data_root = data_root
        self.extra_info_CA1 = extra_info_CA1
        self.extra_info_CA3 = extra_info_CA3
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.features = ["initial shift", "log10(formation gain)"]

        print("loading CA1 pf data")
        self.CA1_stat = BtspStatistics("CA1", data_root, output_root, extra_info_CA1, is_shift_criterion_on=True, is_notebook=False)
        self.CA1_stat.load_data()
        self.CA1_stat.filter_low_behavior_score()
        self.CA1_stat.calc_shift_gain_distribution(unit="cm")
        self.CA1_stat.pfs_df["area"] = "CA1"
        self.CA1_stat.shift_gain_df["area"] = "CA1"

        print("loading CA3 pf data")
        self.CA3_stat = BtspStatistics("CA3", data_root, output_root, extra_info_CA3, is_shift_criterion_on=True, is_notebook=False)
        self.CA3_stat.load_data()
        self.CA3_stat.filter_low_behavior_score()
        self.CA3_stat.calc_shift_gain_distribution(unit="cm")
        self.CA3_stat.pfs_df["area"] = "CA3"
        self.CA3_stat.shift_gain_df["area"] = "CA3"

        self.output_root = f"{output_root}//statistics//Robustness{self.extra_info}"
        self.save_results = save_results
        if self.save_results:
            makedir_if_needed(self.output_root)

        self.pfs_df = pd.concat([self.CA1_stat.pfs_df, self.CA3_stat.pfs_df]).reset_index(drop=True)
        self.shift_gain_df = pd.concat([self.CA1_stat.shift_gain_df, self.CA3_stat.shift_gain_df]).reset_index(drop=True)
        self.n_CA1_sessions = None
        self.n_CA3_sessions = None

        # store all possible equalizations, their parameters, results (tests_dfs)
        self.equalizations = None # columns: ["... to equalize", "equalize by ...", "parameters", "test results"]

        self.tests_equalize_by_sessions = None
        self.tests_equalize_by_cells = None
        self.tests_equalize_by_cells_across_areas = None
        self.tests_equalize_by_pfs_across_areas = None

        # run parameters
        self.params_df = None
        self.selection_sizes_equalize_by_sessions = []  # (n_CA1, n_CA3)
        self.selection_sizes_equalize_by_cells = []
        self.n_runs_equalize_by_cells = None
        self.n_subsets_equalize_by_cells_both_areas = None
        self.selection_sizes_equalize_by_cells_both_areas = []
        self.n_subsets_equalize_by_pfs_both_areas = None
        self.selection_sizes_equalize_by_pfs_both_areas = []
        self.omission = False

    def calc_params(self):
        ca1 = self.CA1_stat.shift_gain_df
        ca3 = self.CA3_stat.shift_gain_df

        animals_to_ignore = ["srb231", "srb269"]  # very low place cell count: (3, 1)
        ca1 = ca1[~ca1["animal id"].isin(animals_to_ignore)]
        ca3 = ca3[~ca3["animal id"].isin(animals_to_ignore)]

        CA1_AxC = ca1.groupby(["area", "animal id"])["cell id"].nunique()
        CA1_AxPFs = ca1.groupby(["area", "animal id"]).count().mean(axis=1)
        CA1_SxC = ca1.groupby(["area", "animal id", "session id"])["cell id"].nunique()
        CA1_SxPFs = ca1.groupby(["area", "animal id", "session id"]).count().mean(axis=1)

        CA3_AxC = ca3.groupby(["area", "animal id"])["cell id"].nunique()
        CA3_AxPFs = ca3.groupby(["area", "animal id"]).count().mean(axis=1)
        CA3_SxC = ca3.groupby(["area", "animal id", "session id"])["cell id"].nunique()
        CA3_SxPFs = ca3.groupby(["area", "animal id", "session id"]).count().mean(axis=1)

        equalization_types = ["min", "avg", "median", "quartile"]
        min_element_count = 3

        def apply_func(df, eq_type):
            if eq_type == "min":
                val = df.min()
            elif eq_type == "avg":
                val = df.mean()
            elif eq_type == "median":
                val = df.median()
            elif eq_type == "quartile":
                val = df.quantile(q=0.25)
            if val < min_element_count:
                return min_element_count
            else:
                return int(val)

        params_df = None
        for eq_type in equalization_types:
            params_dict_CA1 = {
                "eq_type": eq_type,
                "area": "CA1",
                "AxC": apply_func(CA1_AxC, eq_type),
                "AxPFs": apply_func(CA1_AxPFs, eq_type),
                "SxC": apply_func(CA1_SxC, eq_type),
                "SxPFs": apply_func(CA1_SxPFs, eq_type)
            }
            params_dict_CA3 = {
                "eq_type": eq_type,
                "area": "CA3",
                "AxC": apply_func(CA3_AxC, eq_type),
                "AxPFs": apply_func(CA3_AxPFs, eq_type),
                "SxC": apply_func(CA3_SxC, eq_type),
                "SxPFs": apply_func(CA3_SxPFs, eq_type)
            }
            params_df_CA1 = pd.DataFrame.from_dict([params_dict_CA1])
            params_df_CA3 = pd.DataFrame.from_dict([params_dict_CA3])
            params_eq_type = pd.concat([params_df_CA1, params_df_CA3])
            params_df = grow_df(params_df, params_eq_type)
        self.params_df = params_df

    def equalize(self, eq_type, analysis_type, what_to_eq, eq_by_what, params, export=False):
        print(f"equalizing {what_to_eq} by {eq_by_what} to {eq_type}, {analysis_type}\n")
        funcs = {
            "single area": {
                "animal": {
                    "session": self.calc_equalize_animals_by_sessions,
                    "cells": self.calc_equalize_animals_by_cells,
                    "pfs": self.calc_equalize_animals_by_pfs
                },
                "session": {
                    "cells": self.calc_equalize_sessions_by_cells,
                    "pfs": self.calc_equalize_sessions_by_pfs,
                }
            },
            "both areas": {
                "animal": {
                    "session": self.calc_equalize_animals_by_sessions_both_areas,
                    "cells": self.calc_equalize_animals_by_cells_both_areas,
                    "pfs": self.calc_equalize_animals_by_pfs_both_areas
                },
                "session": {
                    "cells": self.calc_equalize_sessions_by_cells_both_areas,  # TODO: to be split by cells vs pfs
                    "pfs": self.calc_equalize_sessions_by_pfs_both_areas
                }
            },
        }
        func = funcs[analysis_type][what_to_eq][eq_by_what]
        tests = func(params)
        equalization_dict = {
             "analysis type": [analysis_type],
             "what to equalize": [what_to_eq],
             "equalize by what": [eq_by_what],
             "equalize to": [eq_type],
             "parameters": [params],
             "test results": [tests]
        }
        equalization_df = pd.DataFrame.from_dict(equalization_dict)
        self.equalizations = grow_df(self.equalizations, equalization_df)
        if export:
            equalization_df.to_pickle(f"{self.output_root}/{analysis_type}_equalize_{what_to_eq}_by_{eq_by_what}_to_{eq_type}.pickle")

    def __combine_sessions(self, stat_obj, selection_size):
        animals_sessions = stat_obj.pfs_df[["animal id", "session id"]].groupby(["animal id", "session id"]).count().reset_index()
        animals_sessions_counts = animals_sessions.groupby(["animal id"]).count()
        dict_animals_sessions_counts = animals_sessions_counts.to_dict()["session id"]  # number of sessions per animal

        # store session combinations within one animal
        combinations_dict = {}
        for animal, count in dict_animals_sessions_counts.items():
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

        permutation_dfs = []
        for permutation in combinations_cartesian:
            # assign animal ids to each session combination within each permutation of combinations
            permutation_dict = dict(zip(combinations_dict.keys(), permutation))
            permutation_df = None
            for animal in list(combinations_dict.keys()):
                animal_df = animals_sessions.set_index("animal id")  # create index out of animal id for easier indexing
                sessions_df = animal_df.loc[[animal]]  # filter sessions only for current animal
                permutation_sessions_df = sessions_df.iloc[list(permutation_dict[animal])]  # select only current permutation of sessions within animal
                permutation_df = grow_df(permutation_df,
                                         permutation_sessions_df)  # append this selection to a dataframe
            permutation_dfs.append(permutation_df)
        return permutation_dfs

    def __run_tests_equalize_by_sessions(self, stat_obj, permutation_dfs, generate_plots):
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
            if self.save_results:
                makedir_if_needed(f"{stat_obj.output_root}/permutations/")
                makedir_if_needed(f"{stat_obj.output_root}/permutations/{i_perm}/")

            try:
                stat_perm.run_tests(stat_perm.shift_gain_df, params=["initial shift", "log10(formation gain)"])
                if self.save_results and generate_plots:
                    stat_perm.plot_shift_gain_distribution(without_transient=False)
            except ValueError:  # happens during tests if too few pfs
                print(f"error during permutation {i_perm}; skipping")
                continue
            #stat_perm.tests_df["permutation"] = i_perm
            tests_df = grow_df(tests_df, stat_perm.tests_df)
        return tests_df

    def __equalize_by_sessions(self, is_both_areas, selection_sizes, generate_plots):
        tests_equalize_animals_by_sessions = None
        permutations_dfs_both_areas = []
        for i_area, stat_obj in enumerate([self.CA1_stat, self.CA3_stat]):
            # step 1: create combinations of sessions with a fixed size of session subsample
            combinations_dict = self.__combine_sessions(stat_obj, selection_sizes[i_area])

            # step 2: create permutations of these session-combinations which can then be used to run analysis on
            permutation_dfs = self.__permutate_combinations(stat_obj, combinations_dict)
            permutations_dfs_both_areas.append(permutation_dfs)

            # step 3: run analyses, tests, plot shift gain distributions in needed
            if is_both_areas:
                continue  # we handle this step separately
            tests_df = self.__run_tests_equalize_by_sessions(stat_obj, permutation_dfs, generate_plots)
            tests_equalize_animals_by_sessions = grow_df(tests_equalize_animals_by_sessions, tests_df)

        # step 3 for both areas: run tests -- it is 'outsourced' to Statistics_BothAreas object
        if is_both_areas:
            init_params = [self.data_root, self.output_root, self.extra_info_CA1, self.extra_info_CA3, self.extra_info]
            stat_both_areas = Statistics_BothAreas(*init_params, create_output_folder=False)
            sg_df = pd.concat([*permutations_dfs_both_areas[0], *permutations_dfs_both_areas[1]])  # 0 = CA1, 1 = CA3
            permut_dfs_CA1 = permutations_dfs_both_areas[0]
            permut_dfs_CA3 = permutations_dfs_both_areas[1]
            for permut_df_CA1 in tqdm.tqdm(permut_dfs_CA1):
                for permut_df_CA3 in permut_dfs_CA3:
                    sg_df_CA1 = self.CA1_stat.shift_gain_df.merge(permut_df_CA1, on="session id")
                    sg_df_CA3 = self.CA3_stat.shift_gain_df.merge(permut_df_CA3, on="session id")
                    sg_df = pd.concat([sg_df_CA1, sg_df_CA3]).reset_index(drop=True)

                    stat_both_areas.shift_gain_df = sg_df
                    stat_both_areas.run_tests(save_results=False)
                    tests_equalize_animals_by_sessions = grow_df(tests_equalize_animals_by_sessions, stat_both_areas.tests_df)
        return tests_equalize_animals_by_sessions

    def calc_equalize_animals_by_sessions(self, params, generate_plots=False):
        """
        selection_sizes: max. number of sessions for each area, e.g [5, 3]
        """
        selection_sizes = [params["max sessions CA1"], params["max sessions CA3"]]
        self.selection_sizes_equalize_by_sessions = selection_sizes  # TODO: obsolete
        tests_df = self.__equalize_by_sessions(is_both_areas=False,
                                               selection_sizes=selection_sizes,
                                               generate_plots=generate_plots)
        self.tests_equalize_by_sessions = tests_df  # TODO: obsolete
        return tests_df

    def calc_equalize_animals_by_sessions_both_areas(self, params, generate_plots=False):
        """
        selection_sizes: max. number of sessions for each area, e.g [5, 3]
        """
        selection_sizes = [params["max sessions CA1"], params["max sessions CA3"]]
        #self.selection_sizes_equalize_by_sessions = selection_sizes  # TODO: obsolete
        tests_df = self.__equalize_by_sessions(is_both_areas=True,
                                               selection_sizes=selection_sizes,
                                               generate_plots=generate_plots)
        #self.tests_equalize_by_sessions = tests_df  # TODO: obsolete
        return tests_df

    def __equalize_by_cells(self, rng, stat, object_type, selection_size):
        objects = stat.pfs_df[f"{object_type} id"].unique()

        stat_subset = deepcopy(stat)
        stat_subset.pfs_df = None
        for object in objects:  # object may be animal or session
            pfs = stat.pfs_df
            pfs = pfs[pfs[f"{object_type} id"] == object]

            cells = pfs["cell id"].unique()
            if len(cells) < 5:
                # TODO: keep track of skipped objects
                continue
            if selection_size > len(cells):
                cells_subset = cells
            else:
                cells_subset = rng.choice(cells, size=selection_size, replace=False)

            pfs_subset = pfs[pfs["cell id"].isin(cells_subset)]
            stat_subset.pfs_df = grow_df(stat_subset.pfs_df, pfs_subset)

        stat_subset.calc_shift_gain_distribution(unit="cm")
        return stat_subset

    def __equalize_by_pfs(self, rng, stat, object_type, selection_size):
        """
        stat_obj : BtspStatistics object
        selection_size : max number of pfs to be sampled
        omission: if True, omits sessions when fewer pfs than selection_size in subset (may result in very few sessions in CA3)
        """
        objects = stat.pfs_df[f"{object_type} id"].unique()

        stat_subset = deepcopy(stat)
        stat_subset.shift_gain_df = None
        for object in objects:
            sg_df = stat.shift_gain_df
            sg_df = sg_df[sg_df[f"{object_type} id"] == object]

            nf = sg_df[sg_df["newly formed"] == True]
            es = sg_df[sg_df["newly formed"] == False]

            if len(nf) < 5 or len(es) < 5:
                # TODO: keep track of skipped objects
                continue

            if selection_size > len(nf):
                if self.omission:
                    print(f"not enough NF pfs in {object}, omitting")
                    continue
                nf_subset = nf
            else:
                nf_subset = nf.sample(n=selection_size, random_state=rng)

            if selection_size > len(es):
                if self.omission:
                    print(f"not enough ES pfs in {object}, omitting")
                    continue
                es_subset = es
            else:
                es_subset = es.sample(n=selection_size, random_state=rng)

            sg_subset = pd.concat([nf_subset, es_subset])
            stat_subset.shift_gain_df = grow_df(stat_subset.shift_gain_df, sg_subset)
        return stat_subset

    def calc_equalize_sessions_by_cells(self, params):
        self.selection_sizes_equalize_by_cells = [params["max cells CA1"], params["max cells CA3"]]  # TODO: obsolete
        self.n_runs_equalize_by_cells = params["runs"]  # TODO: obsolete

        selection_sizes = [params["max cells CA1"], params["max cells CA3"]]
        n_runs = params["runs"]
        rng = np.random.default_rng(1234)

        tests_df = None
        for i_area, stat_area in enumerate([self.CA1_stat, self.CA3_stat]):
            for i_subset in tqdm.tqdm(range(n_runs)):
                stat_subset = self.__equalize_by_cells(rng,
                                                       stat_area,
                                                       "session",
                                                       selection_sizes[i_area])
                sg_df = stat_subset.shift_gain_df
                stat_subset.run_tests(sg_df, params=["initial shift", "log10(formation gain)"])
                tests_df_area = stat_subset.tests_df
                tests_df_area["subset"] = i_subset
                tests_df = grow_df(tests_df, tests_df_area)
        self.tests_equalize_by_cells = tests_df  # TODO:obsolete
        return tests_df

    def calc_equalize_animals_by_cells(self, params):
        selection_sizes = [params["max cells CA1"], params["max cells CA3"]]
        n_runs = params["runs"]
        rng = np.random.default_rng(1234)

        tests_df = None
        for i_area, stat_area in enumerate([self.CA1_stat, self.CA3_stat]):
            for i_subset in tqdm.tqdm(range(n_runs)):
                stat_subset = self.__equalize_by_cells(rng,
                                                       stat_area,
                                                       "animal",
                                                       selection_sizes[i_area])
                sg_df = stat_subset.shift_gain_df
                stat_subset.run_tests(sg_df, params=["initial shift", "log10(formation gain)"])
                tests_df_area = stat_subset.tests_df
                tests_df_area["subset"] = i_subset
                tests_df = grow_df(tests_df, tests_df_area)
        #self.tests_equalize_by_cells = tests_df  # TODO:obsolete
        return tests_df

    def calc_equalize_animals_by_pfs(self, params):
        selection_sizes = [params["max pfs CA1"], params["max pfs CA3"]]
        n_runs = params["runs"]
        rng = np.random.default_rng(1234)

        tests_df = None
        for i_area, stat_area in enumerate([self.CA1_stat, self.CA3_stat]):
            for i_subset in tqdm.tqdm(range(n_runs)):
                stat_subset = self.__equalize_by_pfs(rng,
                                                     stat_area,
                                                     "animal",
                                                     selection_sizes[i_area])
                sg_df = stat_subset.shift_gain_df
                stat_subset.run_tests(sg_df, params=["initial shift", "log10(formation gain)"])
                tests_df_area = stat_subset.tests_df
                tests_df_area["subset"] = i_subset
                tests_df = grow_df(tests_df, tests_df_area)
        # self.tests_equalize_by_cells = tests_df  # TODO:obsolete
        return tests_df

    def calc_equalize_sessions_by_pfs(self, params):
        selection_sizes = [params["max pfs CA1"], params["max pfs CA3"]]
        n_runs = params["runs"]
        rng = np.random.default_rng(1234)

        tests_df = None
        for i_area, stat_area in enumerate([self.CA1_stat, self.CA3_stat]):
            for i_subset in tqdm.tqdm(range(n_runs)):
                stat_subset = self.__equalize_by_pfs(rng,
                                                     stat_area,
                                                     "session",
                                                     selection_sizes[i_area])
                sg_df = stat_subset.shift_gain_df
                stat_subset.run_tests(sg_df, params=["initial shift", "log10(formation gain)"])
                tests_df_area = stat_subset.tests_df
                tests_df_area["subset"] = i_subset
                tests_df = grow_df(tests_df, tests_df_area)
        # self.tests_equalize_by_cells = tests_df  # TODO:obsolete
        return tests_df

    def handle_both_areas(self, equalize_func, object_type, n_subsets, selection_sizes):
        rng = np.random.default_rng(1234)

        ### create cell (or pf) subsets
        CA1_subsets = []  # list of BtspStatistics objects
        CA3_subsets = []
        for i_subset in range(n_subsets):
            CA1_subset = equalize_func(rng, self.CA1_stat, object_type, selection_sizes[0])
            if CA1_subset is not None:
                CA1_subset.shift_gain_df["subset"] = i_subset
                CA1_subsets.append(CA1_subset)

            CA3_subset = equalize_func(rng, self.CA3_stat, object_type, selection_sizes[1])
            if CA3_subset is not None:
                CA3_subset.shift_gain_df["subset"] = i_subset
                CA3_subsets.append(CA3_subset)
        # TODO: do something about these:
        #self.n_CA1_sessions = len(CA1_subsets[0].shift_gain_df["session id"].unique())
        #self.n_CA3_sessions = len(CA3_subsets[0].shift_gain_df["session id"].unique())

        ### combine cell (or pf) subsets across areas, run tests for each combo
        tests = None
        for CA1_subset in tqdm.tqdm(CA1_subsets):
            for CA3_subset in CA3_subsets:
                shift_gain_combo = pd.concat([CA1_subset.shift_gain_df, CA3_subset.shift_gain_df]).reset_index(drop=True)
                stat_bothAreas = Statistics_BothAreas(self.data_root, self.output_root, self.extra_info_CA1, self.extra_info_CA3,
                                                      self.extra_info, create_output_folder=False)
                stat_bothAreas.shift_gain_df = shift_gain_combo
                stat_bothAreas.run_tests(save_results=False)
                tests_combo = stat_bothAreas.tests_df
                tests = grow_df(tests, tests_combo)
        return tests

    def calc_equalize_sessions_by_cells_both_areas(self, params):
        selection_sizes = [params["max cells CA1"], params["max cells CA3"]]
        n_subsets = params["runs"]
        tests = self.handle_both_areas(self.__equalize_by_cells, "session", n_subsets, selection_sizes)
        self.tests_equalize_by_cells_across_areas = tests
        return tests

    def calc_equalize_sessions_by_pfs_both_areas(self, params):
        selection_sizes = [params["max pfs CA1"], params["max pfs CA3"]]
        n_subsets = params["runs"]
        tests = self.handle_both_areas(self.__equalize_by_pfs, "session", n_subsets, selection_sizes)
        self.tests_equalize_by_pfs_across_areas = tests
        return tests

    def calc_equalize_animals_by_cells_both_areas(self, params):
        selection_sizes = [params["max cells CA1"], params["max cells CA3"]]
        n_subsets = params["runs"]
        tests = self.handle_both_areas(self.__equalize_by_cells, "animal", n_subsets, selection_sizes)
        self.tests_equalize_by_pfs_across_areas = tests
        return tests

    def calc_equalize_animals_by_pfs_both_areas(self, params):
        selection_sizes = [params["max pfs CA1"], params["max pfs CA3"]]
        n_subsets = params["runs"]
        tests = self.handle_both_areas(self.__equalize_by_pfs, "animal", n_subsets, selection_sizes)
        self.tests_equalize_by_pfs_across_areas = tests
        return tests

    def plot_tests_within_area(self):
        available_tests_by_pop = {
            "reliables": ["mann-whitney u", "kolmogorov-smirnov", "t-test"],
            "newly formed": ["wilcoxon", "1-sample t-test"],
            "established": ["wilcoxon", "1-sample t-test"],
        }
        run_types_and_test_results = {
            "equalize_by_sessions": self.tests_equalize_by_sessions,
            "equalize_by_cells": self.tests_equalize_by_cells
        }
        for run_type, test_results in run_types_and_test_results.items():
            if type(test_results) == type(None):
                print(f"can't plot test results of '{run_type}' - run calc function first")
                continue

            for feature in self.features:
                # plot titles and filenames
                if run_type == "equalize_by_sessions":
                    selsize = self.selection_sizes_equalize_by_sessions
                    suptitle = f"{run_type}    {feature}\nn_sessions={selsize}"
                    filename = f"cells_{selsize}_{feature}"
                else:  # equalize_by_cells
                    selsize = self.selection_sizes_equalize_by_cells
                    n_runs = self.n_runs_equalize_by_cells
                    suptitle = f"{run_type}    {feature}\nn_cells={selsize}    n_runs={n_runs}"
                    filename = f"cells_{selsize}_runs_{n_runs}_{feature}"

                group_key = "permutation" if run_type == "equalize_by_sessions" else "subset"  # TODO: not a great solution
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

    def plot_tests_across_areas(self, eq_type):
        if eq_type == "cells":
            n_subsets, selection_sizes = self.n_subsets_equalize_by_cells_both_areas, self.selection_sizes_equalize_by_cells_both_areas
            tests_df = self.tests_equalize_by_cells_across_areas
        elif eq_type == "pfs":
            n_subsets, selection_sizes = self.n_subsets_equalize_by_pfs_both_areas, self.selection_sizes_equalize_by_pfs_both_areas
            tests_df = self.tests_equalize_by_pfs_across_areas
        else:
            print("select proper equalization type")
            return

        makedir_if_needed(f"{self.output_root}/cell_subsets_both_areas/")
        makedir_if_needed(f"{self.output_root}/cell_subsets_both_areas/subsets_{n_subsets}_{eq_type}_{selection_sizes}")

        for feature in self.features:
            mpl.rcParams.update(mpl.rcParamsDefault)
            tests_df_feat = tests_df[tests_df["feature"] == feature][["population", "test", "p-value"]]
            for pop in ["newly formed", "established"]:
                fig, axs = plt.subplots(2,2,sharex=True)

                tests_df_feat_pop = tests_df_feat[tests_df_feat["population"] == pop]
                sns.histplot(data=tests_df_feat_pop[tests_df_feat_pop["test"] == "mann-whitney u"], x="p-value", ax=axs[0,0], binwidth=0.05, binrange=(0,1))
                sns.histplot(data=tests_df_feat_pop[tests_df_feat_pop["test"] == "kolmogorov-smirnov"], x="p-value", ax=axs[0,1], binwidth=0.05, binrange=(0,1))

                axs[0,0].set_title("mann-whitney u")
                axs[0,1].set_title("kolmogorov-smirnov")
                axs[0,0].set_ylabel(" CA1 vs CA3)")

                # cumulative plots
                axs[1,0].hist(tests_df_feat_pop[tests_df_feat_pop["test"] == "mann-whitney u"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")
                axs[1,1].hist(tests_df_feat_pop[tests_df_feat_pop["test"] == "kolmogorov-smirnov"]["p-value"].values, bins=np.arange(0, 1, 0.05), cumulative=True, density="True", histtype="step")

                axs[1,0].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))
                axs[1,1].set_yticks(np.arange(0,1.01,0.2), labels=np.round(np.arange(0,1.01,0.2),1))

                for x in range(2):
                    for y in range(2):
                       #axs[x, y].axvline(np.log10(0.05), color="red", linewidth=1.5)
                       axs[x, y].axvline(0.05, color="red", linewidth=1.5)

                suptitle = f"CA1 vs CA3     {pop}     {feature}     {eq_type}\nsessions: CA1={self.n_CA1_sessions}, CA3={self.n_CA3_sessions}"
                fig.suptitle(suptitle)
                fig.tight_layout()
                if pop == "newly formed":
                    pop_suffix = "NF"
                else:
                    pop_suffix = "ES"
                omission_suffix = "_omission_" if self.omission else "_"
                fig.savefig(f"{self.output_root}/cell_subsets_both_areas/subsets_{n_subsets}_{eq_type}_{selection_sizes}/pvalues_bothAreas{omission_suffix}{pop_suffix}_subsets_{n_subsets}_{eq_type}_{selection_sizes}_{feature}.pdf")
                plt.close(fig)

    def summarize_results(self):

        def cell_val(sp, pop):
            try:
                val = np.round(sp[(sp["population"] == pop) & (sp["is significant"] == True)]["prop"].values[0],2)
            except IndexError:
                val = 0.0
            return val

        wb = openpyxl.Workbook()

        run_types_and_test_results = {
            "equalize_by_sessions": self.tests_equalize_by_sessions,
            "equalize_by_cells": self.tests_equalize_by_cells
        }
        run_types_and_counts = {
            "equalize_by_sessions": [self.selection_sizes_equalize_by_sessions],
            "equalize_by_cells": [self.selection_sizes_equalize_by_cells, self.n_runs_equalize_by_cells]
        }
        for run_type, test_results in run_types_and_test_results.items():
            wb.create_sheet(run_type)
            ws = wb.get_sheet_by_name(run_type)
            ws.append(["", run_type])
            ws.append(["", "CA1", "CA3"])
            ws.append(["max num. cells/sessions", *run_types_and_counts[run_type][0]])
            if run_type == "equalize_by_cells":
                ws.append(["num. of runs", run_types_and_counts[run_type][1]])
            else:
                ws.append([""])
            ws.append([""])

            df = test_results
            df["is significant"] = df["p-value"] < 0.05
            df = df[(df["test"] == "mann-whitney u") | (df["test"] == "wilcoxon")]

            for feature in self.features:
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
        suffix = "_sessions_{s}_cells_{c}_runs_{r}".format(s=self.selection_sizes_equalize_by_sessions,
                                                           c=self.selection_sizes_equalize_by_cells,
                                                           r=self.n_runs_equalize_by_cells)
        wb.save(f"{self.output_root}/proportions_of_significant_tests{suffix}.xlsx")


if __name__ == "__main__":
    data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
    output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

    extra_info_CA1 = ""
    extra_info_CA3 = ""
    extra_info = ""

    stats = Statistics_Robustness(data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info)
    stats.calc_params()
    #stats.plot_tests_within_area()
    #stats.plot_tests_across_areas(eq_type="pfs")

    ######## equalized to min #########
    p = PARAMS["equalized to min"]
    stats.equalize("min", "single area", "animal", "session", p["AxS"]["single area"], export=True)
    stats.equalize("min", "single area", "animal", "cells", p["AxC"]["single area"], export=True)
    stats.equalize("min", "single area", "animal", "pfs", p["AxPFs"]["single area"], export=True)
    stats.equalize("min", "single area", "session", "cells", p["SxC"]["single area"], export=True)
    stats.equalize("min", "single area", "session", "pfs", p["SxPFs"]["single area"], export=True)

    stats.equalize("min", "both areas", "animal", "session", p["AxS"]["both areas"], export=True)
    stats.equalize("min", "both areas", "animal", "cells", p["AxC"]["both areas"], export=True)
    stats.equalize("min", "both areas", "animal", "pfs", p["AxPFs"]["both areas"], export=True)
    stats.equalize("min", "both areas", "session", "cells", p["SxC"]["both areas"], export=True)
    stats.equalize("min", "both areas", "session", "pfs", p["SxPFs"]["both areas"], export=True)

    ######### equalized to average ##########
    p = PARAMS["equalized to avg"]
    stats.equalize("avg", "single area", "animal", "session", p["AxS"]["single area"], export=True)
    stats.equalize("avg", "single area", "animal", "cells", p["AxC"]["single area"], export=True)
    stats.equalize("avg", "single area", "animal", "pfs", p["AxPFs"]["single area"], export=True)
    stats.equalize("avg", "single area", "session", "cells", p["SxC"]["single area"], export=True)
    stats.equalize("avg", "single area", "session", "pfs", p["SxPFs"]["single area"], export=True)

    stats.equalize("avg", "both areas", "animal", "session", p["AxS"]["both areas"], export=True)
    stats.equalize("avg", "both areas", "animal", "cells", p["AxC"]["both areas"], export=True)
    stats.equalize("avg", "both areas", "animal", "pfs", p["AxPFs"]["both areas"], export=True)
    stats.equalize("avg", "both areas", "session", "cells", p["SxC"]["both areas"], export=True)
    stats.equalize("avg", "both areas", "session", "pfs", p["SxPFs"]["both areas"], export=True)
