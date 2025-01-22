import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from BtspStatistics import BtspStatistics
from constants import AREA_PALETTE
import scipy
import tqdm
from utils import grow_df, makedir_if_needed


class Statistics_BothAreas:
    def __init__(self, data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info,
                 create_output_folder=True):
        # set parameters
        self.extra_info_CA1 = extra_info_CA1
        self.extra_info_CA3 = extra_info_CA3
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.data_root = data_root
        self.output_root = f"{output_root}//statistics//BothAreas{self.extra_info}"
        if create_output_folder:
            makedir_if_needed(self.output_root)

        # declare variables
        self.CA1_stat = None
        self.CA3_stat = None
        self.pfs_df = None
        self.shift_gain_df = None
        self.tests_df = None

        # constants
        self.bin_length = 2.13  # cm
        self.features = ["initial shift", "log10(formation gain)"]

    def load_data(self):
        self.CA1_stat = BtspStatistics("CA1", self.data_root, self.data_root, self.extra_info_CA1, is_shift_criterion_on=True, is_notebook=False)
        self.CA1_stat.load_data()
        self.CA1_stat.filter_low_behavior_score()
        self.CA1_stat.calc_shift_gain_distribution(unit="cm")
        self.CA1_stat.pfs_df["area"] = "CA1"
        self.CA1_stat.shift_gain_df["area"] = "CA1"

        self.CA3_stat = BtspStatistics("CA3", self.data_root, self.data_root, self.extra_info_CA3, is_shift_criterion_on=True, is_notebook=False)
        self.CA3_stat.load_data()
        self.CA3_stat.filter_low_behavior_score()
        self.CA3_stat.calc_shift_gain_distribution(unit="cm")
        self.CA3_stat.pfs_df["area"] = "CA3"
        self.CA3_stat.shift_gain_df["area"] = "CA3"

        self.pfs_df = pd.concat([self.CA1_stat.pfs_df, self.CA3_stat.pfs_df]).reset_index(drop=True)
        self.shift_gain_df = pd.concat([self.CA1_stat.shift_gain_df, self.CA3_stat.shift_gain_df]).reset_index(drop=True)

        n_minlaps = 10
        if f"NFafter{n_minlaps}Laps" in self.extra_info:
            pfs = self.pfs_df
            if "ESatLap0" in self.extra_info:
                pfs = pfs[(pfs["newly formed"] == True) | (pfs["newly formed"] == False) & (pfs["formation lap"] == 0)]
            self.pfs_df = pfs[(pfs["newly formed"] == False) | (pfs["newly formed"] == True) & (pfs["formation lap"] > n_minlaps)]
            shift_gain_df = self.pfs_df[["area", "animal id", "session id", "cell id", "corridor", "category", "newly formed", "initial shift", "formation gain", "formation rate sum", "formation lap"]].reset_index(drop=True)
            shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"].notna()) & (shift_gain_df["formation gain"].notna())]
            shift_gain_df["log10(formation gain)"] = np.log10(shift_gain_df["formation gain"])
            shift_gain_df["initial shift"] = self.bin_length * shift_gain_df["initial shift"]
            self.shift_gain_df = shift_gain_df.reset_index(drop=True)

    def run_tests(self, save_results=True):
        for pop in ["newly formed", "established"]:
            if pop == "newly formed":
                is_nf = True
            else:
                is_nf = False

            sg_df = self.shift_gain_df[self.shift_gain_df["newly formed"] == is_nf]
            sg_CA1 = sg_df[sg_df["area"] == "CA1"]
            sg_CA3 = sg_df[sg_df["area"] == "CA3"]

            shift_mw = scipy.stats.mannwhitneyu(sg_CA1["initial shift"].values, sg_CA3["initial shift"].values)
            shift_ks = scipy.stats.kstest(sg_CA1["initial shift"].values, cdf=sg_CA3["initial shift"].values)

            gain_mw = scipy.stats.mannwhitneyu(sg_CA1["log10(formation gain)"].values, sg_CA3["log10(formation gain)"].values)
            gain_ks = scipy.stats.kstest(sg_CA1["log10(formation gain)"].values, cdf=sg_CA3["log10(formation gain)"].values)

            n_cells_CA1 = sg_CA1.groupby(["area", "animal id", "session id"])["cell id"].nunique().sum()
            n_cells_CA3 = sg_CA3.groupby(["area", "animal id", "session id"])["cell id"].nunique().sum()

            test_dict = {
                "area": ["both"] * 4,
                "population":[pop] * 4,
                "feature": ["initial shift", "initial shift", "log10(formation gain)", "log10(formation gain)"],
                "test": ["mann-whitney u", "kolmogorov-smirnov", "mann-whitney u", "kolmogorov-smirnov"],
                "statistic": [shift_mw.statistic, shift_ks.statistic, gain_mw.statistic, gain_ks.statistic],
                "p-value": [shift_mw.pvalue, shift_ks.pvalue, gain_mw.pvalue, gain_ks.pvalue],
                "log p-value": [np.log10(shift_mw.pvalue), np.log10(shift_ks.pvalue), np.log10(gain_mw.pvalue), np.log10(gain_ks.pvalue)],
                "n cells CA1": [n_cells_CA1, n_cells_CA1, n_cells_CA1, n_cells_CA1],
                "n cells CA3": [n_cells_CA3, n_cells_CA3, n_cells_CA3, n_cells_CA3],
                "n pfs CA1": [sg_CA1.shape[0], sg_CA1.shape[0], sg_CA1.shape[0], sg_CA1.shape[0]],
                "n pfs CA3": [sg_CA3.shape[0], sg_CA3.shape[0], sg_CA3.shape[0], sg_CA3.shape[0]]
            }
            test_df = pd.DataFrame.from_dict(test_dict)
            self.tests_df = grow_df(self.tests_df, test_df)
        if save_results:
            self.tests_df.to_excel(f"{self.output_root}/tests_bothAreas.xlsx", index=False)

    def plot_shift_gain_both_areas(self):
        palette = list(AREA_PALETTE.values())

        for pop in ["newly formed", "established"]:
            if pop == "newly formed":
                is_nf = True
            else:
                is_nf = False

            sg_df = self.shift_gain_df[self.shift_gain_df["newly formed"] == is_nf]
            scale = 0.85  # 0.55 for poster
            fig, axs = plt.subplots(2, 1, figsize=(scale * 8, scale * 5))

            ###########################
            ###### VIOLINPLOTS

            # shift
            q_lo = sg_df["initial shift"].quantile(0.01)
            q_hi = sg_df["initial shift"].quantile(0.99)
            nf_df_filt = sg_df[(sg_df["initial shift"] < q_hi) & (sg_df["initial shift"] > q_lo)]
            sns.violinplot(data=nf_df_filt, x="initial shift", hue="area", split=True, inner=None,
                           ax=axs[0], fill=False, palette=palette, saturation=1, gap=0.0, linewidth=3, legend=False)
            sns.boxplot(data=nf_df_filt, x="initial shift", hue="area", ax=axs[0], width=0.3, palette=palette,
                        saturation=1, linewidth=2, linecolor="k", legend=False, fliersize=0)
            axs[0].axvline(x=0, c="k", linestyle="--", zorder=0)
            axs[0].set_xlim([-25, 25])
            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[0].spines["left"].set_visible(False)
            axs[0].set_yticks([], [])
            axs[0].set_xticks(np.arange(-25, 26, 5), np.arange(-25, 26, 5))
            #axs[0].set_xlabel("")

            # gain
            q_lo = sg_df["log10(formation gain)"].quantile(0.01)
            q_hi = sg_df["log10(formation gain)"].quantile(0.99)
            nf_df_filt = sg_df[(sg_df["log10(formation gain)"] < q_hi) & (sg_df["log10(formation gain)"] > q_lo)]
            sns.violinplot(data=nf_df_filt, x="log10(formation gain)", hue="area", split=True, inner=None,
                           ax=axs[1], fill=False, palette=palette, saturation=1, gap=0.0, linewidth=3, legend=False)
            sns.boxplot(data=nf_df_filt, x="log10(formation gain)", hue="area", ax=axs[1], width=0.3, palette=palette,
                        saturation=1, linewidth=2, linecolor="k", legend=True, fliersize=0)
            axs[1].axvline(x=0, c="k", linestyle="--", zorder=0)
            axs[1].set_xlim([-1, 1])
            axs[1].spines["top"].set_visible(False)
            axs[1].spines["right"].set_visible(False)
            axs[1].spines["left"].set_visible(False)
            axs[1].set_yticks([], [])
            #axs[1].set_xlabel("")
            plt.suptitle(pop)
            plt.tight_layout()
            plt.savefig(f"{self.output_root}//shift_gain_{pop}.pdf")
            #plt.savefig(f"{self.output_root}//shift_gain_{pop}.svg")
            plt.close()

            ###########################
            ##### CUMULATIVE PLOTS
            scale = 1.3  # poster 0.55
            fig, axs = plt.subplots(1,2, figsize=(scale*6,scale*3.5))
            sg_df[sg_df["area"] == "CA1"].hist(ax=axs[0], density=True, histtype="step", range=[-30, 30], bins=60, cumulative=True,
                                               column="initial shift", color=palette[0], label="CA1", linewidth=2)
            sg_df[sg_df["area"] == "CA3"].hist(ax=axs[0], density=True, histtype="step", range=[-30, 30], bins=60, cumulative=True,
                                               column="initial shift", color=palette[1], label="CA3", linewidth=2)
            sg_df[sg_df["area"] == "CA1"].hist(ax=axs[1], density=True, histtype="step", range=[-1, 1], bins=20, cumulative=True,
                                               column="log10(formation gain)", color=palette[0], label="CA1", linewidth=2)
            sg_df[sg_df["area"] == "CA3"].hist(ax=axs[1], density=True, histtype="step", range=[-1, 1], bins=20, cumulative=True,
                                               column="log10(formation gain)", color=palette[1], label="CA3", linewidth=2)
            axs[1].set_xticks(np.arange(-1,1.1,0.5))
            axs[0].spines["top"].set_visible(False)
            axs[1].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[1].spines["right"].set_visible(False)
            axs[0].set_ylim([0, 1.05])
            axs[1].set_ylim([0, 1.05])
            axs[0].grid(False)
            axs[1].grid(False)
            axs[0].axvline(x=0, c="k", linestyle="--", zorder=0)
            axs[1].axvline(x=0, c="k", linestyle="--", zorder=0)
            #axs[0].set_title("")
            #axs[1].set_title("")
            plt.legend()
            plt.suptitle(pop)
            plt.tight_layout()
            plt.savefig(f"{self.output_root}//shift_gain_{pop}_cumulative.pdf")
            #plt.savefig(f"{self.output_root}//shift_gain_{pop}_cumulative.svg")

    def compare_jedi_NF(self):
        nf_jedi = self.shift_gain_df[(self.shift_gain_df["animal id"] == "srb402") & (self.shift_gain_df["newly formed"] == True)]
        print(f"CA1 n={len(nf_jedi[nf_jedi["area"] == "CA1"])}")
        print(f"CA3 n={len(nf_jedi[nf_jedi["area"] == "CA3"])}")
        test = scipy.stats.mannwhitneyu(nf_jedi[nf_jedi["area"] == "CA1"]["initial shift"].values,
                                        nf_jedi[nf_jedi["area"] == "CA3"]["initial shift"].values)
        print(f"MW jedi CA1 v CA3 - shift, p={np.round(test.pvalue,3)}")
        test = scipy.stats.mannwhitneyu(nf_jedi[nf_jedi["area"] == "CA1"]["log10(formation gain)"].values,
                                        nf_jedi[nf_jedi["area"] == "CA3"]["log10(formation gain)"].values)
        print(f"MW jedi CA1 v CA3 - gain,  p={np.round(test.pvalue,3)}")
        test = scipy.stats.kstest(nf_jedi[nf_jedi["area"] == "CA1"]["initial shift"].values,
                              cdf=nf_jedi[nf_jedi["area"] == "CA3"]["initial shift"].values)
        print(f"KS jedi CA1 v CA3 - shift, p={np.round(test.pvalue,3)}")
        test = scipy.stats.kstest(nf_jedi[nf_jedi["area"] == "CA1"]["log10(formation gain)"].values,
                              cdf=nf_jedi[nf_jedi["area"] == "CA3"]["log10(formation gain)"].values)
        print(f"KS jedi CA1 v CA3 - gain,  p={np.round(test.pvalue,3)}")

    def export_data(self):
        cols = ["area", "animal id", "session id", "cell id", "corridor", "category", "lower bound", "upper bound",
                "formation lap", "end lap", "formation bin", "formation gain", "log10(formation gain)", "initial shift",
                "spearman r", "spearman p", "linear fit m", "linear fit b",
                "is BTSP", "has high gain", "has backwards shift", "has no drift"]
        # export all place field data (selected columns only)
        pfs = self.pfs_df[cols]
        pfs.to_excel(f"{self.output_root}/place_fields.xlsx", index=False)

        # export session, cell, pf counts, averages, etc.
        sg_df = self.shift_gain_df

        # number of animals by area
        n_A = sg_df.groupby(["area"])["animal id"].nunique()

        # number of sessions, cells and PFs by animal, by area
        n_AxS = sg_df.groupby(["area", "animal id"])["session id"].nunique()
        n_AxC = sg_df.groupby(["area", "animal id"])["cell id"].nunique()
        n_AxPF = sg_df.groupby(["area", "animal id"]).count()

        # number of cells, PFs by session, by area
        n_SxC = sg_df.groupby(["area", "session id"])["cell id"].nunique()
        n_SxPF = sg_df.groupby(["area", "session id"]).count()

        # export CA1-only test results
        self.CA1_stat.run_tests(self.CA1_stat.shift_gain_df, params=["initial shift", "log10(formation gain)"], export_results=False)
        self.CA1_stat.tests_df.to_excel(f"{self.output_root}/tests_CA1.xlsx", index=False)

        # export CA3-only test results
        self.CA3_stat.run_tests(self.CA3_stat.shift_gain_df, params=["initial shift", "log10(formation gain)"], export_results=False)
        self.CA3_stat.tests_df.to_excel(f"{self.output_root}/tests_CA3.xlsx", index=False)

        # export CA1 vs CA3 test results
        self.tests_df.to_excel(f"{self.output_root}/tests_bothAreas.xlsx", index=False)

        # export relevant test results for each animal
        pass

    def __create_shift_gain_jointplot(self, features, sg_df, tests, area, animal_or_session, n_sessions, save_path, lims=None):
        palette = ["#00B0F0", "#F5B800"]
        shift, gain = features

        # create plot
        g = sns.jointplot(data=sg_df, x=shift, y=gain,
                          palette=sns.color_palette(palette, 2), hue="newly formed", alpha=0.75, s=30,
                          marginal_kws={"common_norm": False, "cut": 0}, height=5, ratio=3, legend=False)
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")

        if lims is not None:
            plt.xlim(lims)
            plt.ylim(lims)
        else:
            plt.ylim([-1, 1])
            xlims = [-16 * self.bin_length, 16 * self.bin_length]
            plt.xticks(np.arange(-30, 40, 10))
            plt.xlim(xlims)

        nf = sg_df[sg_df["newly formed"] == True]
        es = sg_df[sg_df["newly formed"] == False]
        n_nf = len(nf[(~nf[shift].isna()) & (~nf[gain].isna())])
        n_es = len(es[(~es[shift].isna()) & (~es[gain].isna())])
        plt.annotate(f"n={n_nf}", (20, 0.9), color=palette[1])
        plt.annotate(f"n={n_es}", (20, 0.8), color=palette[0])

        tests_animal_idxed = tests.set_index(["area", "population", "feature", "test"])
        try:
            ax_shift = g.ax_marg_x
            pval_es0_shift = np.round(tests_animal_idxed.loc[area, "established", shift, "wilcoxon"]["p-value"], 3)
            pval_nf0_shift = np.round(tests_animal_idxed.loc[area, "newly formed", shift, "wilcoxon"]["p-value"], 3)
            pval_nf_es_shift = np.round(tests_animal_idxed.loc[area, "reliables", shift, "mann-whitney u"]["p-value"], 3)
            ax_shift.annotate(f"E-0 = {pval_es0_shift}", (0.6, 0.8), xycoords="axes fraction", color=palette[0])
            ax_shift.annotate(f"N-0 = {pval_nf0_shift}", (0.6, 0.6), xycoords="axes fraction", color=palette[1])
            ax_shift.annotate(f"E-N = {pval_nf_es_shift}", (0.6, 0.4), xycoords="axes fraction")
            ax_shift.annotate(f"{area}\n{animal_or_session}\nn={n_sessions}", (0.0, 0.4), xycoords="axes fraction")

            ax_gain = g.ax_marg_y
            pval_es0_gain = np.round(tests_animal_idxed.loc[area, "established", gain, "wilcoxon"]["p-value"], 3)
            pval_nf0_gain = np.round(tests_animal_idxed.loc[area, "newly formed", gain, "wilcoxon"]["p-value"], 3)
            pval_nf_es_gain = np.round(tests_animal_idxed.loc[area, "reliables", gain, "mann-whitney u"]["p-value"], 3)
            ax_gain.annotate(f"E-0 = {pval_es0_gain}", (0.2, 0.95), xycoords="axes fraction", color=palette[0])
            ax_gain.annotate(f"N-0 = {pval_nf0_gain}", (0.2, 0.9), xycoords="axes fraction", color=palette[1])
            ax_gain.annotate(f"E-N = {pval_nf_es_gain}", (0.2, 0.85), xycoords="axes fraction")
        except KeyError:
            print(f"failed to read test results for animal/session {animal_or_session}")

        plt.savefig(save_path)

    """
    def __create_shift_gain_jointplot_AL5vsNormal(self, sg_df, tests, area, animal_or_session, n_sessions, save_path):
        palettes = [["#ff1b00", "#F5B800"],
                    ["#00B0F0", "#5600ff"]]
        nf = sg_df[sg_df["newly formed"] == True]
        es = sg_df[sg_df["newly formed"] == False]
        populations = [nf, es]

        for i_pop, pop in enumerate(populations):
            palette = palettes[i_pop]
    
            # create plot
            g = sns.jointplot(data=sg_df, x=shift, y=gain,
                              palette=sns.color_palette(palette, 2), hue="newly formed", alpha=0.75, s=30,
                              marginal_kws={"common_norm": False, "cut": 0}, height=5, ratio=3, legend=False)
            plt.axvline(x=0, c="k", linestyle="--")
            plt.axhline(y=0, c="k", linestyle="--")
    
            plt.ylim([-1, 1])
            xlims = [-16 * self.bin_length, 16 * self.bin_length]
            plt.xticks(np.arange(-30, 40, 10))
            plt.xlim(xlims)
    
            nf = sg_df[sg_df["newly formed"] == True]
            es = sg_df[sg_df["newly formed"] == False]
            n_nf = len(nf[(~nf[shift].isna()) & (~nf[gain].isna())])
            n_es = len(es[(~es[shift].isna()) & (~es[gain].isna())])
            plt.annotate(f"n={n_nf}", (20, 0.9), color=palette[1])
            plt.annotate(f"n={n_es}", (20, 0.8), color=palette[0])
    
            tests_animal_idxed = tests.set_index(["area", "population", "feature", "test"])
            try:
                ax_shift = g.ax_marg_x
                pval_es0_shift = np.round(tests_animal_idxed.loc[area, "established", shift, "wilcoxon"]["p-value"], 3)
                pval_nf0_shift = np.round(tests_animal_idxed.loc[area, "newly formed", shift, "wilcoxon"]["p-value"], 3)
                pval_nf_es_shift = np.round(tests_animal_idxed.loc[area, "reliables", shift, "mann-whitney u"]["p-value"], 3)
                ax_shift.annotate(f"E-0 = {pval_es0_shift}", (0.6, 0.8), xycoords="axes fraction", color=palette[0])
                ax_shift.annotate(f"N-0 = {pval_nf0_shift}", (0.6, 0.6), xycoords="axes fraction", color=palette[1])
                ax_shift.annotate(f"E-N = {pval_nf_es_shift}", (0.6, 0.4), xycoords="axes fraction")
                ax_shift.annotate(f"{area}\n{animal_or_session}\nn={n_sessions}", (0.0, 0.4), xycoords="axes fraction")
    
                ax_gain = g.ax_marg_y
                pval_es0_gain = np.round(tests_animal_idxed.loc[area, "established", gain, "wilcoxon"]["p-value"], 3)
                pval_nf0_gain = np.round(tests_animal_idxed.loc[area, "newly formed", gain, "wilcoxon"]["p-value"], 3)
                pval_nf_es_gain = np.round(tests_animal_idxed.loc[area, "reliables", gain, "mann-whitney u"]["p-value"], 3)
                ax_gain.annotate(f"E-0 = {pval_es0_gain}", (0.2, 0.95), xycoords="axes fraction", color=palette[0])
                ax_gain.annotate(f"N-0 = {pval_nf0_gain}", (0.2, 0.9), xycoords="axes fraction", color=palette[1])
                ax_gain.annotate(f"E-N = {pval_nf_es_gain}", (0.2, 0.85), xycoords="axes fraction")
            except KeyError:
                print(f"failed to read test results for animal/session {animal_or_session}")
    
            plt.savefig(save_path)
    """

    def plot_shift_gain_for_each_animal(self):
        makedir_if_needed(f"{self.output_root}/shift_gain_by_animals")
        sg_df = self.shift_gain_df

        for i_area, area in enumerate(["CA1", "CA3"]):
            stat_area = [self.CA1_stat, self.CA3_stat][i_area]
            animals = stat_area.shift_gain_df["animal id"].unique()
            for animal in animals:
                sg_animal = sg_df[(sg_df["area"] == area) & (sg_df["animal id"] == animal)]
                n_sessions = len(sg_animal["session id"].unique())

                # run tests for data coming from single animal only
                features_AL5 = ["initial shift AL5", "log10(formation gain AL5)"]
                stat_animal = BtspStatistics(area, self.data_root, self.output_root)
                stat_animal.shift_gain_df = sg_animal
                stat_animal.run_tests(stat_animal.shift_gain_df, params=self.features, export_results=False)
                stat_animal.run_tests(stat_animal.shift_gain_df, params=features_AL5, export_results=False)
                tests_animal = stat_animal.tests_df

                save_path = f"{self.output_root}/shift_gain_by_animals/{area}_{animal}.svg"
                self.__create_shift_gain_jointplot(self.features, sg_animal, tests_animal, area, animal, n_sessions, save_path)

                save_path = f"{self.output_root}/shift_gain_by_animals/{area}_{animal}_AL5.svg"
                self.__create_shift_gain_jointplot(features_AL5, sg_animal, tests_animal, area, animal, n_sessions, save_path)

                save_path = f"{self.output_root}/shift_gain_by_animals/{area}_{animal}_AL5_shiftOnly.svg"
                features_AL5_shiftOnly = ["initial shift", "initial shift AL5"]
                self.__create_shift_gain_jointplot(features_AL5_shiftOnly, sg_animal, tests_animal, area,
                                                   animal, n_sessions, save_path, lims=[-20,20])

                save_path = f"{self.output_root}/shift_gain_by_animals/{area}_{animal}_AL5_gainOnly.svg"
                features_AL5_gainOnly = ["log10(formation gain)", "log10(formation gain AL5)"]
                self.__create_shift_gain_jointplot(features_AL5_gainOnly, sg_animal, tests_animal, area,
                                                   animal, n_sessions, save_path, lims=[-1,1])


                sessions = sg_animal["session id"].unique()
                for session in sessions:
                    makedir_if_needed(f"{self.output_root}/shift_gain_by_animals/{area}_{animal}_sessions")
                    sg_session = sg_animal[sg_animal["session id"] == session]
                    n_sessions = 1

                    # run tests for data coming from single animal only
                    stat_session = BtspStatistics(area, self.data_root, self.output_root)
                    stat_session.shift_gain_df = sg_session
                    try:
                        stat_session.run_tests(stat_session.shift_gain_df, params=self.features, export_results=False)
                        tests_session = stat_session.tests_df

                        save_path = f"{self.output_root}/shift_gain_by_animals/{area}_{animal}_sessions/{session}.svg"
                        self.__create_shift_gain_jointplot(self.features, sg_session, tests_session, area, session, n_sessions, save_path)
                    except ValueError:
                        print(f"error during session {session}; skipping")
                        continue


    """
    def crossvalidation():
        nf_df = shift_gain_df[shift_gain_df["newly formed"] == True]
        for crossval_type in ["animal", "session"]:
            print(f"running crossval on {crossval_type}...")
            items = nf_df[f"{crossval_type} id"].unique()
            p_vals = {
                f"{crossval_type} omitted": [],
                "[CA1 & CA3] SHIFT MW": [],
                "[CA1 & CA3] GAIN MW": [],
                "[CA1 & CA3] SHIFT KS": [],
                "[CA1 & CA3] GAIN KS": [],
                "[CA1] SHIFT WCX": [],
                "[CA1] GAIN WCX": [],
                "[CA3] SHIFT WCX": [],
                "[CA3] GAIN WCX": [],
            }
            for item in tqdm.tqdm(items):
                p_vals[f"{crossval_type} omitted"].append(item)
                pfs = nf_df[nf_df[f"{crossval_type} id"] != item]
                test = scipy.stats.mannwhitneyu(pfs[pfs["area"] == "CA1"]["initial shift"].values, pfs[pfs["area"] == "CA3"]["initial shift"].values)
                p_vals["[CA1 & CA3] SHIFT MW"].append(test.pvalue)
                test = scipy.stats.kstest(pfs[pfs["area"] == "CA1"]["initial shift"].values, cdf=pfs[pfs["area"] == "CA3"]["initial shift"].values)
                p_vals["[CA1 & CA3] SHIFT KS"].append(test.pvalue)
                test = scipy.stats.wilcoxon(pfs[pfs["area"] == "CA1"]["initial shift"].values)
                p_vals["[CA1] SHIFT WCX"].append(test.pvalue)
                test = scipy.stats.wilcoxon(pfs[pfs["area"] == "CA3"]["initial shift"].values)
                p_vals["[CA3] SHIFT WCX"].append(test.pvalue)
    
                test = scipy.stats.mannwhitneyu(pfs[pfs["area"] == "CA1"]["log10(formation gain)"].values, pfs[pfs["area"] == "CA3"]["log10(formation gain)"].values)
                p_vals["[CA1 & CA3] GAIN MW"].append(test.pvalue)
                test = scipy.stats.kstest(pfs[pfs["area"] == "CA1"]["log10(formation gain)"].values, cdf=pfs[pfs["area"] == "CA3"]["log10(formation gain)"].values)
                p_vals["[CA1 & CA3] GAIN KS"].append(test.pvalue)
                test = scipy.stats.wilcoxon(pfs[pfs["area"] == "CA1"]["log10(formation gain)"].values)
                p_vals["[CA1] GAIN WCX"].append(test.pvalue)
                test = scipy.stats.wilcoxon(pfs[pfs["area"] == "CA3"]["log10(formation gain)"].values)
                p_vals["[CA3] GAIN WCX"].append(test.pvalue)
            crossval_df = pd.DataFrame.from_dict(p_vals)
            crossval_df.to_excel(f"{CA1_stat.output_root}//crossval_{crossval_type}.xlsx")
    
            fig, ax = plt.subplots()
            cols = ["[CA1] SHIFT WCX", "[CA3] SHIFT WCX", "[CA1] GAIN WCX", "[CA3] GAIN WCX"]
            axs = crossval_df[cols].hist(bins=np.arange(0, 1, 0.025), ax=ax)
            for ax in axs.flatten():
                ax.axvline(0.05, color="red", linewidth=1.5)
            plt.tight_layout()
            plt.savefig(f"{CA1_stat.output_root}//crossval_{crossval_type}_wcx.pdf")
    
            fig, ax = plt.subplots()
            cols = ["[CA1 & CA3] SHIFT MW", "[CA1 & CA3] SHIFT KS", "[CA1 & CA3] GAIN MW", "[CA1 & CA3] GAIN KS"]
            axs = crossval_df[cols].hist(bins=np.arange(0, 1, 0.025), ax=ax)
            for ax in axs.flatten():
                ax.axvline(0.05, color="red", linewidth=1.5)
            plt.tight_layout()
            plt.savefig(f"{CA1_stat.output_root}//crossval_{crossval_type}_mw_ks.pdf")
            plt.show()
    
    def chi_square_proportions():
        ca1_btsp = CA1_stat.pfs_df.groupby("category").count()["index"]["btsp"]
        ca1_nonbtsp = CA1_stat.pfs_df.groupby("category").count()["index"]["non-btsp"]
        ca3_btsp = CA3_stat.pfs_df.groupby("category").count()["index"]["btsp"]
        ca3_nonbtsp = CA3_stat.pfs_df.groupby("category").count()["index"]["non-btsp"]
        pass
    """

    def plot_pf_counts_both_areas(self):
        pfs_CA1 = stats.CA1_stat.pfs_df
        pfs_CA3 = stats.CA3_stat.pfs_df
        fig, axs = plt.subplots(2, 2, sharex=True)
        for i_is_corridor, is_corridor in enumerate([False, True]):
            idx_cols = ["animal id", "session id", "cell id"]
            label = "by cells"
            if is_corridor:
                idx_cols.append("corridor")
                label = "by corridor"
            pf_counts_CA1 = pfs_CA1.groupby(idx_cols).count().rename(columns={"index": "number of pfs"})["number of pfs"]
            pf_counts_CA3 = pfs_CA3.groupby(idx_cols).count().rename(columns={"index": "number of pfs"})["number of pfs"]

            sns.histplot(data=pf_counts_CA1-0.5, binwidth=1, ax=axs[i_is_corridor,0], legend=True,
                         label="CA1", color=AREA_PALETTE["CA1"])  # -0.5 so bins are centered
            sns.histplot(data=pf_counts_CA3-0.5, binwidth=1, ax=axs[i_is_corridor,1], legend=True,
                         label="CA3", color=AREA_PALETTE["CA3"])  # -0.5 so bins are centered
            axs[i_is_corridor,0].set_xticks(np.arange(1,15), np.arange(1,15))
            axs[i_is_corridor,0].legend(); axs[i_is_corridor,0].set_ylabel(label)
            axs[i_is_corridor,1].legend(); axs[i_is_corridor,1].set_ylabel("")
        fig.suptitle("total PFs")

        fig, axs = plt.subplots(2, 2, sharex=True)
        pfs_CA1_reliable = self.CA1_stat.shift_gain_df.reset_index()
        pfs_CA3_reliable = self.CA3_stat.shift_gain_df.reset_index()
        for i_is_corridor, is_corridor in enumerate([False, True]):
            idx_cols = ["animal id", "session id", "cell id"]
            label = "by cells"
            if is_corridor:
                idx_cols.append("corridor")
                label = "by corridor"
            pf_counts_CA1 = pfs_CA1_reliable.groupby(idx_cols).count().rename(columns={"index": "number of pfs"})["number of pfs"]
            pf_counts_CA3 = pfs_CA3_reliable.groupby(idx_cols).count().rename(columns={"index": "number of pfs"})["number of pfs"]

            sns.histplot(data=pf_counts_CA1-0.5, binwidth=1, ax=axs[i_is_corridor,0], legend=True,
                         label="CA1", color=AREA_PALETTE["CA1"])  # -0.5 so bins are centered
            sns.histplot(data=pf_counts_CA3-0.5, binwidth=1, ax=axs[i_is_corridor,1], legend=True,
                         label="CA3", color=AREA_PALETTE["CA3"])  # -0.5 so bins are centered
            axs[i_is_corridor,0].set_xticks(np.arange(1,15), np.arange(1,15))
            axs[i_is_corridor,0].legend(); axs[i_is_corridor,0].set_ylabel(label)
            axs[i_is_corridor,1].legend(); axs[i_is_corridor,1].set_ylabel("")
        fig.suptitle("reliable PFs")
        plt.show()

if __name__ == "__main__":
    data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
    output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

    extra_info_CA1 = ""
    extra_info_CA3 = ""
    extra_info = ""

    stats = Statistics_BothAreas(data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info)
    stats.load_data()
    stats.run_tests()
    #stats.plot_shift_gain_both_areas()
    stats.plot_shift_gain_for_each_animal()
    #stats.plot_pf_counts_both_areas()
    stats.compare_jedi_NF()
    stats.export_data()