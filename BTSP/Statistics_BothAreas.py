import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from BtspStatistics import BtspStatistics
from constants import AREA_PALETTE
import scipy
import tqdm
from utils import grow_df, makedir_if_needed

data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = ""
extra_info_CA3 = ""
extra_info = ""


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

            test_dict = {
                "population":[pop] * 4,
                "feature": ["initial shift", "initial shift", "log10(formation gain)", "log10(formation gain)"],
                "test": ["mann-whitney u", "kolmogorov-smirnov", "mann-whitney u", "kolmogorov-smirnov"],
                "statistic": [shift_mw.statistic, shift_ks.statistic, gain_mw.statistic, gain_ks.statistic],
                "p-value": [shift_mw.pvalue, shift_ks.pvalue, gain_mw.pvalue, gain_ks.pvalue],
                "log p-value": [np.log10(shift_mw.pvalue), np.log10(shift_ks.pvalue), np.log10(gain_mw.pvalue), np.log10(gain_ks.pvalue)]
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

    def export_data(self):
        cols = ["area", "animal id", "session id", "cell id", "corridor", "category", "lower bound", "upper bound",
                "formation lap", "end lap", "formation bin", "formation gain", "log10(formation gain)", "initial shift",
                "spearman r", "spearman p", "linear fit m", "linear fit b",
                "is BTSP", "has high gain", "has backwards shift", "has no drift"]
        # export all place field data (selected columns only)
        pfs = self.pfs_df[cols]
        pfs.to_excel(f"{self.output_root}/place_fields.xlsx", index=False)

        # export CA1-only test results
        self.CA1_stat.run_tests(self.CA1_stat.shift_gain_df, params=["initial shift", "log10(formation gain)"], export_results=False)
        self.CA1_stat.tests_df.to_excel(f"{self.output_root}/tests_CA1.xlsx", index=False)

        # export CA3-only test results
        self.CA3_stat.run_tests(self.CA3_stat.shift_gain_df, params=["initial shift", "log10(formation gain)"], export_results=False)
        self.CA3_stat.tests_df.to_excel(f"{self.output_root}/tests_CA3.xlsx", index=False)

        # export CA1 vs CA3 test results
        self.tests_df.to_excel(f"{self.output_root}/tests_bothAreas.xlsx", index=False)

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

if __name__ == "__main__":
    stats = Statistics_BothAreas(data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info)
    stats.load_data()
    stats.run_tests()
    #stats.plot_shift_gain_both_areas()
    stats.export_data()
