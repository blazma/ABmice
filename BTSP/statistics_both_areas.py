import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from BtspStatistics import BtspStatistics
from constants import AREA_PALETTE
import scipy
import tqdm

data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = ""
extra_info_CA3 = ""

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

pfs_df = pd.concat([CA1_stat.pfs_df, CA3_stat.pfs_df]).reset_index(drop=True)
shift_gain_df = pd.concat([CA1_stat.shift_gain_df, CA3_stat.shift_gain_df]).reset_index(drop=True)

def plot_shift_gain_both_areas_violinplot():
    scale = 0.55
    fig, axs = plt.subplots(2,1, figsize=(scale*8,scale*5))
    palette = list(AREA_PALETTE.values())
    nf_df = shift_gain_df[shift_gain_df["newly formed"] == True]

    test = scipy.stats.mannwhitneyu(nf_df[nf_df["area"] == "CA1"]["initial shift"].values, nf_df[nf_df["area"] == "CA3"]["initial shift"].values)
    print(f"SHIFT; MW; p={test.pvalue:.3}")
    test = scipy.stats.kstest(nf_df[nf_df["area"] == "CA1"]["initial shift"].values, cdf=nf_df[nf_df["area"] == "CA3"]["initial shift"].values)
    print(f"SHIFT; KS; p={test.pvalue:.3}")
    test = scipy.stats.wilcoxon(nf_df[nf_df["area"] == "CA1"]["initial shift"].values)
    print(f"CA1-SHIFT; wilcox; p={test.pvalue:.3}")
    test = scipy.stats.wilcoxon(nf_df[nf_df["area"] == "CA3"]["initial shift"].values)
    print(f"CA3-SHIFT; wilcox; p={test.pvalue:.3}")

    test = scipy.stats.mannwhitneyu(nf_df[nf_df["area"] == "CA1"]["log10(formation gain)"].values, nf_df[nf_df["area"] == "CA3"]["log10(formation gain)"].values)
    print(f"GAIN; MW; p={test.pvalue:.3}")
    test = scipy.stats.kstest(nf_df[nf_df["area"] == "CA1"]["log10(formation gain)"].values, cdf=nf_df[nf_df["area"] == "CA3"]["log10(formation gain)"].values)
    print(f"GAIN; KS; p={test.pvalue:.3}")
    test = scipy.stats.wilcoxon(nf_df[nf_df["area"] == "CA1"]["log10(formation gain)"].values)
    print(f"CA1-GAIN; wilcox; p={test.pvalue:.3}")
    test = scipy.stats.wilcoxon(nf_df[nf_df["area"] == "CA3"]["log10(formation gain)"].values)
    print(f"CA3-GAIN; wilcox; p={test.pvalue:.3}")

    # shift
    q_lo = nf_df["initial shift"].quantile(0.01)
    q_hi = nf_df["initial shift"].quantile(0.99)
    nf_df_filt = nf_df[(nf_df["initial shift"] < q_hi) & (nf_df["initial shift"] > q_lo)]
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
    axs[0].set_xlabel("")

    # gain
    q_lo = nf_df["log10(formation gain)"].quantile(0.01)
    q_hi = nf_df["log10(formation gain)"].quantile(0.99)
    nf_df_filt = nf_df[(nf_df["log10(formation gain)"] < q_hi) & (nf_df["log10(formation gain)"] > q_lo)]
    sns.violinplot(data=nf_df_filt, x="log10(formation gain)", hue="area", split=True, inner=None,
                   ax=axs[1], fill=False, palette=palette, saturation=1, gap=0.0, linewidth=3, legend=False)
    sns.boxplot(data=nf_df_filt, x="log10(formation gain)", hue="area", ax=axs[1], width=0.3, palette=palette,
                saturation=1, linewidth=2, linecolor="k", legend=False, fliersize=0)
    axs[1].axvline(x=0, c="k", linestyle="--", zorder=0)
    axs[1].set_xlim([-1, 1])
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["left"].set_visible(False)
    axs[1].set_yticks([], [])
    axs[1].set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{CA1_stat.output_root}//newly_formed_differences.pdf")
    plt.savefig(f"{CA1_stat.output_root}//newly_formed_differences.svg")
    plt.close()

    # cumulative distro
    scale = 0.55
    fig, axs = plt.subplots(2, figsize=(scale*3.5,scale*6));
    nf_df[nf_df["area"] == "CA1"].hist(ax=axs[0], density=True, histtype="step", range=[-30, 30], bins=60, cumulative=True,
                                       column="initial shift", color=palette[0], label="CA1", linewidth=2)
    nf_df[nf_df["area"] == "CA3"].hist(ax=axs[0], density=True, histtype="step", range=[-30, 30], bins=60, cumulative=True,
                                       column="initial shift", color=palette[1], label="CA3", linewidth=2)
    nf_df[nf_df["area"] == "CA1"].hist(ax=axs[1], density=True, histtype="step", range=[-1, 1], bins=20, cumulative=True,
                                       column="log10(formation gain)", color=palette[0], label="CA1", linewidth=2)
    nf_df[nf_df["area"] == "CA3"].hist(ax=axs[1], density=True, histtype="step", range=[-1, 1], bins=20, cumulative=True,
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
    axs[0].set_title("")
    axs[1].set_title("")
    plt.tight_layout()
    #plt.legend()
    plt.savefig(f"{CA1_stat.output_root}//newly_formed_differences_cumulative.pdf")
    plt.savefig(f"{CA1_stat.output_root}//newly_formed_differences_cumulative.svg")
    #plt.show()

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

crossvalidation()
#plot_shift_gain_both_areas_violinplot()
#chi_square_proportions()
