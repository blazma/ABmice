import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from BtspStatistics import BtspStatistics
import scipy


def calc_diff(sg_df, param, median=False):
    # calculates difference in mean (or median) shift (or gain) between newly formed and established subpopulations
    # param may be either 'initial shift' or 'formation gain'
    if median:
        median_newlyf = sg_df[sg_df['newly formed'] == True][param].median()
        median_stable = sg_df[sg_df['newly formed'] == False][param].median()
        diff = median_newlyf - median_stable
    else:
        mean_newlyf = sg_df[sg_df['newly formed'] == True][param].mean()
        mean_stable = sg_df[sg_df['newly formed'] == False][param].mean()
        diff = mean_newlyf - mean_stable
    return diff

def bootstrapping(data_path, output_path, median=False, extra_info="", take_diff=True, without_transient=False):
    CA1_stats = BtspStatistics("CA1", data_path, output_path, extra_info=extra_info)
    CA1_stats.load_data()
    CA1_stats.calc_shift_gain_distribution()
    CA1_sg_df = CA1_stats.shift_gain_df
    if without_transient:
        CA1_sg_df = CA1_sg_df[CA1_sg_df["category"] != "transient"]

    CA3_stats = BtspStatistics("CA3", data_path, output_path, extra_info=extra_info)
    CA3_stats.load_data()
    CA3_stats.calc_shift_gain_distribution()
    CA3_sg_df = CA3_stats.shift_gain_df
    if without_transient:
        CA3_sg_df = CA3_sg_df[CA3_sg_df["category"] != "transient"]

    CA1_shifts = []
    CA1_gains = []

    n = CA3_sg_df.shape[0]  # number of place fields in CA3
    print(n)
    for seed in range(1000):
        shift_gain_df_subs = CA1_sg_df.sample(n=n, random_state=seed)
        if take_diff:
            shift = calc_diff(shift_gain_df_subs, param='initial shift', median=median)
            gain = calc_diff(shift_gain_df_subs, param='log10(formation gain)', median=median)
        else:
            shift = shift_gain_df_subs[shift_gain_df_subs["newly formed"] == True]["initial shift"].mean()
            gain = shift_gain_df_subs[shift_gain_df_subs["newly formed"] == True]["log10(formation gain)"].mean()
        CA1_shifts.append(shift)
        CA1_gains.append(gain)

    # Mann-Whitney between CA1 and CA3 newly formed PFs
    CA1_NF = CA1_sg_df[CA1_sg_df["newly formed"]]
    CA3_NF = CA3_sg_df[CA3_sg_df["newly formed"]]

    p_shift = scipy.stats.mannwhitneyu(CA1_NF["initial shift"], CA3_NF["initial shift"]).pvalue
    p_gain = scipy.stats.mannwhitneyu(CA1_NF["log10(formation gain)"], CA3_NF["log10(formation gain)"]).pvalue

    print("Comparison of CA1 and CA3 - Mann-Whitney U")
    print(f"initial shift: p={p_shift:.3}")
    print(f"log10(formation gain): p={p_gain:.3}")

    def plot_kde(diffs, ax, color):
        kde = scipy.stats.gaussian_kde(diffs)
        pos = np.linspace(min(diffs), max(diffs), 100)
        #plt.plot(pos, kde_shift(pos), color="k", linewidth=3)
        ax.fill_between(pos, kde(pos), alpha=0.15, color="b")
        ax.set_ylim([0,max(kde(pos))])

        # 95% confidence interval
        lb = np.percentile(diffs, q=2.5)
        ub = np.percentile(diffs, q=97.5)
        ci_95_lo = np.linspace(min(diffs), lb, 100)
        ci_95_hi = np.linspace(ub, max(diffs), 100)
        ax.fill_between(ci_95_lo, kde(ci_95_lo), alpha=0.25, color=color)
        ax.fill_between(ci_95_hi, kde(ci_95_hi), alpha=0.25, color=color)

        # 99% confidence interval
        lb = np.percentile(diffs, q=0.5)
        ub = np.percentile(diffs, q=99.5)
        ci_99_lo = np.linspace(min(diffs), lb, 100)
        ci_99_hi = np.linspace(ub, max(diffs), 100)
        ax.fill_between(ci_99_lo, kde(ci_99_lo), alpha=0.25, color=color)
        ax.fill_between(ci_99_hi, kde(ci_99_hi), alpha=0.25, color=color)

    scale = 0.8
    fig, ax = plt.subplots(figsize=(scale*5, scale*3))
    plot_kde(CA1_shifts, ax=ax, color="b")

    #sg_df_CA3 = sg_df_CA3[sg_df_CA3["initial shift"] > -20]

    if take_diff:
        CA3_shifts = calc_diff(CA3_sg_df, param='initial shift', median=median)
        ax.set_xlabel("$\Delta\ mean\ initial\ shift$")
    else:
        CA3_shifts = CA3_sg_df[CA3_sg_df["newly formed"] == True]["initial shift"].mean()
        ax.set_xlabel("mean initial shift; newly formed PFs")
    ax.axvline(CA3_shifts, c="r", linewidth=2, label="CA3")
    ax.axvline(0, c="k", linestyle="--")
    ax.set_xlim([-6,6])
    ax.spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    save_path = f"{CA1_stats.output_root}/BOOTSTRAP_SHIFT"
    save_path = f"{save_path}_median" if median else f"{save_path}"
    save_path = f"{save_path}_diffs" if take_diff else f"{save_path}_NF"
    save_path = f"{save_path}_withoutTransients" if without_transient else f"f{save_path}"
    plt.savefig(f"{save_path}.pdf")
    plt.savefig(f"{save_path}.svg")

    scale = 0.8
    fig, ax = plt.subplots(figsize=(scale*5, scale*3))
    plot_kde(CA1_gains, ax=ax, color="b")
    ax.set_xlabel("$\Delta\ mean\ formation\ gain$")
    if take_diff:
        CA3_gains = calc_diff(CA3_stats.shift_gain_df, param='log10(formation gain)', median=median)
        ax.set_xlabel("$\Delta\ mean\ log_{10}(formation\ gain)$")
    else:
        CA3_gains = CA3_sg_df[CA3_sg_df["newly formed"] == True]["log10(formation gain)"].mean()
        ax.set_xlabel("$mean\ log_{10}(formation gain);\quad newly\ formed\ PFs$")
    ax.axvline(CA3_gains, c="r", linewidth=2, label="CA3")
    ax.axvline(0, c="k", linestyle="--")
    ax.set_xlim([-0.2,0.2])
    ax.spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    save_path = f"{CA1_stats.output_root}/BOOTSTRAP_GAIN"
    save_path = f"{save_path}_median" if median else f"{save_path}"
    save_path = f"{save_path}_diffs" if take_diff else f"{save_path}_NF"
    save_path = f"{save_path}_withoutTransients" if without_transient else f"f{save_path}"
    plt.savefig(f"{save_path}.pdf")
    plt.savefig(f"{save_path}.svg")

    # calculate p-values
    CA1_shifts = np.array(CA1_shifts)
    CA1_gains = np.array(CA1_gains)
    def pvalue(sample, mu):
        one_tailed_p = sample[sample >= mu].size / sample.size
        if one_tailed_p > 0.5:
            return np.round(2*(1-one_tailed_p), 3)
        else:
            return np.round(2*one_tailed_p, 3)

    if take_diff:
        tests = f"n={n}\n" +\
                f"0 shift-diff in CA1:\tp={pvalue(CA1_shifts, 0)}\n" +\
                f"0 gain-diff in CA1:\tp={pvalue(CA1_gains, 0)}\n" +\
                f"CA3 shift-diff against CA1:\tp={pvalue(CA1_shifts, CA3_shifts)}\n" +\
                f"CA3 gain-diff against CA1:\tp={pvalue(CA1_gains, CA3_gains)}\n"
        tests_file_name = "tests_bootstrap_withoutTransients.txt" if without_transient else "tests_bootstrap.txt"
        with open(f"{CA1_stats.output_root}/{tests_file_name}", "w") as tests_file:
            tests_file.write(tests)
    else:
        tests = f"n={n}\n" +\
                f"0 NF shift in CA1:\tp={pvalue(CA1_shifts, 0)}\n" +\
                f"0 NF gain in CA1:\tp={pvalue(CA1_gains, 0)}\n" +\
                f"CA3 NF shift against CA1:\tp={pvalue(CA1_shifts, CA3_shifts)}\n" +\
                f"CA3 NF gain against CA1:\tp={pvalue(CA1_gains, CA3_gains)}\n"
        tests_file_name = "tests_bootstrap_withoutTransients_NF.txt" if without_transient else "tests_bootstrap_NF.txt"
        with open(f"{CA1_stats.output_root}/{tests_file_name}", "w") as tests_file:
            tests_file.write(tests)

def newly_vs_established_MW(data_path, output_path, extra_info=""):
    CA1_stats = BtspStatistics("CA1", data_path, output_path, extra_info)
    CA1_stats.load_data()
    CA1_stats.calc_shift_gain_distribution()

    CA1_newly = CA1_stats.shift_gain_df[CA1_stats.shift_gain_df['newly formed'] == True]
    CA1_estab = CA1_stats.shift_gain_df[CA1_stats.shift_gain_df['newly formed'] == False]

    test = scipy.stats.mannwhitneyu(CA1_newly["initial shift"].values, CA1_estab["initial shift"].values)
    print(f"CA1 shift:\tu={np.round(test.statistic,3)},\tp={np.round(test.pvalue,3)}")
    test = scipy.stats.mannwhitneyu(CA1_newly["formation gain"].values, CA1_estab["formation gain"].values)
    print(f"CA1 gain:\tu={np.round(test.statistic,3)},\tp={np.round(test.pvalue,3)}")

    CA3_stats = BtspStatistics("CA3", data_path, output_path, extra_info)
    CA3_stats.load_data()
    CA3_stats.calc_shift_gain_distribution()

    CA3_newly = CA3_stats.shift_gain_df[CA3_stats.shift_gain_df['newly formed'] == True]
    CA3_estab = CA3_stats.shift_gain_df[CA3_stats.shift_gain_df['newly formed'] == False]

    test = scipy.stats.mannwhitneyu(CA3_newly["initial shift"].values, CA3_estab["initial shift"].values)
    print(f"CA3 shift:\tu={np.round(test.statistic,3)},\tp={np.round(test.pvalue,3)}")
    test = scipy.stats.mannwhitneyu(CA3_newly["formation gain"].values, CA3_estab["formation gain"].values)
    print(f"CA3 gain:\tu={np.round(test.statistic,3)},\tp={np.round(test.pvalue,3)}")

def shuffled_MW(data_path, output_path, extra_info=""):
    for area in ["CA1", "CA3"]:
        stats = BtspStatistics(area, data_path, output_path, extra_info)
        stats.load_data()
        stats.calc_shift_gain_distribution()

        extra_info_sh = f"{extra_info}_shuffledLaps"
        stats_sh = BtspStatistics(area, data_path, output_path, extra_info_sh)
        stats_sh.load_data()
        stats_sh.calc_shift_gain_distribution()

        fig, axs = plt.subplots(2,2)
        for i_param, param in enumerate(["gain", "shift"]):
            for i_nf_cond, is_newly_formed in enumerate([True, False]):
                sg_df = stats.shift_gain_df
                sg_df["shuffled"] = False
                sg_sh_df = stats_sh.shift_gain_df
                sg_sh_df["shuffled"] = True

                sg_df = sg_df[sg_df['newly formed'] == is_newly_formed]
                sg_sh_df = sg_sh_df[sg_sh_df['newly formed'] == is_newly_formed]

                param_col = "log10(formation gain)" if param == "gain" else "initial shift"
                test = scipy.stats.mannwhitneyu(sg_df[param_col].values, sg_sh_df[param_col].values)
                #test = scipy.stats.ttest_ind(sg_df[param_col].dropna().values, sg_sh_df[param_col].dropna().values, equal_var=True)

                mean_no_shuffle = np.nanmean(sg_df[param_col].values)
                mean_shuffle = np.nanmean(sg_sh_df[param_col].values)

                newly = "new PF" if is_newly_formed else "establ"
                print(f"{area} {newly} {param}:\tu={np.round(test.statistic, 3)},\tp={np.round(test.pvalue, 3)},"
                      f"\tmean_no_shuffle={np.round(mean_no_shuffle, 3)},\tmean_shuffle={np.round(mean_shuffle,3)}")

                # violinplot
                sg_df_merged = pd.concat((sg_df, sg_sh_df)).reset_index()
                sg_df_merged_filt = sg_df_merged[sg_df_merged['newly formed'] == is_newly_formed][["shuffled", param_col]]
                if is_newly_formed:
                    palette = "Oranges"
                else:
                    palette = "Blues"
                sns.violinplot(sg_df_merged_filt, y=param_col, hue="shuffled", split=True, ax=axs[i_param, i_nf_cond], palette=palette, inner=None)
                sns.pointplot(sg_df_merged_filt, y=param_col, hue="shuffled", ax=axs[i_param, i_nf_cond], palette="dark:black", estimator=np.mean, dodge=True)
                if is_newly_formed:
                    n_NL = sg_df[sg_df["newly formed"] == True].shape[0]
                    n_SL = sg_sh_df[sg_sh_df["newly formed"] == True].shape[0]
                    title = f"newly formed, p={np.round(test.pvalue,3)},\nn(NL)={n_NL}, n(SL)={n_SL}"
                else:
                    n_NL = sg_df[sg_df["newly formed"] == False].shape[0]
                    n_SL = sg_sh_df[sg_sh_df["newly formed"] == False].shape[0]
                    title = f"established, p={np.round(test.pvalue,3)},\nn(NL)={n_NL}, n(SL)={n_SL}"

                if test.pvalue<0.05:
                    axs[i_param, i_nf_cond].set_title(title, fontdict={"color": "red"})
                else:
                    axs[i_param, i_nf_cond].set_title(title)

                if param == "shift":
                    axs[i_param, i_nf_cond].set_ylim([-15, 15])
                else:
                    axs[i_param, i_nf_cond].set_ylim([-1.5, 1.5])

                if i_param != 0:
                    axs[i_param, i_nf_cond].get_legend().set_visible(False)
                else:
                    axs[i_param, i_nf_cond].legend(title="shuffled", loc='center left', bbox_to_anchor=(1, 0.5))

                pass
        plt.tight_layout()
        plt.suptitle(area)
        plt.savefig(f"{stats.output_root}/SGS_shuffled_MW.pdf")
        plt.close()

if __name__ == "__main__":
    data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
    output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

    extra_info = ""
    bootstrapping(data_root, output_root, extra_info=extra_info, take_diff=True, without_transient=False)
    #bootstrapping(data_root, output_root, extra_info=extra_info, median=True)
    #bootstrapping(data_root, output_root, extra_info=f"{extra_info}_shuffled_laps")
    #bootstrapping(data_root, output_root, extra_info=f"{extra_info}_shuffled_laps", median=True)
    #newly_vs_established_MW(data_root, output_root)
    #shuffled_MW(data_root, output_root, extra_info=extra_info)
