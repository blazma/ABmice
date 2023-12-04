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

def bootstrapping(data_path, output_path, median=False, extra_info=""):
    CA1_stats = BtspStatistics("CA1", data_path, output_path, extra_info=extra_info)
    CA1_stats.load_data()
    CA1_stats.calc_shift_gain_distribution()

    CA3_stats = BtspStatistics("CA3", data_path, output_path, extra_info=extra_info)
    CA3_stats.load_data()
    CA3_stats.calc_shift_gain_distribution()

    CA1_shift_diffs = []
    CA1_gain_diffs = []

    n = CA3_stats.shift_gain_df.shape[0]  # number of place fields in CA3
    print(n)
    for seed in range(1000):
        shift_gain_df_subs = CA1_stats.shift_gain_df.sample(n=n, random_state=seed)
        shift_diff = calc_diff(shift_gain_df_subs, param='initial shift', median=median)
        gain_diff = calc_diff(shift_gain_df_subs, param='formation gain', median=median)
        CA1_shift_diffs.append(shift_diff)
        CA1_gain_diffs.append(gain_diff)
    fig, axs = plt.subplots(2, 1)
    sns.kdeplot(data=CA1_shift_diffs, ax=axs[0], color="k")
    sns.kdeplot(data=CA1_gain_diffs, ax=axs[1], color="k")
    axs[0].set_title("shift")
    axs[1].set_title("gain")

    # plot CA3 values on distributions:
    shift_diff_CA3 = calc_diff(CA3_stats.shift_gain_df, param='initial shift', median=median)
    gain_diff_CA3 = calc_diff(CA3_stats.shift_gain_df, param='formation gain', median=median)
    axs[0].axvline(shift_diff_CA3, c="red", label="CA3")
    axs[1].axvline(gain_diff_CA3, c="red", label="CA3")

    # plot confidence intervals
    axs[0].axvline(np.percentile(CA1_shift_diffs, q=5), c="blue", label="CA1 5th p.")
    axs[0].axvline(np.percentile(CA1_shift_diffs, q=1), c="cyan", label="CA1 1st p.")
    axs[0].axvline(np.percentile(CA1_shift_diffs, q=95), c="blue", label="CA1 95th p.")
    axs[0].axvline(np.percentile(CA1_shift_diffs, q=99), c="cyan", label="CA1 99th p.")
    axs[1].axvline(np.percentile(CA1_gain_diffs, q=5), c="blue", label="CA1 5th p.")
    axs[1].axvline(np.percentile(CA1_gain_diffs, q=1), c="cyan", label="CA1 1st p.")
    axs[1].axvline(np.percentile(CA1_gain_diffs, q=95), c="blue", label="CA1 95th p.")
    axs[1].axvline(np.percentile(CA1_gain_diffs, q=99), c="cyan", label="CA1 99th p.")

    # add legends
    axs[0].legend()
    axs[1].legend()

    # set xlims
    axs[0].set_xlim([-3,3])
    axs[1].set_xlim([-1,1])

    plt.suptitle("median(newly) - median(establ)" if median else "mean(newly) - mean(establ)")
    plt.tight_layout()
    save_path = f"{CA1_stats.output_root}/SGS_plot_shift_gain_bootstrap"
    save_path = f"{save_path}_median" if median else f"{save_path}"
    plt.savefig(f"{save_path}.pdf")

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

        extra_info_sh = f"{extra_info}_shuffled_laps"
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

    extra_info = "FGactiveLaps_FLfix_noCorr"
    bootstrapping(data_root, output_root, extra_info=extra_info)
    bootstrapping(data_root, output_root, extra_info=extra_info, median=True)
    bootstrapping(data_root, output_root, extra_info=f"{extra_info}_shuffled_laps")
    bootstrapping(data_root, output_root, extra_info=f"{extra_info}_shuffled_laps", median=True)
    #newly_vs_established_MW(data_root, output_root)
    shuffled_MW(data_root, output_root, extra_info=extra_info)
