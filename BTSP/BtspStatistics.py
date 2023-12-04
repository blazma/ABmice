import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_unweighted
import matplotlib
import seaborn as sns
import cmasher
import os
import pickle
import argparse
import json
from constants import ANIMALS, SESSIONS_TO_IGNORE, CORRIDORS, CATEGORIES
from utils import grow_df, makedir_if_needed
from plotly import express as px
import sklearn
import scipy


class BtspStatistics:
    def __init__(self, area, data_path, output_path, extra_info=""):
        self.area = area
        self.animals = ANIMALS[area]
        self.data_path = data_path
        self.categories_colors_RGB = [category.color for _, category in CATEGORIES.items()]
        self.btsp_criteria = ["has high gain", "has backwards shift", "has no drift"]
        self.reward_zones = {  # TODO: these are guesses based on plots, needs to be exact
            0: [38, 46],  # corridor 14
            1: [61, 69]  # corridor 15
        }

        # set output folder
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/statistics/{self.area}{self.extra_info}"
        makedir_if_needed(f"{output_path}/statistics")
        makedir_if_needed(self.output_root)

        # dataframes for place fields and cell statistics
        self.pfs_df = None
        self.cell_stats_df = None
        self.shift_gain_df = None

        # "debug"
        self.long_pf_only = True

    def _use_font(self):
        from matplotlib import font_manager
        font_dirs = ['C:\\home\\phd\\']
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
           font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = 'Trade Gothic Next LT Pro BdCn'

    def load_data(self):
        # read place fields and cell statistics dataframes for each animal
        self.pfs_df = None
        self.cell_stats_df = None
        for animal in self.animals:
            try:
                pfs_df_animal = pd.read_pickle(f"{self.data_path}/place_fields/{self.area}{self.extra_info}/{animal}_place_fields_df.pickle")
                #cell_stats_df_animal = pd.read_pickle(f"{self.data_path}/{animal}_cell_stats_df.pickle")
            except Exception:
                print(f"ERROR occured for animal {animal} during DF pickle loading; skipping")
                continue
            #self.cell_stats_df = grow_df(self.cell_stats_df, cell_stats_df_animal)
            self.pfs_df = grow_df(self.pfs_df, pfs_df_animal)
        self.pfs_df = self.pfs_df.reset_index().drop("index", axis=1)
        #if self.long_pf_only:
        #    self.pfs_df = self.pfs_df[self.pfs_df["end lap"] - self.pfs_df["formation lap"] > 15].reset_index()
        #self.cell_stats_df = self.cell_stats_df.reset_index().drop("index", axis=1)

        # assign category orders
        for category in CATEGORIES:
            cond = self.pfs_df["category"] == category
            self.pfs_df.loc[cond, "category_order"] = CATEGORIES[category].order

        # tag newly formed place fields
        self.pfs_df["newly formed"] = np.where(self.pfs_df["category"].isin(["transient", "non-btsp", "btsp"]), True, False)

    def calc_place_field_proportions(self):
        pf_proportions_by_category_df = None
        sessions = self.pfs_df["session id"].unique()
        for session in sessions:
            animalID, _, _ = session.partition("_")
            pfs_session = self.pfs_df.loc[self.pfs_df["session id"] == session]
            n_pfs_session_by_category = pfs_session.groupby("category").count()
            n_pfs_session_total = len(pfs_session)
            pfs_proportions_session = n_pfs_session_by_category / n_pfs_session_total
            pfs_proportions_session = pfs_proportions_session["session id"].to_dict()  # select arbitrary column - they all contain the same values anyway

            # dict containing proportions of each PF category in a given session (to be converted into DF)
            pf_proportions_dict = {
                "animalID": animalID,
                "session id": session,
            }
            for category in CATEGORIES:
                if category in pfs_proportions_session:
                    pf_proportions_dict[category] = pfs_proportions_session[category]
                else:
                    pf_proportions_dict[category] = 0

            # PF proportions dataframe creation
            if pf_proportions_by_category_df is None:
                pf_proportions_by_category_df = pd.DataFrame.from_dict([pf_proportions_dict])
            else:
                pf_proportions_by_category_df = pd.concat(
                    (pf_proportions_by_category_df, pd.DataFrame.from_dict([pf_proportions_dict])))
        self.pf_proportions_by_category_df = pf_proportions_by_category_df.reset_index().drop("index", axis=1)

    def plot_cells(self):
        fig, ax = plt.subplots(figsize=(2.5, 4), dpi=125)
        sns.boxplot(data=self.cell_stats_df, ax=ax, width=0.8, palette="flare", showfliers=False)
        sns.swarmplot(data=self.cell_stats_df, ax=ax, x=1, y="total cells", hue="animalID", palette="Blues", alpha=1.0)
        sns.swarmplot(data=self.cell_stats_df, ax=ax, x=2, y="active cells", hue="animalID", palette="Blues", alpha=1.0, legend=False)
        sns.swarmplot(data=self.cell_stats_df, ax=ax, x=3, y="tuned cells", hue="animalID", palette="Blues", alpha=1.0, legend=False)
        ax.set_ylabel("# cells")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_cells.pdf")

    def plot_place_fields(self):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=125)
        pfs_by_category = self.pfs_df.groupby(["category", "category_order"]).count().sort_values(by="category_order")
        pfs_by_category.plot(kind="bar", y="session id", ax=axs[0], color=self.categories_colors_RGB, legend=False)
        pfs_by_category.plot(kind="pie", y="session id", ax=axs[1], colors=self.categories_colors_RGB, legend=False, autopct=lambda pct: f'{np.round(pct, 1)}%')
        axs[1].set_ylabel("")
        axs[0].set_ylabel("# place fields")
        axs[0].set_xlabel("")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        #axs[0].set_ylim([0, 250])
        # axs[0].set_yticks(np.linspace(0,pfs_by_category["session id"].values.max(),num=5))
        axs[0].spines[['right', 'top']].set_visible(False)
        # plt.suptitle(f"[{area}] Distribution of place fields by PF categories across all animals")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_place_fields.pdf")

    def plot_place_fields_by_session(self):
        fig, ax = plt.subplots()
        pfs_by_category = self.pfs_df.groupby(["category", "session id", "category_order"]).count().sort_values(by="category_order")
        pf_counts = pfs_by_category.reset_index()[["category", "session id", "cell id"]]
        sns.boxplot(data=pf_counts, x="category", y="cell id", ax=ax, palette=self.categories_colors_RGB, showfliers=False)
        sns.swarmplot(data=pf_counts, x="category", y="cell id", ax=ax, color="black", alpha=0.5)
        plt.title(f"[{self.area}] Number of various place fields for each session")
        ax.set_ylabel("# place fields")
        ax.set_xlabel("")
        plt.savefig(f"{self.output_root}/plot_place_fields_by_session.pdf")

    def plot_place_field_proportions(self):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=125)
        sns.boxplot(data=self.pf_proportions_by_category_df, ax=ax, palette=self.categories_colors_RGB, showfliers=False)
        sns.swarmplot(data=self.pf_proportions_by_category_df, ax=ax, color="black", alpha=0.5)
        # plt.title(f"[{area}] Proportion of various place field categories by session")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_ylabel("proportion")
        ax.set_xlabel("")
        plt.savefig(f"{self.output_root}/plot_place_field_proportions.pdf")

    def plot_place_fields_criteria(self):
        fig, ax = plt.subplots()
        plt.title(f"[{self.area}] Number of place fields with a given BTSP criterion satisfied")
        nonbtsp_pfs_w_criteria_df = self.pfs_df[self.pfs_df["category"] == "non-btsp"][self.btsp_criteria].apply(pd.value_counts).transpose()
        nonbtsp_pfs_w_criteria_df.plot(kind="bar", stacked=True, color=["r", "g"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        plt.savefig(f"{self.output_root}/plot_place_fields_criteria.pdf")

    def plot_place_field_properties(self):
        fig, ax = plt.subplots()
        self.pfs_df["pf width"] = pd.to_numeric(self.pfs_df["upper bound"] - self.pfs_df["lower bound"])
        pf_width_means = self.pfs_df.groupby("category")["pf width"].mean()
        pf_width_stds = self.pfs_df.groupby("category")["pf width"].std()
        sns.violinplot(data=self.pfs_df[["category", "pf width"]], x="category", y="pf width", ax=ax, palette=self.categories_colors_RGB)
        for i_category, category in enumerate(CATEGORIES.keys()):
            x = i_category + 0.1
            y = pf_width_means[category] + 5 * pf_width_stds[category]
            text = f"mean: {np.round(pf_width_means[category], 1)}\nstd: {np.round(pf_width_stds[category], 1)}"
            plt.annotate(text, (x, y))
        plt.savefig(f"{self.output_root}/plot_place_field_properties.pdf")

    def plot_place_field_heatmap(self):
        n_bins = self.pfs_df["upper bound"].max() - self.pfs_df["lower bound"].min()  # 75 bins
        n_laps = self.pfs_df["end lap"].max() - self.pfs_df["formation lap"].min()  # 225 laps, ez tuti túl sok biztos benne vannak 3 korridorosok

        earlies = np.zeros((n_laps, n_bins))
        transients = np.zeros((n_laps, n_bins))
        nonbtsps = np.zeros((n_laps, n_bins))
        btsps = np.zeros((n_laps, n_bins))

        fig, axs = plt.subplots(5, 4, sharex=True)

        categories = ["early", "transient", "non-btsp", "btsp"]
        colormaps = ["Blues", "Oranges", "Reds", "Purples"]
        for i_corridor in range(2):
            i_ax = 2 * i_corridor

            print(f"corridor: {i_corridor}")
            RZ_start, RZ_end = self.reward_zones[i_corridor]
            place_fields_corridor = self.pfs_df[((self.pfs_df["category"] == "early") |
                                                 (self.pfs_df["category"] == "transient") |
                                                 (self.pfs_df["category"] == "non-btsp") |
                                                 (self.pfs_df["category"] == "btsp")) &
                                                 (self.pfs_df["corridor"] == 0)]
            N_place_fields_corridor = place_fields_corridor['session id'].count()
            for i_category, category in enumerate(categories):

                full_span = None
                if category == "early":
                    full_span = earlies
                elif category == "transient":
                    full_span = transients
                elif category == "non-btsp":
                    full_span = nonbtsps
                elif category == "btsp":
                    full_span = btsps

                ax = axs[i_category, i_ax]
                colormap = colormaps[i_category]
                pfs = self.pfs_df.loc[(self.pfs_df["category"] == category) & (self.pfs_df["corridor"] == i_corridor)]
                print(category, 1 / len(pfs.index))
                for _, pf in pfs.iterrows():
                    lb = pf["lower bound"]
                    ub = pf["upper bound"]
                    fl = pf["formation lap"]
                    el = pf["end lap"]

                    pf_span = np.ones((el - fl, ub - lb))
                    full_span[fl:el, lb:ub] += pf_span
                full_span = 100 * full_span / N_place_fields_corridor
                ax.axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
                full_span = full_span[:40, :]
                im = ax.imshow(full_span, cmap=colormap, origin="lower", aspect="auto")
                plt.colorbar(im, ax=ax)

                ax = axs[i_category, i_ax + 1]
                full_span_marginal = np.sum(full_span, axis=0)
                color = colormap.lower()[:-1]
                ax.bar(np.arange(len(full_span_marginal)), full_span_marginal, width=1.0, color=color)
                ax.axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
                ax.set_ylim([0, 1.1 * full_span_marginal.max()])

            ax = axs[4, i_ax]
            btsp_nonbtsp_proportion = np.divide(btsps[:40, :], nonbtsps[:40, :])
            # ax.axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
            ax.set_ylim([2, 40])
            im = ax.imshow(btsp_nonbtsp_proportion, cmap="binary", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax)
        plt.savefig(f"{self.output_root}/plot_place_field_heatmap.pdf")

    def plot_place_fields_criteria_venn_diagram(self, with_btsp=False):
        if with_btsp:
            place_fields_filtered = self.pfs_df[(self.pfs_df["category"] == "non-btsp") | (self.pfs_df["category"] == "btsp")]
        else:
            place_fields_filtered = self.pfs_df[self.pfs_df["category"] == "non-btsp"]
        criteria = ["has high gain", "has backwards shift", "has no drift"]
        criteria_counts = place_fields_filtered.groupby(criteria).count()["session id"].reset_index()

        def select_row(c1, is_c1, c2, is_c2, c3, is_c3):
            cond1 = criteria_counts[c1] if is_c1 else ~criteria_counts[c1]
            cond2 = criteria_counts[c2] if is_c2 else ~criteria_counts[c2]
            cond3 = criteria_counts[c3] if is_c3 else ~criteria_counts[c3]
            try:
                return criteria_counts[(cond1) & (cond2) & (cond3)]["session id"].iloc[0] / criteria_counts["session id"].sum()
            except IndexError:  # no element found means no such combo existed
                return 0

        gain, shift, drift = criteria
        subsets = (
            select_row(gain, True, shift, False, drift, False),  # Set 1
            select_row(gain, False, shift, True, drift, False),  # Set 2
            select_row(gain, True, shift, True, drift, False),  # Set 1n2
            select_row(gain, False, shift, False, drift, True),  # Set 3
            select_row(gain, True, shift, False, drift, True),  # Set 1n3
            select_row(gain, False, shift, True, drift, True),  # Set 2n3
            select_row(gain, True, shift, True, drift, True) if with_btsp else 0,  # Set 1n2n3
        )
        plt.figure(figsize=(5, 3), dpi=100)
        none_satisfied = select_row(gain, False, shift, False, drift, False)
        plt.text(0.6, -0.3, f"none: {np.round(100 * none_satisfied, 2)}%")
        v = venn3_unweighted(subsets, set_labels=criteria,
                             subset_label_formatter=lambda label: f"{np.round(100 * label, 2)}%")

        norm = matplotlib.colors.Normalize(vmin=min(subsets), vmax=max(subsets), clip=True)
        cmap = cmasher.get_sub_cmap('Reds', 0.1, 0.6)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        patch_ids = ['100', '010', '110', '001', '101', '011', '111']
        for i_patch_id, patch_id in enumerate(patch_ids):
            patch = v.get_patch_by_id(patch_id)
            patch.set_color(mapper.to_rgba(subsets[i_patch_id]))

        if with_btsp:
            filename = f"{self.output_root}/plot_place_fields_criteria_venn_diagram_with_btsp.pdf"
        else:
            filename = f"{self.output_root}/plot_place_fields_criteria_venn_diagram.pdf"
        plt.savefig(filename)

    def calc_shift_gain_distribution(self):
        shift_gain_df = self.pfs_df[["category", "newly formed", "initial shift", "formation gain", "formation rate sum"]].reset_index(drop=True)
        shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"].notna()) & (shift_gain_df["formation gain"].notna())]
        shift_gain_df["log10(formation gain)"] = np.log10(shift_gain_df["formation gain"])
        self.shift_gain_df = shift_gain_df.reset_index(drop=True)

    def plot_highgain_vs_lowgain_shift_diffs(self):
        def take_same_sized_subsample(df):
            n = min(df[df["newly formed"] == True].shape[0], df[df["newly formed"] == False].shape[0])
            ss1 = df[df["newly formed"] == True].sample(n=n)
            ss2 = df[df["newly formed"] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        sg_df = self.shift_gain_df

        sg_df_hg = sg_df[sg_df["log10(formation gain)"] > 0]
        sg_df_hg = take_same_sized_subsample(sg_df_hg)

        sg_df_lg = sg_df[sg_df["log10(formation gain)"] <= 0]
        sg_df_lg = take_same_sized_subsample(sg_df_lg)

        n = min(sg_df_lg.shape[0], sg_df_hg.shape[0])
        sg_df_hg = sg_df_hg.sample(n)
        sg_df_lg = sg_df_lg.sample(n)

        n_subsample = n // 10

        shift_diffs_hg = []
        for seed in range(1000):
            sg_df_subsample = sg_df_hg.sample(n=n_subsample, random_state=seed)
            mean_newlyf = sg_df_subsample[sg_df_subsample['newly formed'] == True]["initial shift"].mean()
            mean_stable = sg_df_subsample[sg_df_subsample['newly formed'] == False]["initial shift"].mean()
            shift_diff = mean_newlyf - mean_stable
            shift_diffs_hg.append(shift_diff)
        shift_diffs_lg = []
        for seed in range(1000):
            sg_df_subsample = sg_df_lg.sample(n=n_subsample, random_state=seed)
            mean_newlyf = sg_df_subsample[sg_df_subsample['newly formed'] == True]["initial shift"].mean()
            mean_stable = sg_df_subsample[sg_df_subsample['newly formed'] == False]["initial shift"].mean()
            shift_diff = mean_newlyf - mean_stable
            shift_diffs_lg.append(shift_diff)
        fig, axs = plt.subplots(2, 1, sharex=True)
        sns.kdeplot(data=shift_diffs_hg, ax=axs[0], color="k")
        sns.kdeplot(data=shift_diffs_lg, ax=axs[1], color="k")
        axs[0].set_title("shift diffs (high gain)")
        axs[1].set_title("shift diffs (low gain)")
        axs[1].set_xlim([-2.5, 2.5])
        axs[0].axvline(0, linestyle="--", color="red")
        axs[1].axvline(0, linestyle="--", color="red")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_highgain_vs_lowgain_shift_diffs.pdf")

    def plot_shift_gain_distribution(self):
        def take_same_sized_subsample(df):
            n = min(df[df["newly formed"] == True].shape[0], df[df["newly formed"] == False].shape[0])
            ss1 = df[df["newly formed"] == True].sample(n=n)
            ss2 = df[df["newly formed"] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        def run_tests_and_plot_results(df, g, params=None):
            sh = "shuffled_laps" if "shuffled_laps" in self.extra_info else ""
            test_results = f"{self.area}\n\n{sh}\n\n"
            for param in params:
                test_results += f"{param}\n"
                df_newlyF = df[df["newly formed"] == True]
                df_establ = df[df["newly formed"] == False]
                test = None
                for test_type in ["t", "u"]:
                    if test_type == "t":
                        test = scipy.stats.ttest_ind(df_newlyF[param].values, df_establ[param].values)
                    elif test_type == "u":
                        test = scipy.stats.mannwhitneyu(df_newlyF[param].values, df_establ[param].values)
                    test_results += f"    p_{test_type}={np.round(test.pvalue, 2)}\n"
                test_results += f"\n"
            ax = g.ax_joint
            #ax.set_aspect(1)
            ax.annotate(test_results, (1.25, 0.25), xycoords="axes fraction")

        sg_df = self.shift_gain_df
        ss_cc = take_same_sized_subsample(sg_df)
        plt.figure(figsize=(12,12), dpi=130)
        g = sns.jointplot(data=ss_cc, x="initial shift", y="log10(formation gain)",
                      hue="newly formed", alpha=0.2, s=10, marginal_kws={"common_norm": False})
        #plt.tight_layout()
        run_tests_and_plot_results(self.shift_gain_df, g=g, params=["initial shift", "log10(formation gain)"])
        plt.ylim([-1, 1])
        plt.xlim([-15, 15])
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")
        g.fig.set_size_inches((8,5.5))
        plt.subplots_adjust(right=0.75)
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions.pdf")
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions.png")
        plt.close()

        # high gain only
        sg_df_hg = sg_df[sg_df["log10(formation gain)"] > 0]
        ss_cc = take_same_sized_subsample(sg_df_hg)
        g = sns.jointplot(data=ss_cc, x="initial shift", y="log10(formation gain)",
                      hue="newly formed", alpha=0.2, s=10, marginal_kws={"common_norm": False})
        run_tests_and_plot_results(sg_df_hg, g=g, params=["initial shift"])
        plt.ylim([-1, 1])
        plt.xlim([-15, 15])
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")
        g.fig.set_size_inches((8,5.5))
        plt.subplots_adjust(right=0.75)
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions_highgain.pdf")
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions_highgain.png")
        plt.close()

        # low gain only
        sg_df_lg = sg_df[sg_df["log10(formation gain)"] <= 0]
        ss_cc = take_same_sized_subsample(sg_df_lg)
        g = sns.jointplot(data=ss_cc, x="initial shift", y="log10(formation gain)",
                      hue="newly formed", alpha=0.2, s=10, marginal_kws={"common_norm": False})
        run_tests_and_plot_results(sg_df_lg, g=g, params=["initial shift"])
        plt.ylim([-1, 1])
        plt.xlim([-15, 15])
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")
        g.fig.set_size_inches((8,5.5))
        plt.subplots_adjust(right=0.75)
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions_lowgain.pdf")
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions_lowgain.png")
        plt.close()

        # plot gain-FL distro
        s=0.9
        fig, ax1 = plt.subplots(figsize=(s*4.5, s*3.5), dpi=150)
        ax2 = ax1.twinx()
        #pfs_df_filt = self.pfs_df[self.pfs_df["end lap"] - self.pfs_df["formation lap"] > 15]
        pfs_df_filt = self.pfs_df
        FL_gain_df = pfs_df_filt[["formation lap", "formation gain"]].reset_index(drop=True)
        FL_gain_df = FL_gain_df[(FL_gain_df["formation lap"].notna()) & (FL_gain_df["formation gain"].notna())]
        FL_gain_df["log10(formation gain)"] = np.log10(FL_gain_df["formation gain"])
        sns.histplot(FL_gain_df, x="formation lap", ax=ax2, binwidth=1, color="orange", alpha=0.25)
        sns.pointplot(FL_gain_df, x="formation lap", y="log10(formation gain)", ax=ax1)
        plt.xlim([0,15])
        ax1.set_ylim([-0.5, 0.5])
        ax2.set_ylim([0, 7500])
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_gain_FL_distribution.pdf")
        plt.savefig(f"{self.output_root}/plot_gain_FL_distribution_nofilt.png")
        plt.close()

        # plot FL-lifespan distro
        #fig, ax1 = plt.subplots()
        #self.pfs_df["lifespan"] = self.pfs_df["end lap"]# - self.pfs_df["formation lap"]
        #pfs_df_filt = self.pfs_df[self.pfs_df["formation lap"] > -1]
        #FL_LS_df = pfs_df_filt[["formation lap", "end lap"]].reset_index(drop=True)
        #FL_LS_df = FL_LS_df[(FL_LS_df["formation lap"].notna()) & (FL_LS_df["end lap"].notna())]
        #sns.violinplot(FL_LS_df, x="formation lap", y="end lap", ax=ax1, inner=None, native_scale=True)
        #plt.xlim([0,10])
        #plt.savefig(f"{self.output_root}/plot_FL_EL_distribution.pdf")
        #plt.close()

    def plot_distance_from_RZ_distribution(self, corr=0):
        # pfoi = place fields of interest
        pfoi = self.pfs_df[(self.pfs_df["corridor"] == 0) & (self.pfs_df["category"] != "unreliable")].reset_index(drop=True)
        pfoi["pf middle bin"] = pfoi["lower bound"] + (pfoi["upper bound"] - pfoi["lower bound"]) / 2

        # TODO: same as with all RZs, these are guesses
        if corr==0:
            RZ_middle_bin = 42
        elif corr==1:
            RZ_middle_bin = 65
        else:
            raise Exception("choose valid corridor (0 or 1)")

        # calculate PF middle bin distance from reward zone middle bin
        pfoi["distance from RZ"] = pfoi["pf middle bin"] - RZ_middle_bin

        # make distances from RZ symmetrical:
        pfoi = pfoi[(pfoi["distance from RZ"] > -20) & (pfoi["distance from RZ"] < 20)]

        shift_distance_df = pfoi[["newly formed", "initial shift", "distance from RZ"]].reset_index(drop=True)
        shift_distance_df = shift_distance_df[(shift_distance_df["initial shift"].notna())].reset_index(drop=True)

        gain_distance_df = pfoi[["newly formed", "formation gain", "distance from RZ"]].reset_index(drop=True)
        gain_distance_df = gain_distance_df[(gain_distance_df["formation gain"].notna())].reset_index(drop=True)

        prop_stable_afterRZ_to_stable_beforeRZ = pfoi[(pfoi["newly formed"] == False) & (pfoi["distance from RZ"] > 0)].shape[0] / pfoi[(pfoi["newly formed"] == False) & (pfoi["distance from RZ"] < 0)].shape[0]
        prop_newlyf_afterRZ_to_newlyf_beforeRZ = pfoi[(pfoi["newly formed"] == True) & (pfoi["distance from RZ"] > 0)].shape[0] / pfoi[(pfoi["newly formed"] == True) & (pfoi["distance from RZ"] < 0)].shape[0]
        print(f"proportion of 'non-NF PFs after RZ' to 'non-NF PFs before RZ' = {prop_stable_afterRZ_to_stable_beforeRZ}")
        print(f"proportion of 'newlyF PFs after RZ' to 'newlyF PFs before RZ' = {prop_newlyf_afterRZ_to_newlyf_beforeRZ}")

        # g = sns.jointplot(data=gain_distance_df, y="formation gain", x="distance from RZ",
        #                  hue="newly formed", alpha=1, s=3, marginal_kws={"common_norm": False})
        plt.figure()
        sns.histplot(data=pfoi, x="distance from RZ", hue="newly formed", multiple="dodge", binwidth=1)
        plt.savefig(f"{self.output_root}/plot_distance_from_RZ_distribution.pdf")

    def plot_no_shift_criterion(self, noshift_path=None):
        if noshift_path == None:  # if no path was provided then we don't want this comparison
            return

        def get_pf_counts(df):
            return df[(df["category"] == "non-btsp") | (df["category"] == "btsp")].groupby("category").count()["session id"]

        pfs_Yshift = self.pfs_df
        pfs_Nshift = None
        for animal in self.animals:
            pfs_Nshift_animal = pd.read_pickle(f"{noshift_path}/{animal}_place_fields_df.pickle")
            pfs_Nshift = grow_df(pfs_Nshift, pfs_Nshift_animal)

        counts_Nshift_CA1 = get_pf_counts(pfs_Nshift)
        counts_Yshift_CA1 = get_pf_counts(pfs_Yshift)

        counts_CA1 = pd.DataFrame({
            "with shift": counts_Yshift_CA1 / counts_Yshift_CA1.sum(),
            "without shift": counts_Nshift_CA1 / counts_Yshift_CA1.sum(),
        })

        factor = 0.8
        fig, ax = plt.subplots(figsize=(2 * factor, 4 * factor), dpi=120)
        ax = counts_CA1.T.plot.bar(stacked=True, rot=45, color=["#be70ff", "#ff000f"], width=0.9, legend=False, ax=ax)
        ax.spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/no_shift_criterion_plot.pdf")

    def plot_cv_com_distro(self):
        sns.histplot(self.pfs_df["CV(COMs)"], binrange=[0, np.max(self.pfs_df["CV(COMs)"])])
        plt.savefig(f"{self.output_root}/plot_cv_com_distro.pdf")

        # distribution of place field categories by CV(COM) range
        pfoi = self.pfs_df[["category", "CV(COMs)"]]
        cv_com_bounds = np.arange(0,9.5,0.5)
        pfoi_catcounts_all = None
        for i in range(len(cv_com_bounds)-1):
            lb = cv_com_bounds[i]
            ub = cv_com_bounds[i+1]
            pfoi_filt = pfoi[(pfoi["CV(COMs)"] > lb) & (pfoi["CV(COMs)"] < ub)]
            pfoi_catcounts = pfoi_filt.groupby("category").count().T.reset_index()
            pfoi_catcounts["index"] = f"{lb}-{ub}"
            pfoi_catcounts_all = grow_df(pfoi_catcounts_all, pfoi_catcounts)

        fig, axs = plt.subplots(2,1, sharex=True)
        pfoi_catcounts_all = pfoi_catcounts_all[["index", "early", "transient", "non-btsp", "btsp"]]
        pfoi_catcounts_all.plot(kind="bar", stacked=True, color=["cyan", "orange", "red", "purple"], x="index", ax=axs[0])

        pfoi_catcounts_all["p_early"] = pfoi_catcounts_all["early"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all["p_transient"] = pfoi_catcounts_all["transient"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all["p_non-btsp"] = pfoi_catcounts_all["non-btsp"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all["p_btsp"] = pfoi_catcounts_all["btsp"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all_p = pfoi_catcounts_all[["index", "p_early", "p_transient", "p_non-btsp", "p_btsp"]]

        pfoi_catcounts_all_p.plot(kind="bar", stacked=True, color=["cyan", "orange", "red", "purple"], x="index", ax=axs[1])
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_cv_com_distro.pdf")
        plt.close()

    def plot_formation_rate_sum_hist_by_categories(self):
        # distribution of place field categories by FRS range
        pfoi = self.pfs_df[["category", "formation rate sum"]]
        pfoi = pfoi[pfoi["formation rate sum"] > 0]
        pfoi["log10(formation rate sum)"] = np.log10(pfoi["formation rate sum"])
        pfoi = pfoi[["category", "log10(formation rate sum)"]]
        logfrs_bounds = np.arange(0.1,3.2,0.2)
        logfrs_bounds = [np.round(val,1) for val in logfrs_bounds]
        pfoi_catcounts_all = None
        for i in range(len(logfrs_bounds)-1):
            lb = logfrs_bounds[i]
            ub = logfrs_bounds[i+1]
            pfoi_filt = pfoi[(pfoi["log10(formation rate sum)"] > lb) & (pfoi["log10(formation rate sum)"] < ub)]
            pfoi_catcounts = pfoi_filt.groupby("category").count().T.reset_index()
            pfoi_catcounts["index"] = f"{lb}-{ub}"
            pfoi_catcounts_all = grow_df(pfoi_catcounts_all, pfoi_catcounts)

        fig, axs = plt.subplots(2,1, sharex=True)
        pfoi_catcounts_all = pfoi_catcounts_all[["index", "early", "transient", "non-btsp", "btsp"]]
        pfoi_catcounts_all.plot(kind="bar", stacked=True, color=["cyan", "orange", "red", "purple"], x="index", ax=axs[0])

        pfoi_catcounts_all["p_early"] = pfoi_catcounts_all["early"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all["p_transient"] = pfoi_catcounts_all["transient"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all["p_non-btsp"] = pfoi_catcounts_all["non-btsp"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all["p_btsp"] = pfoi_catcounts_all["btsp"] / pfoi_catcounts_all[["early", "transient", "non-btsp", "btsp"]].sum(axis=1)
        pfoi_catcounts_all_p = pfoi_catcounts_all[["index", "p_early", "p_transient", "p_non-btsp", "p_btsp"]]

        pfoi_catcounts_all_p.plot(kind="bar", stacked=True, color=["cyan", "orange", "red", "purple"], x="index", ax=axs[1])
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #for index, label in enumerate(axs[1].xaxis.get_ticklabels()):
        #    if index % 2 != 0:
        #        label.set_visible(False)
        #plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_formation_rate_sum_hist_by_categories.pdf")
        plt.close()

    def plot_example_place_fields_by_cv_com(self):
        session_id = "KS030_110521"
        tcl_path = rf"C:\Users\martin\home\phd\btsp_project\analyses\manual\tuned_cells\CA1\tuned_cells_{session_id}.pickle"
        with open(tcl_path, "rb") as tcl_file:
            tcl = pickle.load(tcl_file)

        def get_tc(session_id, cellid, corridor):
            return [tc for tc in tcl if tc.sessionID == session_id and tc.cellid == cellid and tc.corridor == corridor][0]

        def get_pf_color(pf):
            if pf["category"] == "early":
                pf_color = "cyan"
            elif pf["category"] == "transient":
                pf_color = "orange"
            elif pf["category"] == "non-btsp":
                pf_color = "red"
            elif pf["category"] == "btsp":
                pf_color = "purple"
            else:
                pf_color = "black"
            return pf_color

        pfoi = self.pfs_df[["animal id", "session id", "cell id", "corridor", "category", "lower bound", "upper bound", "formation lap", "end lap", "CV(COMs)"]]
        pfoi = pfoi[pfoi["category"] != "unreliable"]
        pfoi = pfoi[pfoi["end lap"] - pfoi["formation lap"] > 10]

        cv_com_ranges = [
            [0, 0.5],
            [0.5, 1.0],
            [1.0, 1.5],
            [1.5, 2.0],
            [2.0, 2.5],
            [2.5, 3.0],
            [3.0, 3.5],
            [3.5, 4.0],
            [4.0, 4.5],
            [4.5, 5.0]
        ]
        X = len(cv_com_ranges)
        Y = 10
        fig, axs = plt.subplots(X,Y, sharex=True, sharey=True)
        x = 0
        for cv_com_lb, cv_com_ub in cv_com_ranges:
            pfoi_range = pfoi[(pfoi["CV(COMs)"] > cv_com_lb) & (pfoi["CV(COMs)"] < cv_com_ub)]
            pfoi_range = pfoi_range[pfoi_range["session id"] == session_id]
            N = Y if len(pfoi_range) > Y else len(pfoi_range)
            pfoi_sample = pfoi_range.sample(n=N, random_state=0)
            y = 0
            for _, pf in pfoi_sample.iterrows():
                tc = get_tc(pf["session id"], pf["cell id"], pf["corridor"])
                rate_matrix = tc.rate_matrix
                sns.heatmap(rate_matrix.T, ax=axs[x,y], cbar=False, cmap="Greys")
                axs[x,y].invert_yaxis()
                if y == 0:
                    axs[x,y].set_ylabel(f"({cv_com_lb}, {cv_com_ub})", rotation=0)
                axs[x,y].axhline(y=0, c="k")
                axs[x,y].axhline(y=rate_matrix.T.shape[0]-0.1, c="k")
                axs[x,y].axvline(x=0, c="k")
                axs[x,y].axvline(x=rate_matrix.T.shape[1]-0.1, c="k")
                axs[x,y].tick_params(left=False, bottom=False)

                # overlay place field
                pf_color = get_pf_color(pf)
                lb = pf["lower bound"]
                ub = pf["upper bound"]
                fl = pf["formation lap"]
                el = pf["end lap"]
                axs[x,y].axvspan(xmin=lb, xmax=ub, ymin=fl/rate_matrix.T.shape[0], ymax=el/rate_matrix.T.shape[0],
                                 color=pf_color, alpha=0.1)
                axs[x,y].get_yaxis().set_ticks([])
                y += 1
            x += 1
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(f"{self.output_root}/plot_example_place_fields_by_cv_com.pdf")
        plt.close()

    def plot_formation_rate_sum(self, log=True):
        """
        frs_df = self.pfs_df[["newly formed", "formation lap", "initial shift", "has backwards shift", "formation gain", "formation rate sum"]].reset_index(drop=True)
        frs_df = frs_df[frs_df["formation lap"] >= 0]  # throw away unreliable pfs
        frs_df = frs_df[frs_df["formation rate sum"] >= 0]
        frs_df["log10(formation gain)"] = np.log10(frs_df["formation gain"])
        frs_df["log10(formation rate sum)"] = np.log10(frs_df["formation rate sum"])
        frs_df["has negative shift"] = frs_df["initial shift"] < 0
        frs_df["has large negative shift (<-2)"] = frs_df["initial shift"] < -2
        frs_df["FL = 0"] = frs_df["formation lap"] == 0
        """
        #plt.figure(figsize=(5,2.5))
        #sns.histplot(self.pfs_df[["formation rate sum"]], bins=np.arange(0,1000,20))
        #plt.xlim([0,990])
        #plt.savefig(f"{self.output_root}/formation_rate_sum_hist.pdf")
        #plt.close()
        #plt.show()

        #### PLOT NEWLY FORMED (REGARDLESS OF SHIFT SIZE)
        #n = min(frs_df[frs_df["newly formed"] == True].shape[0], frs_df[frs_df["newly formed"] == False].shape[0])
        #ss1 = frs_df[frs_df["newly formed"] == True].sample(n=n)
        #ss2 = frs_df[frs_df["newly formed"] == False].sample(n=n)
        #ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))

        #sns.jointplot(ss_cc, x="initial shift", y="log10(formation rate sum)", hue="newly formed", marginal_kws={"common_norm": False})
        #plt.xlim([-15, 15])
        #plt.ylim([0,3.5])

        #sns.jointplot(ss_cc, x="formation gain", y="formation rate sum", hue="newly formed", marginal_kws={"common_norm": False, "log_scale": True})
        #plt.ylim([1,3000])
        #plt.xlim([0.1,10])
        #plt.show()

        #### PLOT SHIFT SIZE (REGARDLESS OF NEWLY FORMED)
        #n = min(frs_df[frs_df["has high shift"] == True].shape[0], frs_df[frs_df["has high shift"] == False].shape[0])
        #ss1 = frs_df[frs_df["has high shift"] == True].sample(n=n)
        #ss2 = frs_df[frs_df["has high shift"] == False].sample(n=n)
        #ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))

        #sns.jointplot(ss_cc, x="initial shift", y="log10(formation rate sum)", hue="has high shift", marginal_kws={"common_norm": False})
        #plt.xlim([-15, 15])
        #plt.ylim([0,3.5])

        #sns.jointplot(ss_cc, x="formation gain", y="formation rate sum", hue="has high shift", marginal_kws={"common_norm": False, "log_scale": True})
        #plt.ylim([1,3000])
        #plt.xlim([0.1,10])
        #plt.show()

        """
        #### PLOT NEWLY FORMED AND HIGH SHIFT
        frs_df_nf = frs_df[frs_df["FL = 0"] == True]
        frs_df_nf = frs_df_nf[frs_df_nf["initial shift"].notna()]

        n = min(frs_df_nf[frs_df_nf["has negative shift"] == True].shape[0], frs_df_nf[frs_df_nf["has negative shift"] == False].shape[0])
        ss1 = frs_df_nf[frs_df_nf["has negative shift"] == True].sample(n=n)
        ss2 = frs_df_nf[frs_df_nf["has negative shift"] == False].sample(n=n)
        ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))

        sns.jointplot(ss_cc, x="formation gain", y="formation rate sum", hue="has negative shift",
                      marginal_kws={"common_norm": True, "log_scale": True})
        plt.ylim([1, 3000])
        plt.xlim([0.1, 10])
        plt.suptitle(f"established, n={n}")

        #### PLOT ESTABLISHED AND HIGH SHIFT
        frs_df_nf = frs_df[frs_df["FL = 0"] == False]
        frs_df_nf = frs_df_nf[frs_df_nf["initial shift"].notna()]

        n = min(frs_df_nf[frs_df_nf["has negative shift"] == True].shape[0], frs_df_nf[frs_df_nf["has negative shift"] == False].shape[0])
        ss1 = frs_df_nf[frs_df_nf["has negative shift"] == True].sample(n=n)
        ss2 = frs_df_nf[frs_df_nf["has negative shift"] == False].sample(n=n)
        ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))

        sns.jointplot(ss_cc, x="formation gain", y="formation rate sum", hue="has negative shift",
                      marginal_kws={"common_norm": True, "log_scale": True})
        plt.ylim([1, 3000])
        plt.xlim([0.1, 10])
        plt.suptitle(f"newly formed, n={n}")
        plt.show()
        """

        for newly_formed in [True, False]:
            frs_df = self.pfs_df[["newly formed", "initial shift", "formation gain", "formation rate sum"]].reset_index(drop=True)
            frs_df["has high shift"] = frs_df["initial shift"] < -2
            frs_df["log10(formation gain)"] = np.log10(frs_df["formation gain"])
            frs_df["log10(FRS)"] = np.log10(frs_df["formation rate sum"])
            #frs_df = frs_df[(frs_df["formation gain"].notna()) & (frs_df["log10(formation gain)"].notna())]
            frs_df = frs_df.reset_index(drop=True)
            frs_df = frs_df[frs_df["newly formed"] == newly_formed]
            #frs_df = frs_df[(frs_df["initial shift"] < 3) & (frs_df["initial shift"] > -3)]
            #if newly_formed:
            #    colors = ["#C24900", "#FFB403"]
            #else:
            #    colors = ["#12476B", "#1FD4FF"]
            #sns.jointplot(data=frs_df, x="formation rate sum", y="formation gain",
            #              hue="has high shift", alpha=1, s=10, marginal_kws={"log_scale": log},
            #              palette=colors)
            n = min(frs_df[(frs_df["initial shift"] > 0)].shape[0], frs_df[(frs_df["initial shift"] <= 0)].shape[0])
            ss1 = frs_df[(frs_df["initial shift"] > 0)].sample(n=n)
            ss2 = frs_df[(frs_df["initial shift"] <= 0)].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            #sns.jointplot(data=ss_cc, x="formation rate sum", y="formation gain", palette="cool", kind="scatter",
            #              hue="initial shift", alpha=0.33, s=50, marginal_kws={"log_scale": log})
            sns.jointplot(data=frs_df, x="initial shift", y="log10(formation gain)", palette="cool", kind="scatter",
                          hue="log10(FRS)", alpha=0.33, s=50)
            plt.ylim([-1, 1])
            plt.xlim([-10, 10])
            plt.axvline(x=0, c="k", linestyle="--")
            plt.axhline(y=0, c="k", linestyle="--")

            #frs_df_nhhs = frs_df[frs_df["has high shift"] == False]
            #plt.plot(frs_df_nhhs["formation rate sum"].mean(), frs_df_nhhs["formation gain"].mean(), markersize=10, marker="o", markerfacecolor=colors[0], markeredgecolor="black")
            #frs_df_hhs = frs_df[frs_df["has high shift"] == True]
            #plt.plot(frs_df_hhs["formation rate sum"].mean(), frs_df_hhs["formation gain"].mean(), markersize=10, marker="o", markerfacecolor=colors[1], markeredgecolor="black")
            #plt.xlim([1,2000])
            #plt.ylim([0.1, 10])
            #plt.xlim([10,2000])
            #plt.ylim([0.4, 10])
            plt.suptitle(f"newly formed = {newly_formed}")
            plt.tight_layout()
            plt.savefig(f"{self.output_root}/plot_formation_rate_sum_NF={newly_formed}_SG.pdf")
            plt.close()

    def calc_tests(self):
        params = ["initial shift", "log10(formation gain)"]
        for param in params:
            sg_df_newlyF = self.shift_gain_df[self.shift_gain_df["newly formed"] == True]
            sg_df_establ = self.shift_gain_df[self.shift_gain_df["newly formed"] == False]
            test = scipy.stats.mannwhitneyu(sg_df_newlyF[param].values, sg_df_establ[param].values)
            print(f"{self.area} {param}:\tu={np.round(test.statistic, 3)},\tp={np.round(test.pvalue, 3)}")

    def export_results(self, export_path=""):
        path = f"{self.output_root}/results.json"
        if export_path:
            path = export_path

        sg_newlyf = self.shift_gain_df[self.shift_gain_df["newly formed"] == True]
        sg_establ = self.shift_gain_df[self.shift_gain_df["newly formed"] == False]
        results_dict = {
            "mean(shift) newlyF": sg_newlyf["initial shift"].mean(),
            "mean(shift) establ": sg_establ["initial shift"].mean(),
            "median(shift) newlyF": sg_newlyf["initial shift"].median(),
            "median(shift) establ": sg_establ["initial shift"].median(),
            "mean(gain) newlyF": sg_newlyf["formation gain"].mean(),
            "mean(gain) establ": sg_establ["formation gain"].mean(),
            "median(gain) newlyF": sg_newlyf["formation gain"].median(),
            "median(gain) establ": sg_establ["formation gain"].median(),
            "mean(log10(gain)) newlyF": sg_newlyf["log10(formation gain)"].mean(),
            "mean(log10(gain)) establ": sg_establ["log10(formation gain)"].mean(),
            "median(log10(gain)) newlyF": sg_newlyf["log10(formation gain)"].median(),
            "median(log10(gain)) establ": sg_establ["log10(formation gain)"].median(),
        }
        with open(path, "w") as results_file:
            json.dump(results_dict, results_file)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
    parser.add_argument("-dp", "--data-path", required=True)
    parser.add_argument("-op", "--output-path")
    parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
    #parser.add_argument("-np", "--no-shift-path")
    args = parser.parse_args()

    area = args.area
    data_path = args.data_path
    output_path = args.output_path
    extra_info = args.extra_info
    #noshift_path = args.no_shift_path

    #run analysis
    btsp_statistics = BtspStatistics(area, data_path, output_path, extra_info)
    btsp_statistics.load_data()
    btsp_statistics.calc_place_field_proportions()
    btsp_statistics.calc_shift_gain_distribution()
    #btsp_statistics.plot_cells()  # TODO: this needs cell stats saved
    #btsp_statistics.plot_place_fields()
    #btsp_statistics.plot_place_fields_by_session()
    #btsp_statistics.plot_place_field_proportions()
    #btsp_statistics.plot_place_fields_criteria()
    #btsp_statistics.plot_place_field_properties()
    #btsp_statistics.plot_place_field_heatmap()  # TODO: corridor nincs a self.pf_dfs-ben -- elvileg mar van
    #btsp_statistics.plot_place_fields_criteria_venn_diagram()
    btsp_statistics.plot_shift_gain_distribution()
    btsp_statistics.plot_highgain_vs_lowgain_shift_diffs()
    #btsp_statistics.plot_distance_from_RZ_distribution()  # TODO: ZeroDivisionError
    #btsp_statistics.plot_no_shift_criterion(noshift_path=noshift_path)
    #btsp_statistics.plot_cv_com_distro()
    #btsp_statistics.plot_example_place_fields_by_cv_com()
    #btsp_statistics.export_results()
    #btsp_statistics.plot_formation_rate_sum()
    #btsp_statistics.plot_formation_rate_sum_hist_by_categories()
    #btsp_statistics.plot_formation_rate_sum(log=False)
