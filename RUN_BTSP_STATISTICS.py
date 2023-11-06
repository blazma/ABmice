import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_unweighted
import matplotlib
import seaborn as sns
import cmasher
import os
import scipy
import argparse
from datetime import datetime
from collections import namedtuple

ANIMALS = {
    "CA1": ["KS028",
            "KS029",
            "KS030",
            "srb131"],
    "CA3": ["srb231",
            "srb251",
            "srb269",
            "srb270"]
}
SESSIONS_TO_IGNORE = {
    "CA1": [#'KS028_110521',  # error
            'KS029_110321',   # no p95
            'KS029_110721',   # error
            'KS029_110821',   # error
            'KS029_110521',   # error
            'KS030_110721',   # error
            'srb131_211019'], # reshuffles
    "CA3": []
}
CORRIDORS = [14, 15,  # random
             16, 18,  # block
             17]  # new environment

Category = namedtuple('Category', ["order", "color"])
CATEGORIES = {
    "unreliable": Category(0, "#a2a1a1"),
    "early": Category(1, "#00e3ff"),
    "transient": Category(2, "#ffe000"),
    "non-btsp": Category(3, "#ff000f"),
    "btsp": Category(4, "#be70ff"),
}


def grow_df(df_a, df_b):
    if df_a is None:
        df_a = df_b
    else:
        df_a = pd.concat((df_a, df_b))
    return df_a


def makedir_if_needed(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class BtspStatistics:
    def __init__(self, area, data_path, output_path, extra_info=""):
        self.area = area
        self.animals = ANIMALS[area]
        self.data_path = data_path
        self.date = datetime.today().strftime("%y%m%d")
        self.categories_colors_RGB = [category.color for _, category in CATEGORIES.items()]
        self.btsp_criteria = ["has high gain", "has backwards shift", "has no drift"]
        self.reward_zones = {  # TODO: these are guesses based on plots, needs to be exact
            0: [38, 46],  # corridor 14
            1: [61, 69]  # corridor 15
        }

        # set output folder
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/BTSP_plots_{self.area}_{self.date}{self.extra_info}"
        makedir_if_needed(self.output_root)

        # dataframes for place fields and cell statistics
        self.pfs_df = None
        self.cell_stats_df = None

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
                pfs_df_animal = pd.read_pickle(f"{self.data_path}/{animal}_place_fields_df.pickle")
                cell_stats_df_animal = pd.read_pickle(f"{self.data_path}/{animal}_cell_stats_df.pickle")
            except Exception:
                print(f"ERROR occured for animal {animal} during DF pickle loading; skipping")
                continue
            self.cell_stats_df = grow_df(self.cell_stats_df, cell_stats_df_animal)
            self.pfs_df = grow_df(self.pfs_df, pfs_df_animal)
        self.pfs_df = self.pfs_df.reset_index().drop("index", axis=1)
        self.cell_stats_df = self.cell_stats_df.reset_index().drop("index", axis=1)

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
        axs[0].set_ylim([0, 250])
        # axs[0].set_yticks(np.linspace(0,pfs_by_category["session id"].values.max(),num=5))
        axs[0].spines[['right', 'top']].set_visible(False)
        # plt.suptitle(f"[{area}] Distribution of place fields by PF categories across all animals")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_place_fields.pdf")

    def plot_place_fields(self):
        fig, ax = plt.subplots()
        pfs_by_category = self.pfs_df.groupby(["category", "session id", "category_order"]).count().sort_values(by="category_order")
        pf_counts = pfs_by_category.reset_index()[["category", "session id", "cell id"]]
        sns.boxplot(data=pf_counts, x="category", y="cell id", ax=ax, palette=self.categories_colors_RGB, showfliers=False)
        sns.swarmplot(data=pf_counts, x="category", y="cell id", ax=ax, color="black", alpha=0.5)
        plt.title(f"[{self.area}] Number of various place fields for each session")
        ax.set_ylabel("# place fields")
        ax.set_xlabel("")
        plt.savefig(f"{self.output_root}/plot_place_fields.pdf")

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

    def plot_shift_gain_distribution(self):
        shift_gain_df = self.pfs_df[["newly formed", "initial shift", "formation gain"]].reset_index(drop=True)
        shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"].notna()) & (shift_gain_df["formation gain"].notna())]

        # plot all PFs
        g = sns.jointplot(data=shift_gain_df.reset_index(drop=True), x="initial shift", y="formation gain",
                          hue="newly formed", alpha=1, s=3, marginal_kws={"common_norm": False})
        g.ax_joint.set_yscale("log")
        plt.ylim([0.1,10])
        plt.xlim([-15, 15])
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=1, c="k", linestyle="--")
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions.pdf")

        # plot only newly formed PFs
        shift_gain_df_CA1 = shift_gain_df[shift_gain_df["newly formed"] == True]
        g = sns.jointplot(data=shift_gain_df_CA1.reset_index(drop=True), x="initial shift", y="formation gain",
                          alpha=1, s=3, marginal_kws={"common_norm": False}, color="orange")
        g.ax_joint.set_yscale("log")
        plt.ylim([0.1,10])
        plt.xlim([-15, 15])
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=1, c="k", linestyle="--")
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions_newlyF_only.pdf")

        # subsampling
        if self.area == "CA3":  # we want to subsample only the CA1 PFs and compare against CA3
            return

        shift_diffs = []
        gain_diffs = []
        for seed in range(1000):
            shift_gain_df_subs = shift_gain_df.sample(n=274, random_state=seed)

            mean_shift_newlyf = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == True]['initial shift'].mean()
            mean_shift_stable = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == False]['initial shift'].mean()
            shift_diff = mean_shift_newlyf - mean_shift_stable
            shift_diffs.append(shift_diff)

            mean_gain_newlyf = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == True]['formation gain'].mean()
            mean_gain_stable = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == False]['formation gain'].mean()
            gain_diff = mean_gain_newlyf - mean_gain_stable
            gain_diffs.append(gain_diff)
        fig, axs = plt.subplots(2,1)
        sns.kdeplot(data=shift_diffs, ax=axs[0], color="k")
        sns.kdeplot(data=gain_diffs, ax=axs[1], color="k")
        axs[0].set_title("shift")
        axs[1].set_title("gain")

        # plot CA3 values on distributions
        shift_diff_CA3 = -0.091
        gain_diff_CA3 = 0.252
        axs[0].axvline(shift_diff_CA3, c="red", label="CA3")
        axs[1].axvline(gain_diff_CA3, c="red", label="CA3")

        # plot confidence intervals
        axs[0].axvline(np.percentile(shift_diffs, q=95), c="blue", label="CA1 95th p.")
        axs[0].axvline(np.percentile(shift_diffs, q=99), c="cyan", label="CA1 99th p.")
        axs[1].axvline(np.percentile(gain_diffs, q=5), c="blue", label="CA1 5th p.")
        axs[1].axvline(np.percentile(gain_diffs, q=1), c="cyan", label="CA1 1st p.")

        # add legends
        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_shift_gain_bootstrap_CA3.pdf")

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

    def plot_no_shift_criterion(self, noshift_path):
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

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
parser.add_argument("-dp", "--data-path", required=True)
parser.add_argument("-op", "--output-path", default=os.getcwd())
parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
parser.add_argument("-np", "--no-shift-path")
args = parser.parse_args()

area = args.area
data_path = args.data_path
output_path = args.output_path
extra_info = args.extra_info
noshift_path = args.no_shift_path

# run analysis
btsp_statistics = BtspStatistics(area, data_path, output_path, extra_info)
btsp_statistics.load_data()
btsp_statistics.calc_place_field_proportions()
btsp_statistics.plot_cells()
btsp_statistics.plot_place_fields()
btsp_statistics.plot_place_field_proportions()
btsp_statistics.plot_place_fields_criteria()
btsp_statistics.plot_place_field_properties()
btsp_statistics.plot_place_field_heatmap()
btsp_statistics.plot_place_fields_criteria_venn_diagram()
btsp_statistics.plot_shift_gain_distribution()
btsp_statistics.plot_distance_from_RZ_distribution()
btsp_statistics.plot_no_shift_criterion(noshift_path=noshift_path)
