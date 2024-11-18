import pandas as pd
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_unweighted
import matplotlib
import seaborn as sns
import cmasher
import os
import pickle
import argparse
import json
import pingouin

from BTSP.constants import ANIMALS, CATEGORIES, CATEGORIES_DARK, ANIMALS_PALETTE, AREA_PALETTE, BEHAVIOR_SCORE_THRESHOLD
from utils import grow_df, makedir_if_needed
import sklearn
import scipy
from BTSP.BehavioralStatistics import BehaviorStatistics

class BtspStatistics:
    def __init__(self, area, data_path, output_path, extra_info="", is_shift_criterion_on=True, is_notebook=False, history="ALL"):
        self.area = area
        self.animals = ANIMALS[area]
        self.data_path = data_path
        self.categories_colors_RGB = [category.color for _, category in CATEGORIES.items()]
        self.categories_colors_RGB_dark = [category.color for _, category in CATEGORIES_DARK.items()]
        self.btsp_criteria = ["has high gain", "has backwards shift", "has no drift"]
        self.reward_zones = {
            0: [38, 47],  # corridor 14
            1: [60, 69]  # corridor 15
        }
        self.bin_length = 2.13  # cm
        self._use_font()
        self.is_shift_criterion_on = is_shift_criterion_on
        self.is_notebook = is_notebook
        self.history = history

        # set output folder
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/statistics/{self.area}{self.extra_info}"
        if not self.is_shift_criterion_on:
            self.output_root = f"{self.output_root}_withoutShiftCriterion"
        if "historyDependent" in self.extra_info and history in ["CHANGE", "STAY"]:
            self.output_root = f"{self.output_root}_{self.history}"
            self.history_dependent = True
        else:
            self.history_dependent = False
        makedir_if_needed(f"{output_path}/statistics")
        makedir_if_needed(self.output_root)

        # dataframes for place fields and cell statistics
        self.pfs_df = None
        self.pfs_laps_df = None
        self.cell_stats_df = None
        self.shift_gain_df = None
        self.tc_df = None
        self.tests_df = None

        # "debug"
        self.long_pf_only = True

    def _use_font(self):
        #from matplotlib import font_manager
        #font_dirs = ['C:\\home\\phd\\']
        #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        #for font_file in font_files:
        #   font_manager.fontManager.addfont(font_file)
        #plt.rcParams['font.family'] = 'Trade Gothic Next LT Pro BdCn'

        from matplotlib import font_manager
        font_dirs = ['C:\\Users\\martin\\home\\phd\\misc']
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = 'Roboto'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Roboto'
        plt.rcParams['mathtext.it'] = 'Roboto'

    def load_data(self):
        # read place fields and cell statistics dataframes for each animal
        self.pfs_df = None
        self.cell_stats_df = None
        for animal in self.animals:
            try:
                pfs_df_animal = pd.read_pickle(f"{self.data_path}/place_fields/{self.area}{self.extra_info}/{animal}_place_fields_df.pickle")
                cell_stats_df_animal = pd.read_pickle(f"{self.data_path}/tuned_cells/{self.area}{self.extra_info}/cell_counts_{animal}.pickle")
            except Exception:
                print(f"ERROR occured for animal {animal} during DF pickle loading; skipping")
                continue
            self.cell_stats_df = grow_df(self.cell_stats_df, cell_stats_df_animal)
            self.pfs_df = grow_df(self.pfs_df, pfs_df_animal)
        self.pfs_df = self.pfs_df.reset_index().drop("index", axis=1)
        if not self.is_shift_criterion_on:
            # if shift criterion is not considered: reset all BTSP to non-BTSP PFs
            self.pfs_df.loc[self.pfs_df["category"] == "btsp", "category"] = "non-btsp"
            # then set those non-BTSP to BTSP who satisfy gain and drift only
            self.pfs_df.loc[(self.pfs_df["has high gain"] == True) & (self.pfs_df["has no drift"] == True), "category"] = "btsp"
        #if self.long_pf_only:
        #    self.pfs_df = self.pfs_df[self.pfs_df["end lap"] - self.pfs_df["formation lap"] > 15].reset_index()
        #self.cell_stats_df = self.cell_stats_df.reset_index().drop("index", axis=1)

        # filter by history
        if self.history_dependent:
            self.pfs_df = self.pfs_df[self.pfs_df["history"] == self.history].reset_index(drop=True)

        # assign category orders
        for category in CATEGORIES:
            cond = self.pfs_df["category"] == category
            self.pfs_df.loc[cond, "category_order"] = CATEGORIES[category].order

        # tag newly formed place fields
        self.pfs_df["newly formed"] = np.where(self.pfs_df["category"].isin(["transient", "non-btsp", "btsp"]), True, False)

        # calculate cell count ratios
        self.cell_stats_df["active / total"] = self.cell_stats_df["active cells"] / self.cell_stats_df["total cells"]
        self.cell_stats_df["tuned / total"] = self.cell_stats_df["tuned cells"] / self.cell_stats_df["total cells"]
        self.cell_stats_df["tuned / active"] = self.cell_stats_df["tuned cells"] / self.cell_stats_df["active cells"]

    def filter_low_behavior_score(self):
        # filter sessions where behavior score is lower than threshold -- to homogenize dataset
        if len(self.extra_info) > 0:
            extra_info = self.extra_info if self.extra_info[0] != "_" else self.extra_info[1:]
        else:
            extra_info = ""
        behav_stats = BehaviorStatistics(self.area, self.data_path, self.output_root, extra_info=extra_info, makedir=False)
        behav_stats.calc_behavior_score()
        behavior_scores = behav_stats.behavior_df[["sessionID", "behavior score"]]
        self.cell_stats_df = self.cell_stats_df.merge(behavior_scores, on="sessionID")
        self.cell_stats_df =  self.cell_stats_df[self.cell_stats_df["behavior score"] > BEHAVIOR_SCORE_THRESHOLD].reset_index()

        behavior_scores = behavior_scores.rename({"sessionID": "session id"}, axis=1)  # rename in order to do merge with pfs_df
        self.pfs_df = self.pfs_df.merge(behavior_scores, on="session id")
        self.pfs_df = self.pfs_df[self.pfs_df["behavior score"] > BEHAVIOR_SCORE_THRESHOLD].reset_index()

    def load_cell_data(self):
        def get_tc(session_id, cellid, corridor):
            return [tc for tc in tcl_all_sessions if tc.sessionID == session_id and tc.cellid == cellid and tc.corridor == corridor][0]

        sessions = np.unique(self.pfs_df["session id"].values)
        tcl_all_sessions = []
        for session in sessions:
            print(session)
            tcl_path = f"{self.data_path}/tuned_cells/{self.area}{self.extra_info}/tuned_cells_{session}.pickle"
            with open(tcl_path, "rb") as tcl_file:
                tcl = pickle.load(tcl_file)
                tcl_all_sessions += tcl
        if self.history_dependent:
            self.tc_df = pd.DataFrame([{"sessionID": tc.sessionID, "cellid": tc.cellid, "corridor": tc.corridor, "history": tc.history, "tc": tc} for tc in tcl_all_sessions])
        else:
            self.tc_df = pd.DataFrame([{"sessionID": tc.sessionID, "cellid": tc.cellid, "corridor": tc.corridor, "tc": tc} for tc in tcl_all_sessions])

    def calc_place_field_proportions(self):
        pf_proportions_by_category_df = None
        sessions = self.pfs_df["session id"].unique()
        for session in sessions:
            animalID, _, _ = session.partition("_")
            pfs_session = self.pfs_df.loc[self.pfs_df["session id"] == session]
            n_pfs_session_by_category = pfs_session.groupby("category").count()
            n_pfs_session_total = len(pfs_session)

            #if n_pfs_session_total < 10:
            #    print(f"{session} has less than 10 place fields in total...")
            #    continue

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

    def plot_cells(self, swarmplot=True):
        if self.is_notebook:
            scale = 0.5
            dpi = 150
        else:
            scale = 0.5
            dpi = 200

        # various proportions
        w, h = 16, 8
        w_poster, h_poster = 10, 18
        w_pres, h_pres = 10, 10

        #w, h = w_pres, h_pres
        scale_poster = 0.275
        scale = 0.4
        fig, ax = plt.subplots(figsize=(scale*w, scale*h), dpi=dpi)
        sns.boxplot(self.cell_stats_df[["total cells", "active cells", "tuned cells"]], color=AREA_PALETTE[self.area],
                    ax=ax, width=0.85, showfliers=False)
        if swarmplot:
            sns.swarmplot(data=self.cell_stats_df[["sessionID", "animalID", "total cells", "active cells", "tuned cells"]].melt(id_vars=["sessionID", "animalID"]),
                          x="variable", y="value", hue="animalID", s=5, alpha=0.85, palette=ANIMALS_PALETTE[self.area], ax=ax,
                          linewidth=1, dodge=True)  # 4 = number of animals (nr. of distinct hues
            ax.legend(loc='upper right', title="animals", bbox_to_anchor=(1.25, 1.35))
        ax.set_ylabel("# cells")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        ax.set_xlabel("")
        if self.area == "CA1":
            ax.set_ylim([0,2000])
            ax.set_yticks(np.linspace(0,2000,9))
        else:
            ax.set_ylim([0,200])
            ax.set_yticks(np.linspace(0,200,9))
        #plt.title(f"Number of cells in {self.area}")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_cells.pdf")
        plt.savefig(f"{self.output_root}/plot_cells.svg")
        plt.close()

        cmap = sns.color_palette('mako', as_cmap=True)
        norm = plt.Normalize(vmin=0, vmax=8)
        palette = {h: cmap(norm(h)) for h in self.cell_stats_df['behavior score']}

        fig, ax = plt.subplots(figsize=(scale*14, scale*8), dpi=dpi)
        sns.boxplot(self.cell_stats_df[["total cells", "active cells", "tuned cells"]], color=AREA_PALETTE[self.area],
                    ax=ax, width=0.8, showfliers=False)
        if swarmplot:
            ax = sns.swarmplot(data=self.cell_stats_df[["sessionID", "total cells", "active cells", "tuned cells", "behavior score"]].melt(id_vars=["sessionID", "behavior score"]),
                          x="variable", y="value", hue="behavior score", s=4, alpha=0.85, ax=ax, palette=palette,
                          linewidth=1, legend=False)  # 4 = number of animals (nr. of distinct hues
        ax.set_ylabel("# cells")
        ax.spines[['right', 'top']].set_visible(False)
        from matplotlib.cm import ScalarMappable
        plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)  # optionally add a colorbar
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        #ax.legend(loc='upper right', title="animals", bbox_to_anchor=(1.25, 1))
        ax.set_xlabel("")
        if self.area == "CA1":
            ax.set_ylim([0,1750])
            ax.set_yticks(np.linspace(0,1750,6))
        else:
            ax.set_ylim([0,200])
            ax.set_yticks(np.linspace(0,200,9))
        plt.title(f"Number of cells in {self.area}")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_cells_behavior_score.pdf")
        plt.savefig(f"{self.output_root}/plot_cells_behavior_score.svg")
        plt.close()

        fig, axs = plt.subplots(1, 3, figsize=(scale*18, scale*6), dpi=dpi)
        sns.scatterplot(data=self.cell_stats_df, x="behavior score", y="active / total", hue="animalID", ax=axs[0],
                        legend=False, palette=ANIMALS_PALETTE[self.area], edgecolor="black")
        sns.scatterplot(data=self.cell_stats_df, x="behavior score", y="tuned / total", hue="animalID", ax=axs[1],
                        legend=False, palette=ANIMALS_PALETTE[self.area], edgecolor="black")
        ax = sns.scatterplot(data=self.cell_stats_df, x="behavior score", y="tuned / active", hue="animalID", ax=axs[2],
                             palette=ANIMALS_PALETTE[self.area], edgecolor="black")
        ax.legend(loc='upper right', title="animals", bbox_to_anchor=(1.75, 1))
        axs[0].set_ylim([0,1])
        axs[1].set_ylim([0, 1])
        axs[2].set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_cell_ratios.pdf")
        plt.savefig(f"{self.output_root}/plot_cell_ratios.svg")
        plt.close()


    def plot_place_fields(self):
        if self.is_notebook:
            scale = 1
            dpi = 150
        else:
            scale = 0.75
            #scale = 1
            dpi=125
        fig, ax = plt.subplots(figsize=(scale*3, scale*3), dpi=dpi, num=1, clear=True)  # , gridspec_kw={"width_ratios": [1,3]}
        pfs_by_category = self.pfs_df.groupby(["category", "category_order"]).count().sort_values(by="category_order")
        pfs_by_category = pfs_by_category.reset_index().set_index("category")  # so we can get rid of the category ordering from the label names
        pfs_by_category.plot(kind="bar", y="session id", ax=ax, color=self.categories_colors_RGB, legend=False)
        if self.area == "CA1":
            ax.set_yticks(np.linspace(0,8000,9))
        else:
            ax.set_yticks(np.linspace(0,500,5))
            if "noCaThreshold" in self.extra_info:
                ax.set_yticks(np.linspace(0, 800, 9))
        ax.set_title(f"n = {len(self.pfs_df)}")
        ax.set_ylabel("# place fields")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        ax.spines[['right', 'top']].set_visible(False)
        #plt.suptitle(f"Number of place fields in {self.area}")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_place_fields.pdf")
        plt.savefig(f"{self.output_root}/plot_place_fields.svg", transparent=True)
        if self.is_notebook:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # pie chart with percentages
        fig, ax = plt.subplots(figsize=(scale*3, scale*3), dpi=125)  # , gridspec_kw={"width_ratios": [1,3]}
        pfs_by_category.plot(kind="pie", y="session id", ax=ax, colors=self.categories_colors_RGB, legend=False, pctdistance=1.2, labels=[""]*5,
                             autopct=lambda pct: f'{np.round(pct, 1)}%', startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_place_fields_pie.pdf")
        plt.savefig(f"{self.output_root}/plot_place_fields_pie.svg", transparent=True)
        plt.clf()
        plt.cla()
        plt.close()

        # pie chart without percentages
        fig, ax = plt.subplots(figsize=(scale*3, scale*3), dpi=125)  # , gridspec_kw={"width_ratios": [1,3]}
        pfs_by_category.plot(kind="pie", y="session id", ax=ax, colors=self.categories_colors_RGB, legend=False, pctdistance=1.5, labels=None,
                             autopct=lambda pct: "", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_place_fields_pie_notext.pdf")
        plt.savefig(f"{self.output_root}/plot_place_fields_pie_notext.svg", transparent=True)
        plt.clf()
        plt.cla()
        plt.close()

    def plot_place_fields_by_session(self):
        scale = 1
        fig, ax = plt.subplots(figsize=(scale*5, scale*3), dpi=150, num=2, clear=True)
        pfs_by_category = self.pfs_df.groupby(["category", "session id", "category_order"]).count().sort_values(by="category_order")
        pf_counts = pfs_by_category.reset_index()[["category", "session id", "cell id"]]
        sns.boxplot(data=pf_counts, x="category", y="cell id", ax=ax, palette=self.categories_colors_RGB, showfliers=False)
        sns.swarmplot(data=pf_counts, x="category", y="cell id", ax=ax, color="black", alpha=0.5)
        plt.title(f"[{self.area}] Number of various place fields for each session")
        ax.set_ylabel("# place fields")
        ax.set_xlabel("")
        plt.savefig(f"{self.output_root}/plot_place_fields_by_session.pdf")
        if self.is_notebook:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

    def plot_place_field_proportions(self):
        scale = 0.75
        #scale = 1
        fig, ax = plt.subplots(figsize=(scale*3, scale*3), dpi=150, num=3, clear=True)
        #sns.boxplot(data=self.pf_proportions_by_category_df, ax=ax, palette=self.categories_colors_RGB,
        #            showfliers=False, width=0.7)
        #sns.swarmplot(data=self.pf_proportions_by_category_df, ax=ax, palette=self.categories_colors_RGB, alpha=0.75, size=4)
        sns.violinplot(data=self.pf_proportions_by_category_df, ax=ax, palette=self.categories_colors_RGB,
                      alpha=1.0, saturation=1, cut=0, inner_kws={"box_width": 3.5})
        # plt.title(f"[{area}] Proportion of various place field categories by session")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([-0.01, 1.01])
        ax.set_ylabel("proportion")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        #plt.suptitle(f"Proportion of PFs by session in {self.area}")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_place_field_proportions.pdf")
        plt.savefig(f"{self.output_root}/plot_place_field_proportions.svg", transparent=True)
        if self.is_notebook:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

    def plot_place_fields_criteria(self):
        fig, ax = plt.subplots()
        plt.title(f"[{self.area}] Number of place fields with a given BTSP criterion satisfied")
        nonbtsp_pfs_w_criteria_df = self.pfs_df[self.pfs_df["category"] == "non-btsp"][self.btsp_criteria].apply(pd.value_counts).transpose()
        nonbtsp_pfs_w_criteria_df.plot(kind="bar", stacked=True, color=["r", "g"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        plt.savefig(f"{self.output_root}/plot_place_fields_criteria.pdf")
        if self.is_notebook:
            plt.show()
        else:
            plt.close()

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
        plt.close()

    def plot_place_field_distro(self):
        n_bins = 75
        fig, axs = plt.subplots(2, 2, sharex=True)
        for i_corr, corridor in enumerate([14, 15]):
            counts_nonbtsp = np.zeros(n_bins)
            counts_btsp = np.zeros(n_bins)
            pfs_corr = self.pfs_df[self.pfs_df["corridor"] == corridor]
            for i_pf, pf in pfs_corr.iterrows():
                if pf["category"] == "non-btsp":
                    counts_nonbtsp[pf["lower bound"]:pf["upper bound"]] += 1
                if pf["category"] == "btsp":
                    counts_btsp[pf["lower bound"]:pf["upper bound"]] += 1
            axs[0,i_corr].stairs(counts_nonbtsp, label="non-BTSP", color="red", fill=True)
            axs[1,i_corr].stairs(counts_btsp, label="BTSP", color="purple", fill=True)
            RZ_start, RZ_end = self.reward_zones[i_corr]
            axs[0,i_corr].axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
            axs[1,i_corr].axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
        plt.xlim([0,n_bins])
        plt.savefig(f"{self.output_root}/plot_place_field_distro.pdf")
        plt.close()

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
        plt.close()

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
        scale = 0.9
        plt.figure(figsize=(scale*5, scale*3), dpi=150)
        none_satisfied = select_row(gain, False, shift, False, drift, False)
        plt.text(0.6, -0.3, f"none: {np.round(100 * none_satisfied, 1)}%")
        v = venn3_unweighted(subsets, set_labels=criteria,
                             subset_label_formatter=lambda label: f"{np.round(100 * label, 1)}%")

        norm = matplotlib.colors.Normalize(vmin=min(subsets), vmax=max(subsets), clip=True)
        if self.area == "CA1":
            cmap = cmasher.get_sub_cmap('Purples', 0.1, 0.55)
        else:
            cmap = cmasher.get_sub_cmap('Reds', 0.1, 0.55)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        patch_ids = ['100', '010', '110', '001', '101', '011', '111']
        for i_patch_id, patch_id in enumerate(patch_ids):
            patch = v.get_patch_by_id(patch_id)
            patch.set_color(mapper.to_rgba(subsets[i_patch_id]))
            patch.set(edgecolor="white", alpha=1, linewidth=2.5)

        if with_btsp:
            filename = f"{self.output_root}/plot_place_fields_criteria_venn_diagram_with_btsp"
        else:
            filename = f"{self.output_root}/plot_place_fields_criteria_venn_diagram"
        plt.savefig(f"{filename}.pdf", transparent=True)
        plt.savefig(f"{filename}.svg", transparent=True)
        if self.is_notebook:
            plt.show()
        else:
            plt.close()

    def calc_shift_gain_distribution(self, unit="cm"):
        #shift_gain_df = self.pfs_df[["category", "newly formed", "initial shift", "formation gain", "formation rate sum"]].reset_index(drop=True)
        shift_gain_df = self.pfs_df[["animal id", "session id", "cell id", "corridor", "category", "newly formed", "initial shift", "formation gain", "formation rate sum"]].reset_index(drop=True)
        shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"].notna()) & (shift_gain_df["formation gain"].notna())]
        shift_gain_df["log10(formation gain)"] = np.log10(shift_gain_df["formation gain"])

        if unit == "cm":
            shift_gain_df["initial shift"] = self.bin_length * shift_gain_df["initial shift"]
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
        plt.close()

    def shift_gain_pca_sklearn(self):
        def take_same_sized_subsample(df):
            n = min(df[df["newly formed"] == True].shape[0], df[df["newly formed"] == False].shape[0])
            ss1 = df[df["newly formed"] == True].sample(n=n)
            ss2 = df[df["newly formed"] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        sg_df = self.shift_gain_df
        ss_cc = take_same_sized_subsample(sg_df)
        df_pca = ss_cc[["initial shift", "log10(formation gain)"]]
        df_pca_norm = (df_pca - df_pca.mean())/df_pca.std()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(df_pca_norm)
        print(f"PCA components: {pca.components_}")
        print(f"PCA explained variance: {pca.explained_variance_}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    def shift_gain_pca(self):
        def take_same_sized_subsample(df):
            n = min(df[df["newly formed"] == True].shape[0], df[df["newly formed"] == False].shape[0])
            ss1 = df[df["newly formed"] == True].sample(n=n)
            ss2 = df[df["newly formed"] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        def pca(df):
            # centering
            df["initial shift"] = df["initial shift"] - df["initial shift"].mean()
            df["log10(formation gain)"] = df["log10(formation gain)"] - df["log10(formation gain)"].mean()

            # standardization
            df["initial shift"] = df["initial shift"] / df["initial shift"].std()
            df["log10(formation gain)"] = df["log10(formation gain)"] / df["log10(formation gain)"].std()

            # eigendecomposition of covariance matrix
            df_pca = df[["initial shift", "log10(formation gain)"]]
            covariance_matrix = df_pca.cov()
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            # sort eigenvalues and eigenvectors (from largest eigenvalue to smallest)
            idx = eigenvalues.argsort()[::-1]  # indices of eigenvalues by descending order
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:,idx]
            return eigenvalues, eigenvectors

        def factor_analysis(df):
            df_fa = df[["initial shift", "log10(formation gain)"]]
            from sklearn.decomposition import FactorAnalysis
            fa = FactorAnalysis(n_components=2)
            fa.fit(df_fa)
            return fa

        sg_df = self.shift_gain_df
        ss_cc = take_same_sized_subsample(sg_df)
        #ss_cc = ss_cc[ss_cc["newly formed"]==True]
        #eigenvalues_all, eigenvectors_all = pca(ss_cc)
        g = sns.jointplot(data=ss_cc, x="initial shift", y="log10(formation gain)",
                          hue="newly formed", alpha=0.2, s=10, marginal_kws={"common_norm": False})
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")
        #plt.quiver([0,0], [0,0], np.multiply(eigenvalues_all, eigenvectors_all[0,:]), np.multiply(eigenvalues_all, eigenvectors_all[1,:]), units="xy", width=0.02)
        #plt.arrow(0,0, eigenvalues_all[0]*eigenvectors_all[0,0], eigenvalues_all[0]*eigenvectors_all[1,0], head_starts_at_zero=True, width=0.05)
        #plt.arrow(0,0, eigenvalues_all[1]*eigenvectors_all[0,1], eigenvalues_all[1]*eigenvectors_all[1,1], head_starts_at_zero=True, width=0.05)
        #plt.axline((0,0), (eigenvectors_all[:,0]), c="red", label=np.round(eigenvalues_all[0],2))
        #plt.axline((0,0), (eigenvectors_all[:,1]), c="k", label=np.round(eigenvalues_all[1],2))
        #plt.ylim([-4, 4])
        #plt.xlim([-4, 4])
        #plt.legend()

        ss_cc_nf = ss_cc[ss_cc["newly formed"]==True]
        #eigenvalues_nf, eigenvectors_nf = pca(ss_cc_nf)
        fa_newly = factor_analysis(ss_cc_nf)
        g = sns.jointplot(data=ss_cc_nf, x="initial shift", y="log10(formation gain)",
                          color="orange", alpha=0.2, s=10, marginal_kws={"common_norm": False})
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")
        #plt.axline((0,0), (eigenvectors_nf[:,0]), c="red", label=np.round(eigenvalues_nf[0],2))
        #plt.axline((0,0), (eigenvectors_nf[:,1]), c="k", label=np.round(eigenvalues_nf[1],2))
        plt.ylim([-4, 4])
        plt.xlim([-4, 4])
        plt.legend()

        ss_cc_es = ss_cc[ss_cc["newly formed"]==False]
        eigenvalues_es, eigenvectors_es = pca(ss_cc_es)
        g = sns.jointplot(data=ss_cc_es, x="initial shift", y="log10(formation gain)", color="blue",
                          alpha=0.2, s=10, marginal_kws={"common_norm": False})
        plt.axvline(x=0, c="k", linestyle="--")
        plt.axhline(y=0, c="k", linestyle="--")
        plt.axline((0,0), (eigenvectors_es[:,0]), c="red", label=np.round(eigenvalues_es[0],2))
        plt.axline((0,0), (eigenvectors_es[:,1]), c="k", label=np.round(eigenvalues_es[1],2))
        plt.ylim([-4, 4])
        plt.xlim([-4, 4])
        plt.legend()
        if self.is_notebook:
            plt.show()
        else:
            plt.close()

    def run_tests(self, df, params=None, export_results=True, suffix=""):

        def stars(pvalue):
            if pvalue < 0.001:
                return "***"
            elif pvalue < 0.01:
                return "**"
            elif pvalue < 0.05:
                return "*"
            else:
                return ""

        sh = "shuffledLaps" if "shuffledLaps" in self.extra_info else ""
        test_results = f"{self.area}\n\n{sh}\n\n"
        test_dict = {
            "area": [],
            "population": [],
            "feature": [],
            "test": [],
            "statistic": [],
            "p-value": [],
            "log p-value": []
        }
        for param in params:
            test_results += f"{param}\n"
            df_newlyF = df[df["newly formed"] == True]
            df_establ = df[df["newly formed"] == False]

            # t-test: do the 2 samples have equal means?
            test = scipy.stats.ttest_ind(df_newlyF[param].values, df_establ[param].values)
            test_results += f"    p={test.pvalue:.3} (t-test) {stars(test.pvalue)}\n"
            test_dict["area"].append(self.area)
            test_dict["population"].append("reliables")
            test_dict["feature"].append(param)
            test_dict["test"].append("t-test")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            # mann-whitney u: do the 2 samples come from same distribution?
            test = scipy.stats.mannwhitneyu(df_newlyF[param].values, df_establ[param].values)
            test_results += f"    p={test.pvalue:.3} (MW-U) {stars(test.pvalue)}\n"
            test_dict["area"].append(self.area)
            test_dict["population"].append("reliables")
            test_dict["feature"].append(param)
            test_dict["test"].append("mann-whitney u")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            # kolmogorov-smirnov: do the 2 samples come from the same distribution?
            test = scipy.stats.kstest(df_newlyF[param].values, cdf=df_establ[param].values)
            test_results += f"    p={test.pvalue:.3} (KS) {stars(test.pvalue)}\n"
            test_dict["area"].append(self.area)
            test_dict["population"].append("reliables")
            test_dict["feature"].append(param)
            test_dict["test"].append("kolmogorov-smirnov")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            # wilcoxon
            test = scipy.stats.wilcoxon(df_newlyF[param].values)
            test_results += f"    p={test.pvalue:.3} (WX-NF) {stars(test.pvalue)}\n"
            test_dict["area"].append(self.area)
            test_dict["population"].append("newly formed")
            test_dict["feature"].append(param)
            test_dict["test"].append("wilcoxon")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            test = scipy.stats.wilcoxon(df_establ[param].values)
            test_results += f"    p={test.pvalue:.3} (WX-ES) {stars(test.pvalue)}\n"
            test_dict["area"].append(self.area)
            test_dict["population"].append("established")
            test_dict["feature"].append(param)
            test_dict["test"].append("wilcoxon")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            # t-test (1 sample)
            test = scipy.stats.ttest_1samp(df_newlyF[param].values, popmean=0)
            test_results += (f"    p={test.pvalue:.3} (TT1S-NF) {stars(test.pvalue)}\n")
            test_dict["area"].append(self.area)
            test_dict["population"].append("newly formed")
            test_dict["feature"].append(param)
            test_dict["test"].append("1-sample t-test")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            test = scipy.stats.ttest_1samp(df_establ[param].values, popmean=0)
            test_results += f"    p={test.pvalue:.3} (TT1S_ES) {stars(test.pvalue)}\n"
            test_dict["area"].append(self.area)
            test_dict["population"].append("established")
            test_dict["feature"].append(param)
            test_dict["test"].append("1-sample t-test")
            test_dict["statistic"].append(test.statistic)
            test_dict["p-value"].append(test.pvalue)
            test_dict["log p-value"].append(np.log10(test.pvalue))

            # shapiro-wilk: does the 1 sample have normal distribution?
            #_, pvalue = scipy.stats.shapiro(df_newlyF[param].values)
            #test_results += f"    p={np.round(pvalue, 2):.2f} (SW; newly formed) {stars(pvalue)}\n"
            #test_results += f"    p={test.pvalue} (t-test) {stars(test.pvalue)}\n"
            #_, pvalue = scipy.stats.shapiro(df_establ[param].values)
            #test_results += f"    p={np.round(pvalue, 2):.2f} (SW; established) {stars(pvalue)}\n"
            #test_results += f"    p={test.pvalue} (t-test) {stars(test.pvalue)}\n"

            # kolmogorov-smirnov: does the 1 sample have the same distribution as the given distribution?
            #test = scipy.stats.kstest(df_newlyF[param].values, cdf=scipy.stats.norm.cdf)
            #test_results += f"    p={np.round(test.pvalue, 2):.2f} (KS; newly formed) {stars(test.pvalue)}\n"
            #test_results += f"    p={test.pvalue} (t-test) {stars(test.pvalue)}\n"
            #test = scipy.stats.kstest(df_establ[param].values, cdf=scipy.stats.norm.cdf)
            #test_results += f"    p={np.round(test.pvalue, 2):.2f} (KS; established) {stars(test.pvalue)}\n"
            #test_results += f"    p={test.pvalue} (t-test) {stars(test.pvalue)}\n"

            test_results += f"\n"
        if export_results:  # TODO: make this an excel using dataframe
            with open(f"{self.output_root}/tests{suffix}.txt", "w") as test_results_file:
                test_results_file.write(test_results)
        self.tests_df = pd.DataFrame.from_dict(test_dict)

    def plot_shift_gain_distribution(self, unit="cm", without_transient=False):
        def take_same_sized_subsample(df):
            if self.area == "CA3":
                return df
            n = min(df[df["newly formed"] == True].shape[0], df[df["newly formed"] == False].shape[0])
            ss1 = df[df["newly formed"] == True].sample(n=n)
            ss2 = df[df["newly formed"] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        def convert_to_z_score(df):
            # centering
            df["initial shift"] = df["initial shift"] - df["initial shift"].mean()
            df["log10(formation gain)"] = df["log10(formation gain)"] - df["log10(formation gain)"].mean()
            # standardization
            df["initial shift"] = df["initial shift"] / df["initial shift"].std()
            df["log10(formation gain)"] = df["log10(formation gain)"] / df["log10(formation gain)"].std()
            return df

        sg_df = self.shift_gain_df
        suffix = ""
        if without_transient:
            sg_df = sg_df[sg_df["category"] != "transient"]
            suffix = "_withoutTransient"

        ss_cc = take_same_sized_subsample(sg_df)
        ss_cc.to_excel(f"{self.output_root}/data_shift_gain.xlsx")
        self.pfs_df.to_excel(f"{self.output_root}/data_pfs_df.xlsx")
        #ss_cc = convert_to_z_score(ss_cc)

        palette = ["#00B0F0", "#F5B800"]
        if self.area == "CA1":
            alpha=0.4
        else:
            alpha=0.8

        is_legend = False
        fig, ax = plt.subplots()
        g = sns.jointplot(data=ss_cc, x="initial shift", y="log10(formation gain)", palette=sns.color_palette(palette, 2),
                      hue="newly formed", alpha=alpha, s=30, marginal_kws={"common_norm": False}, height=3.5, ratio=3, legend=is_legend)

        # change legend
        if is_legend:
            g.ax_joint.legend_.set_title("")
            labels = ["Established PFs", "Newly formed PFs"]
            for i_label, label in enumerate(labels):
                # remove opacity of dots in legend
                g.ax_joint.legend_.legend_handles[i_label].set_alpha(1)
                # set custom legend labels
                text_obj = g.ax_joint.legend_.texts[i_label]
                text_obj.set_text(label)
            g.ax_joint.legend_.set_bbox_to_anchor((0.5, 0.85))

        # change labels
        g.set_axis_labels("initial shift [cm]", r"$log_{10}(formation\ gain)$")
        #.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #plt.tight_layout()
        #sg_df_z = convert_to_z_score(self.shift_gain_df)
        plt.ylim([-1, 1])
        xlims = [-16, 16]
        if unit == "cm":
            xlims = [-16 * self.bin_length, 16 * self.bin_length]
            plt.xticks(np.arange(-30, 40, 10))
        plt.xlim(xlims)

        #plt.ylim([-4, 4])
        #plt.xlim([-4, 4])
        plt.axvline(x=0, c="k", linestyle="--", zorder=1)
        plt.axhline(y=0, c="k", linestyle="--", zorder=1)

        #mean_newly_shift = ss_cc[ss_cc["newly formed"] == True]["initial shift"].mean()
        #mean_newly_gain = ss_cc[ss_cc["newly formed"] == True]["log10(formation gain)"].mean()
        #mean_estab_shift = ss_cc[ss_cc["newly formed"] == False]["initial shift"].mean()
        #mean_estab_gain = ss_cc[ss_cc["newly formed"] == False]["log10(formation gain)"].mean()
        #means_palette = ["#00B0F0", "#F5B800"]
        #plt.scatter(mean_estab_shift, mean_estab_gain, marker="X", c=palette[0], edgecolors="white", s=100)
        #plt.scatter(mean_newly_shift, mean_newly_gain, marker="X", c=palette[1], edgecolors="white", s=100)

        if self.is_notebook:
            scale = 0.25
        else:
            scale = 1
        height = 3.5
        annotate = False

        self.run_tests(sg_df, params=["initial shift", "log10(formation gain)"], suffix=suffix)
        g.figure.set_size_inches((height, height))

        if annotate:
            g.figure.set_size_inches((scale*1.81*height, scale*height))
            plt.subplots_adjust(right=0.65)

        #else:
        #    #g.fig.set_size_inches((height, height))
        #    run_tests_and_plot_results(sg_df, g=g, params=["initial shift", "log10(formation gain)"])
        #    plt.tight_layout()
        #title_suffix = "without transient PFs" if without_transient else "transient PFs included"
        #g.fig.suptitle(f"Distribution of PFs by shift and gain ({title_suffix})")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions{suffix}.pdf")
        plt.savefig(f"{self.output_root}/plot_shift_gain_distributions{suffix}.svg")
        plt.clf()
        plt.cla()
        plt.close()

        ###########################
        ####### VIOLINPLOTS #######
        ###########################

        if self.is_notebook:
            scale = 0.5
        else:
            scale = 0.9
            sns.set_context("poster")
        fig, axs = plt.subplots(1,2, figsize=(scale*9,scale*9), num=5, clear=True)

        q_lo = ss_cc["initial shift"].quantile(0.01)
        q_hi = ss_cc["initial shift"].quantile(0.99)
        ss_cc_filt = ss_cc[(ss_cc["initial shift"] < q_hi) & (ss_cc["initial shift"] > q_lo)]
        df_violin = ss_cc_filt[["newly formed", "initial shift"]].reset_index(drop=True).melt(id_vars="newly formed")
        g = sns.violinplot(data=df_violin, x="variable", y="value", hue="newly formed", split=True, fill=False, ax=axs[0],
                       palette=palette, saturation=1, legend=False, gap=0.0, inner=None, linewidth=5)  # inner_kws={"color": "0.5"}
        sns.boxplot(data=df_violin, x="variable", y="value", hue="newly formed", ax=axs[0], width=0.3, palette=palette,
                    saturation=1, linewidth=3, linecolor="k", legend=False, fliersize=0)
        median_newly_shift = np.round(ss_cc_filt[ss_cc_filt["newly formed"] == True]["initial shift"].median(), 1)
        median_estab_shift = np.round(ss_cc_filt[ss_cc_filt["newly formed"] == False]["initial shift"].median(), 1)
        if self.area == "CA1":
            axs[0].annotate(median_estab_shift, (-0.38, median_estab_shift-1.8), color="k")
            axs[0].annotate(median_newly_shift, (0.2, median_newly_shift-0.75), color="k")
        else:
            axs[0].annotate(median_estab_shift, (-0.38, median_estab_shift-1.8), color="k")
            axs[0].annotate(median_newly_shift, (0.2, median_newly_shift-0.85), color="k")

        axs[0].tick_params(axis="x", which='both', length=0, pad=15)
        axs[0].set_yticks(np.arange(-25, 30, 5))
        #axs[0].set_yticklabels(np.arange(-25, 30, 5))
        axs[0].axhline(y=0, c="k", linestyle="--", zorder=0)
        axs[0].set_ylim([-25, 25])
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].set_xlabel("")
        axs[0].set_ylabel("")
        axs[0].set_xticklabels([r"initial shift [cm]"])

        q_lo = ss_cc["log10(formation gain)"].quantile(0.001)
        q_hi = ss_cc["log10(formation gain)"].quantile(0.999)
        ss_cc_filt = ss_cc[(ss_cc["log10(formation gain)"] < q_hi) & (ss_cc["log10(formation gain)"] > q_lo)]
        df_violin = ss_cc_filt[["newly formed", "log10(formation gain)"]].reset_index(drop=True).melt(id_vars="newly formed")
        g = sns.violinplot(data=df_violin, x="variable", y="value", hue="newly formed", split=True, fill=False, ax=axs[1],
                       palette=palette, saturation=1, legend=False, gap=0.0, inner=None, linewidth=5)  # inner_kws={"color": "0.5"}
        sns.boxplot(data=df_violin, x="variable", y="value", hue="newly formed", ax=axs[1], width=0.3, palette=palette,
                    saturation=1, linewidth=3, linecolor="k", legend=False, fliersize=0)
        median_newly_gain = np.round(ss_cc_filt[ss_cc_filt["newly formed"] == True]["log10(formation gain)"].median(), 1)
        median_estab_gain = np.round(ss_cc_filt[ss_cc_filt["newly formed"] == False]["log10(formation gain)"].median(), 2)
        if self.area == "CA1":
            axs[1].annotate(median_estab_gain, (-0.43, median_estab_gain+0.02), color="k")
            axs[1].annotate(median_newly_gain, (0.2, median_newly_gain-0.02), color="k")
        else:
            axs[1].annotate(median_estab_gain, (-0.46, median_estab_gain-0.05), color="k")
            axs[1].annotate(median_newly_gain, (0.2, median_newly_gain-0.03), color="k")

        # change legend
        #g.legend_.set_title("")
        #labels = ['Established PFs', 'Newly formed PFs']
        #for i_label, label in enumerate(labels):
        #    # set custom legend labels
        #    text_obj = g.legend_.texts[i_label]
        #    text_obj.set_text(label)
        #g.legend_.set_bbox_to_anchor((1, 0.95))

        axs[1].tick_params(axis="x", which='both', length=0, pad=15)
        axs[1].set_xticklabels([r"$log_{10}(formation\ gain)$"])
        axs[1].axhline(y=0, c="k", linestyle="--", zorder=0)
        axs[1].set_ylim([-1, 1])
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].set_xlabel("")
        axs[1].set_ylabel("")
        #plt.suptitle(f"Distribution of PFs by shift and gain\n({title_suffix})")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_shift_gain_violinplots{suffix}.pdf")
        plt.savefig(f"{self.output_root}/plot_shift_gain_violinplots{suffix}.svg", transparent=True)
        if self.is_notebook:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()




        ###########################
        ##### REGRESSION PLOT #####
        ###########################

        #fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(12,6))
        #df_newlyF = ss_cc[ss_cc["newly formed"] == True]
        #df_newlyF = df_newlyF[(df_newlyF["initial shift"] > -15) &
        #                      (df_newlyF["initial shift"] <= 15) &
        #                      (df_newlyF["log10(formation gain)"] > -1) &
        #                      (df_newlyF["log10(formation gain)"] <= 1)]
        #df_newlyF_z = convert_to_z_score(df_newlyF)
        #sns.regplot(df_newlyF_z, x="initial shift", y="log10(formation gain)", color="orange", scatter_kws={'alpha':0.15},
        #            line_kws=dict(color="r"), ax=axs[0])
        #df_establ = ss_cc[ss_cc["newly formed"] == False]
        #df_establ = df_establ[(df_establ["initial shift"] > -15) &
        #                      (df_establ["initial shift"] <= 15) &
        #                      (df_establ["log10(formation gain)"] > -1) &
        #                      (df_establ["log10(formation gain)"] <= 1)]
        #df_establ_z = convert_to_z_score(df_establ)
        #sns.regplot(df_establ_z, x="initial shift", y="log10(formation gain)", color="blue", scatter_kws={'alpha':0.15},
        #            line_kws=dict(color="purple"), ax=axs[1])
        ##plt.ylim([-1, 1])
        ##plt.xlim([-15, 15])
        #plt.ylim([-4, 4])
        #plt.xlim([-4, 4])
        #axs[0].axvline(x=0, c="k", linestyle="--")
        #axs[0].axhline(y=0, c="k", linestyle="--")
        #axs[1].axvline(x=0, c="k", linestyle="--")
        #axs[1].axhline(y=0, c="k", linestyle="--")
        #plt.tight_layout()
        #plt.savefig(f"{self.output_root}/plot_regression_zscore.pdf")
        #plt.close()

        #### plot gain-FL distro
        #s=0.9
        #fig, ax1 = plt.subplots(figsize=(s*4.5, s*3.5), dpi=150)
        #ax2 = ax1.twinx()
        ##pfs_df_filt = self.pfs_df[self.pfs_df["end lap"] - self.pfs_df["formation lap"] > 15]
        #pfs_df_filt = self.pfs_df
        #FL_gain_df = pfs_df_filt[["formation lap", "formation gain"]].reset_index(drop=True)
        #FL_gain_df = FL_gain_df[(FL_gain_df["formation lap"].notna()) & (FL_gain_df["formation gain"].notna())]
        #FL_gain_df["log10(formation gain)"] = np.log10(FL_gain_df["formation gain"])
        #sns.histplot(FL_gain_df, x="formation lap", ax=ax2, binwidth=1, color="orange", alpha=0.25)
        #sns.pointplot(FL_gain_df, x="formation lap", y="log10(formation gain)", ax=ax1)
        #plt.xlim([0,15])
        #ax1.set_ylim([-0.5, 0.5])
        #ax2.set_ylim([0, 7500])
        #plt.tight_layout()
        #plt.savefig(f"{self.output_root}/plot_gain_FL_distribution.pdf")
        #plt.savefig(f"{self.output_root}/plot_gain_FL_distribution_nofilt.png")
        #plt.close()

        #### plot FL-lifespan distro
        #fig, ax1 = plt.subplots()
        #self.pfs_df["lifespan"] = self.pfs_df["end lap"]# - self.pfs_df["formation lap"]
        #pfs_df_filt = self.pfs_df[self.pfs_df["formation lap"] > -1]
        #FL_LS_df = pfs_df_filt[["formation lap", "end lap"]].reset_index(drop=True)
        #FL_LS_df = FL_LS_df[(FL_LS_df["formation lap"].notna()) & (FL_LS_df["end lap"].notna())]
        #sns.violinplot(FL_LS_df, x="formation lap", y="end lap", ax=ax1, inner=None, native_scale=True)
        #plt.xlim([0,10])
        #plt.savefig(f"{self.output_root}/plot_FL_EL_distribution.pdf")
        #plt.close()

    def plot_slopes_like_Madar(self):
        def take_same_sized_subsample(df, param):
            n = min(df[df[param] == True].shape[0], df[df[param] == False].shape[0])
            ss1 = df[df[param] == True].sample(n=n)
            ss2 = df[df[param] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        pfs_df_r_slope = self.pfs_df[(self.pfs_df["spearman r"].notna()) & (self.pfs_df["linear fit m"].notna())]
        pfs_df_r_slope = pfs_df_r_slope[(pfs_df_r_slope["end lap"] - pfs_df_r_slope["formation lap"] > 20)]
        pfs_df_r_slope["log10(formation gain)"] = np.log10(pfs_df_r_slope["formation gain"])
        pfs_df_r_slope = pfs_df_r_slope[["spearman r", "spearman p", "linear fit m", "has backwards shift", "has high gain", "log10(formation gain)"]]
        pfs_df_r_slope["r2"] = pfs_df_r_slope["spearman r"] ** 2
        #pfs_df_r_slope = pfs_df_r_slope[pfs_df_r_slope["spearman p"] <= 0.05]
        #ax1 = pfs_df_r_slope[pfs_df_r_slope["initial shift"] >= 0].plot.scatter(x="linear fit m", y="r2", c="blue", alpha=0.1)
        #ax2 = pfs_df_r_slope[pfs_df_r_slope["initial shift"] < 0].plot.scatter(x="linear fit m", y="r2", c="red", ax=ax1, alpha=0.1)
        #pfs_df_r_slope.plot.scatter(x="linear fit m", y="r2", c="k", alpha=0.1)
        ss_cc = take_same_sized_subsample(pfs_df_r_slope, "has backwards shift")
        sns.jointplot(data=ss_cc, x="linear fit m", y="r2", hue="has backwards shift", palette="Set1", alpha=0.2)
        plt.xlim([-1, 1])
        plt.ylim([0, 1])
        plt.axvline(0, c="k", linestyle="--")
        plt.savefig(f"{self.output_root}/plot_slopes_like_Madar.pdf")
        plt.close()

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
        plt.close()

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
        plt.close()

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

    def plot_shift_scores(self):
        n_laps = 15

        def make_shift_score_matrix(pfs_df):
            shift_score_matrix = np.zeros((pfs_df.shape[0], n_laps))
            shift_score_matrix[:] = np.nan
            shift_scores_all = pfs_df[["shift scores"]]
            for i_row in range(len(shift_scores_all)):
                shift_scores = shift_scores_all.iloc[i_row][0][0]  # first 0: bc. part of Series, second 0: bc list of shift scores is itself an element of a list of size 1
                if len(shift_scores) > n_laps:
                    shift_scores = shift_scores[:n_laps]
                shift_score_matrix[i_row, :len(shift_scores)] = shift_scores
            shift_score_means = np.nanmean(shift_score_matrix, axis=0)
            shift_score_stds = np.nanstd(shift_score_matrix, axis=0)
            return shift_score_matrix, shift_score_means, shift_score_stds

        def make_plot(ax, means, stds, label):
            ax.plot(means, label=label)
            ax.fill_between(x = np.arange(0,n_laps), y1 = means-stds, y2 = means+stds, alpha=0.1)

        establ_df = self.pfs_df[(self.pfs_df["newly formed"] == False) & (self.pfs_df["category"] != "unreliable")]
        newlyf_df = self.pfs_df[(self.pfs_df["newly formed"] == True)]
        btsp_df = self.pfs_df[self.pfs_df["category"] == "btsp"]
        nonbtsp_df = self.pfs_df[self.pfs_df["category"] == "non-btsp"]

        ssm_establ, mean_ss_establ, stds_ss_establ = make_shift_score_matrix(establ_df)
        ssm_newlyf, mean_ss_newlyf, stds_ss_newlyf = make_shift_score_matrix(newlyf_df)
        #ssm_btsp, mean_ss_btsp, stds_ss_btsp = make_shift_score_matrix(btsp_df)
        #ssm_nonbtsp, mean_ss_nonbtsp, stds_ss_nonbtsp = make_shift_score_matrix(nonbtsp_df)

        fig, ax = plt.subplots()
        make_plot(ax, mean_ss_establ, stds_ss_establ, label="established")
        make_plot(ax, mean_ss_newlyf, stds_ss_newlyf, label="newly formed")
        #make_plot(ax, mean_ss_btsp, stds_ss_btsp, label="btsp")
        #make_plot(ax, mean_ss_nonbtsp, stds_ss_nonbtsp, label="non-btsp")
        plt.xlim([0,n_laps-1])
        plt.axhline(0, c="k", linestyle="--")
        plt.legend()
        plt.savefig(f"{self.output_root}/plot_shift_scores.pdf")
        plt.close()

    def dF_F_shifts(self):
        pfs_df = self.pfs_df[self.pfs_df["category"] != "unreliable"]
        pfs_df = sklearn.utils.shuffle(pfs_df)
        lapbylap_df = None
        idx = 0
        for i_pf, pf in pfs_df.iterrows():
            print(f"{idx} / {pfs_df.shape[0]}")
            cat = pf["category"]
            shift = pf["initial shift"]
            gain = pf["formation gain"]
            dF_F_maxima = pf["dF/F maxima"][0]
            dCOM_lap_by_lap = pf["dCOM (lap-by-lap)"][0]
            for i_lap in range(len(dCOM_lap_by_lap)):
                dF_F = dF_F_maxima[i_lap]
                dCOM = dCOM_lap_by_lap[i_lap]
                if np.isnan(dCOM):
                    continue
                lapbylap_df_dict = {
                    "category": cat,
                    "initial shift": shift,
                    "formation gain": gain,
                    "newly formed": cat in ["transient", "non-btsp", "btsp"],
                    "dF/F maximum": dF_F,
                    "log(dF/F maximum)": np.log10(dF_F),
                    "dCOM": dCOM
                }
                lapbylap_df_lap = pd.DataFrame.from_dict([lapbylap_df_dict])
                lapbylap_df = grow_df(lapbylap_df, lapbylap_df_lap)
            idx += 1
            if idx == 1000:
                break
        sns.jointplot(lapbylap_df, y="log(dF/F maximum)", x="dCOM", alpha=0.15, hue="newly formed", marginal_kws={"common_norm": False})
        plt.savefig(f"{self.output_root}/plot_dF_F_shifts.pdf")
        plt.close()
        # ötletek: csak aktív lapek, mindenkire végigfuttatni, shift score diffek miben különböznek?, LEGFONTOSABB: TAU0.8-cal megnézni!!

    def plot_shift_gain_dependence(self):
        def take_same_sized_subsample(df, param):
            n = min(df[df[param] == True].shape[0], df[df[param] == False].shape[0])
            ss1 = df[df[param] == True].sample(n=n)
            ss2 = df[df[param] == False].sample(n=n)
            ss_cc = sklearn.utils.shuffle(pd.concat((ss1, ss2)))
            return ss_cc

        pfs_df = self.pfs_df
        pfs_df["log10(formation gain)"] = np.log10(pfs_df["formation gain"])
        pfs_df = pfs_df[~pfs_df["log10(formation gain)"].isna()]
        pfs_df = take_same_sized_subsample(pfs_df, "newly formed")
        pfs_df["gain decile"] = np.nan
        for i_pct, pct in enumerate(np.arange(0, 1, 0.1)):
            pct_lb = pfs_df["log10(formation gain)"].quantile(pct)
            pct_ub = pfs_df["log10(formation gain)"].quantile(pct+0.1)
            pfs_df.loc[(pfs_df["log10(formation gain)"] >= pct_lb) & (pfs_df["log10(formation gain)"] < pct_ub), "gain decile"] = i_pct
        sns.pointplot(pfs_df, x="gain decile", y="initial shift", hue="newly formed", errorbar=("ci",99))
        plt.axhline(0, c="r")
        plt.savefig(f"{self.output_root}/plot_SG_dependence_point.pdf")
        plt.close()

        plt.figure()
        sns.boxplot(pfs_df, x="gain decile", y="initial shift", hue="newly formed", fliersize=0)
        plt.ylim([-10, 10])
        plt.axhline(0, c="r")
        plt.savefig(f"{self.output_root}/plot_SG_dependence_box.pdf")
        plt.close()

    def plot_ratemaps(self):
        ratemaps_folder = f"{self.output_root}/ratemaps"
        makedir_if_needed(ratemaps_folder)

        selection = {
            "CA1": [
                # animal, session, cellid, corridor
                ["KS028", "KS028_103121", 48, 15],
                ["KS028", "KS028_110121", 19, 15],
                ["KS028", "KS028_103121", 387, 14],
                ["KS028", "KS028_103121", 123, 14],
                ["KS028", "KS028_103121", 142, 15],
            ],
            "CA3": [
                ["srb231", "srb231_220808", 16, 14],
                ["srb231", "srb231_220808", 26, 14],
                ["srb231", "srb231_220808", 33, 14],
                ["srb231", "srb231_220809_002", 53, 15],
                ["srb231", "srb231_220808", 1, 14],
            ]
        }
        #for animal, session, cellid, corridor in selection[self.area]:
        sessions = np.unique(self.pfs_df[["session id"]].values[:,0])
        corridors = [14, 15]
        for session in tqdm.tqdm(sessions):
            #if session != "KS029_110621":
            #    continue
            tcl_path = f"{self.data_path}/tuned_cells/{self.area}{self.extra_info}/tuned_cells_{session}.pickle"
            animal, _, _ = session.partition("_")
            p95_path = f"D:\\{self.area}\\data\\{animal}_imaging\\{session}\\analysed_data\\p95.npy"
            p95 = np.load(p95_path)  # (corridor x spacebin x cellid)
            with open(tcl_path, "rb") as tcl_file:
                tcl = pickle.load(tcl_file)
            for i_cell in tqdm.tqdm(range(len(tcl))):
                cell = tcl[i_cell]
                #print(f"cell {i_cell} out of {len(tcl)} in session {session}")
                cellid = cell.cellid
                corridor = cell.corridor
                corridor_idx = corridors.index(corridor)
                n_laps = cell.rate_matrix.shape[1]
                if self.history_dependent:
                    history = cell.history
                else:
                    history = "ALL"

                scale = 4.2
                fig, axs = plt.subplots(2,1,figsize=(scale*2.4, scale*3.5), sharex=True, height_ratios=[3,2])
                im = axs[0].imshow(np.transpose(cell.rate_matrix), aspect='auto', origin='lower', cmap='binary', interpolation="none")
                axs[1].plot(np.mean(cell.rate_matrix,axis=1), marker="o")
                p95_cell = p95[corridor_idx,:,cellid]
                axs[1].plot(p95_cell, color="red")

                pfs = self.pfs_df[(self.pfs_df["session id"] == session) &
                                  (self.pfs_df["cell id"] == cellid) &
                                  (self.pfs_df["corridor"] == corridor) &
                                  (self.pfs_df["history"] == history)]
                if pfs.empty:
                    plt.close()
                    continue

                inset_data = []
                for _, pf in pfs.iterrows():
                    category = pf["category"]
                    lb, ub = pf["lower bound"] - 0.5, pf["upper bound"] - 0.5  # correcting by half bin width so it plots correctly on imshow
                    fl, el = pf["formation lap"] / n_laps, (pf["end lap"] + 1) / n_laps

                    if category == "unreliable":
                        axs[0].axvspan(lb, ub, color="black", alpha=0.3, linewidth=0)
                    else:
                        axs[0].axvspan(lb, ub, ymin=fl, ymax=el, color=CATEGORIES[category].color, alpha=0.3, linewidth=0)

                    # btsp inset
                    #if category == "btsp":
                    #    # formation lap
                    #    #frames_pf_mask = (cell.frames_pos_bins[pf["formation lap"]] >= pf["lower bound"]) & (cell.frames_pos_bins[pf["formation lap"]] <= pf["upper bound"]+1)
                    #    frames_pf_dF_F = cell.frames_dF_F[pf["formation lap"]]#[frames_pf_mask]
                    #    inset_data.append(frames_pf_dF_F)

                    #    # first lap after formation lap
                    #    #frames_pf_mask = (cell.frames_pos_bins[pf["formation lap"]+1] >= pf["lower bound"]) & (cell.frames_pos_bins[pf["formation lap"]+1] <= pf["upper bound"]+1)
                    #    frames_pf_dF_F = cell.frames_dF_F[pf["formation lap"]+1]#[frames_pf_mask]
                    #    inset_data.append(frames_pf_dF_F)

                plt.ylabel("laps")
                plt.xlabel("spatial position")
                plt.colorbar(im, orientation='horizontal', ax=axs[0])
                plt.tight_layout()

                categories = np.unique(pfs["category"].values)
                for category in categories:
                    category_folder = f"{ratemaps_folder}/{category}"
                    makedir_if_needed(category_folder)

                    plt.savefig(f"{category_folder}/{session}_Cell{cellid}_Corr{corridor}_{history}.pdf")
                    #plt.savefig(f"{category_folder}/{session}_Cell{cellid}_Corr{corridor}.svg", transparent=True)
                    plt.close()

                    #if category == "btsp":
                    #    fig, ax = plt.subplots(figsize=(2,1))
                    #    offset = 1
                    #    plt.plot(inset_data[1]+offset, color="k", linewidth=1.3)
                    #    plt.plot(inset_data[0], color="r", linewidth=1.3)
                    #    scalebar_x = max(len(inset_data[0]), len(inset_data[1])) + 10
                    #    plt.plot([scalebar_x, scalebar_x+30], [0, 0], color="k")
                    #    plt.plot([scalebar_x, scalebar_x], [0, 1], color="k")
                    #    ax.set_axis_off()
                    #    #plt.savefig(f"{category_folder}/{session}_Cell{cellid}_Corr{corridor}_INSET.pdf")
                    #    #plt.savefig(f"{category_folder}/{session}_Cell{cellid}_Corr{corridor}_INSET.svg", transparent=True)
                    #    plt.close()

    def calc_place_field_quality(self):
        for i_pf, pf in self.pfs_df.iterrows():
            if i_pf % 100 == 0:
                print(f"{i_pf} / {len(self.pfs_df)}")
            if self.history_dependent:
                tc = self.tc_df[(self.tc_df["sessionID"] == pf["session id"]) &
                                (self.tc_df["cellid"] == pf["cell id"]) &
                                (self.tc_df["corridor"] == pf["corridor"]) &
                                (self.tc_df["history"] == pf["history"])]["tc"].iloc[0]
            else:
                tc = self.tc_df[(self.tc_df["sessionID"] == pf["session id"]) &
                                (self.tc_df["cellid"] == pf["cell id"]) &
                                (self.tc_df["corridor"] == pf["corridor"])]["tc"].iloc[0]

            #rate_matrix_pf = tc.rate_matrix[pf["lower bound"]:pf["upper bound"], pf["formation lap"]:pf["end lap"]]
            lb, ub, fl, el = pf["lower bound"], pf["upper bound"], pf["formation lap"], pf["end lap"]
            rates_normalized_within_bounds = np.nanmax(tc.rate_matrix[lb:ub, :], axis=0) / np.nanmax(tc.rate_matrix[lb:ub, :])

            # ALR = active lap ratio: N(active laps within PF bounds)/N(total laps)
            alr = np.sum(rates_normalized_within_bounds >= 0.1) / len(rates_normalized_within_bounds)
            self.pfs_df.loc[i_pf, "ALR"] = alr

            # PF width (in cm)
            self.pfs_df.loc[i_pf, "PF width"] = (self.pfs_df.loc[i_pf, "upper bound"] - self.pfs_df.loc[i_pf, "lower bound"]) * self.bin_length

            # COM standard deviance in active laps
            act_laps_idxs = np.where(rates_normalized_within_bounds >= 0.1)[0]
            RM_active_laps = tc.rate_matrix[lb:ub][:, act_laps_idxs]
            COM = lambda arr: np.average(np.array(list(range(len(arr)))), weights=arr)  # weighted avg. of indices of input array, weights = input array itself
            COMs_active_laps = np.apply_along_axis(COM, axis=0, arr=RM_active_laps)
            COM_SD = COMs_active_laps.std()
            self.pfs_df.loc[i_pf, "COM SD"] = COM_SD
            #self.pfs_df.loc[i_pf, "dCOMs"] = [np.diff(COMs_active_laps)]

            # PF width-normalized COM SD
            self.pfs_df.loc[i_pf, "Norm. COM SD"] = COM_SD / self.pfs_df.loc[i_pf, "PF width"]

            # PF mean COM: LB added so that it is no longer relative to LB, but relative to the beginning of track
            self.pfs_df.loc[i_pf, "Mean COM"] = self.pfs_df.loc[i_pf, "lower bound"] + COMs_active_laps.mean()
            self.pfs_df.loc[i_pf, "Mean rate"] = RM_active_laps.mean()
            self.pfs_df.loc[i_pf, "Max rate"] = RM_active_laps.max()

            # CALR = captured active lap ratio: N(active laps within PF bounds *and lifespan*)/N(total active laps)
            if pf["category"] == "unreliable":
                continue
            calr = np.sum(rates_normalized_within_bounds[fl:el] >= 0.1) / np.sum(rates_normalized_within_bounds >= 0.1)
            self.pfs_df.loc[i_pf, "CALR"] = calr

    def plot_place_field_quality(self):
        def sort_df(series):
            return series.apply(lambda x: CATEGORIES[x].order)
        pfs_sorted = self.pfs_df.sort_values(by="category", key=sort_df)
        cols = ['ALR', 'CALR', 'COM SD', 'PF width', 'Norm. COM SD']
        #cols = ['ALR', 'COM SD', 'PF width', 'Norm. COM SD']
        ylims = [[0,1], [0,1], [0,7], [0,100], [0, 0.2]]
        #ylims = [[0,1], [0,7], [0,100], [0,0.5]]

        scale = 1.3
        fig, axs = plt.subplots(1, len(cols), figsize=(scale * len(cols)*5,scale * 2.8))
        palette = ["#00B0F0", "#F5B800"]
        for i, col in enumerate(cols):
            pfs_filtered = pfs_sorted[pfs_sorted["category"] != "unreliable"]
            pf_quality_df = pfs_filtered[["newly formed", col]]
            is_legend = False #if i < len(cols)-1 else True
            sns.violinplot(pf_quality_df.melt(id_vars="newly formed"), x="variable", y="value", hue="newly formed",
                        palette=palette, saturation=1, ax=axs[i], legend=is_legend, linewidth=2, cut=0)
            axs[i].set_ylim(ylims[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("")
            axs[i].set_title(col)
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].set_xticks([], [])
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_PFquality_NFvEst.pdf")
        plt.savefig(f"{self.output_root}/plot_PFquality_NFvEst.svg")
        plt.close()

        fig, axs = plt.subplots(1, len(cols), figsize=(scale * len(cols)*5,scale * 2.8))
        for i, col in enumerate(cols):
            pf_quality_df = pfs_sorted[["category", col]]
            is_legend = False #if i < len(cols)-1 else True
            sns.violinplot(pf_quality_df.melt(id_vars="category"), x="variable", y="value", hue="category",
                        palette=self.categories_colors_RGB, saturation=1, ax=axs[i], legend=is_legend,
                        linewidth=2, cut=0)
            axs[i].set_ylim(ylims[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("")
            axs[i].set_title(col)
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].set_xticks([], [])
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_PFquality_categories.pdf")
        plt.savefig(f"{self.output_root}/plot_PFquality_categories.svg")
        plt.close()

    def calc_lap_by_lap_metrics(self):
        rng = np.random.default_rng(seed=1234)

        self.pfs_df = self.pfs_df.sample(frac=1).reset_index(drop=True)  # shuffle place fields
        pf_lap_dicts_all = []
        for i_pf, pf in self.pfs_df.iterrows():
            if pf["category"] == "unreliable":
                continue
            if i_pf % 100 == 0:
                print(f"{i_pf} / {len(self.pfs_df)}")

            tc = self.tc_df[(self.tc_df["sessionID"] == pf["session id"]) & (self.tc_df["cellid"] == pf["cell id"]) & (self.tc_df["corridor"] == pf["corridor"])]["tc"].iloc[0]

            #rate_matrix_pf = tc.rate_matrix[pf["lower bound"]:pf["upper bound"], pf["formation lap"]:pf["end lap"]]
            lb, ub, fl, el = pf["lower bound"], pf["upper bound"], pf["formation lap"], pf["end lap"]
            rates_normalized_within_bounds = np.nanmax(tc.rate_matrix[lb:ub, :], axis=0) / np.nanmax(tc.rate_matrix[lb:ub, :])

            # contains indices of active laps:
            act_laps_idxs = np.where(rates_normalized_within_bounds >= 0.1)[0]

            # filter active laps: remove formation lap and subsequent 2 active laps to allow initial shifting
            #act_laps_idxs = act_laps_idxs[(act_laps_idxs >= fl + 3) | (act_laps_idxs < fl)]

            # list of indices of consecutively active laps (list of lists)
            list_cons_act_laps_idxs = np.array_split(act_laps_idxs, np.flatnonzero(np.diff(act_laps_idxs) > 1) + 1)

            pf_lap_dicts = []
            for cons_act_laps_idxs in list_cons_act_laps_idxs:
                # this is where we filter non-consecutive active laps:
                # chunks of len < 2 mean that an inactive lap came immediately after an active one
                if len(cons_act_laps_idxs) < 2:
                    continue

                # RM = rate matrix
                RM_all_active_laps = tc.rate_matrix[lb:ub][:,act_laps_idxs]
                RM_cons_active_laps = tc.rate_matrix[lb:ub][:,cons_act_laps_idxs]  # skip first 3 active laps to allow for initial shift
                RM_all_active_laps_z = (RM_all_active_laps - tc.rate_matrix.mean()) / tc.rate_matrix.std()  # z-scoring using the whole cell, not just PF spatio-temporal bounds
                RM_cons_active_laps_z = (RM_cons_active_laps - tc.rate_matrix.mean()) / tc.rate_matrix.std()

                COM = lambda arr: np.average(np.array(list(range(len(arr)))), weights=arr)  # weighted avg. of indices of input array, weights = input array itself
                COMs_cons_active_laps = np.apply_along_axis(COM, axis=0, arr=RM_cons_active_laps)

                max_activity_cons_active_laps = RM_cons_active_laps.max(axis=0)
                max_activity_cons_active_laps_normalized = max_activity_cons_active_laps / RM_cons_active_laps.max()
                max_activity_cons_active_laps_z = RM_cons_active_laps_z.max(axis=0)
                max_activity_cons_active_laps_normalized_z = max_activity_cons_active_laps_z / RM_all_active_laps_z.max()

                sum_activity_cons_active_laps = RM_cons_active_laps.sum(axis=0)
                sum_activity_cons_active_laps_normalized = sum_activity_cons_active_laps / RM_cons_active_laps.sum()
                sum_activity_cons_active_laps_z = RM_cons_active_laps_z.sum(axis=0)
                sum_activity_cons_active_laps_normalized_z = sum_activity_cons_active_laps_z / RM_all_active_laps_z.sum()

                dCOMs = list(np.diff(COMs_cons_active_laps))
                for i_lap_idx, lap_idx in enumerate(cons_act_laps_idxs[1:-1]):  # [1:-1] so that we can check lap before and after and calc. ratio
                    # i_lap_idx: lap location within consecutive list (so always starts at zero)
                    # lap_idx: lap location among all laps (doesn't necessarily start at zero)
                    if i_lap_idx < len(cons_act_laps_idxs):
                        dCOM = dCOMs[i_lap_idx]
                        next_lap_max_act = max_activity_cons_active_laps_normalized[i_lap_idx+1]
                        next_lap_sum_act = sum_activity_cons_active_laps_normalized[i_lap_idx+1]
                        next_lap_max_act_z = max_activity_cons_active_laps_normalized_z[i_lap_idx+1]
                        next_lap_sum_act_z = sum_activity_cons_active_laps_normalized_z[i_lap_idx+1]
                    else:
                        dCOM = np.nan
                        next_lap_max_act = np.nan
                        next_lap_sum_act = np.nan
                        next_lap_max_act_z = np.nan
                        next_lap_sum_act_z = np.nan
                    curr_lap_max_act = max_activity_cons_active_laps_normalized[i_lap_idx]
                    curr_lap_sum_act = sum_activity_cons_active_laps_normalized[i_lap_idx]
                    curr_lap_max_act_z = max_activity_cons_active_laps_normalized_z[i_lap_idx]
                    curr_lap_sum_act_z = sum_activity_cons_active_laps_normalized_z[i_lap_idx]

                    prev_lap_max_act_z = max_activity_cons_active_laps_normalized_z[i_lap_idx-1]
                    prev_lap_sum_act_z = sum_activity_cons_active_laps_normalized_z[i_lap_idx-1]
                    max_act_z_ratio = next_lap_max_act_z / prev_lap_max_act_z
                    sum_act_z_ratio = next_lap_sum_act_z / prev_lap_sum_act_z
                    pf_lap_dict = {
                        "session id": pf["session id"],
                        "cell id": pf["cell id"],
                        "corridor": pf["corridor"],
                        "category": pf["category"],
                        "lap number": lap_idx,
                        "dCOM": dCOM,
                        "max activity[i] (norm.)": curr_lap_max_act,
                        "max activity[i+1] (norm.)": next_lap_max_act,
                        "sum activity[i] (norm.)": curr_lap_sum_act,
                        "sum activity[i+1] (norm.)": next_lap_sum_act,
                        "max activity[i] (norm., z-scored)": curr_lap_max_act_z,
                        "max activity[i+1] (norm., z-scored)": next_lap_max_act_z,
                        "sum activity[i] (norm., z-scored)": curr_lap_sum_act_z,
                        "sum activity[i+1] (norm., z-scored)": next_lap_sum_act_z,
                        "max activity ratio (z)": max_act_z_ratio,
                        "sum activity ratio (z)": sum_act_z_ratio,
                    }
                    pf_lap_dicts.append(pf_lap_dict)
            shuffled_laps_idxs = rng.permutation(len(pf_lap_dicts))
            pf_lap_dicts_extended = []
            for i_pf_lap_dict, pf_lap_dict in enumerate(pf_lap_dicts):
                shuffled_idx = shuffled_laps_idxs[i_pf_lap_dict]  # select index from shuffled lap index list
                if shuffled_idx+1 == i_pf_lap_dict:
                    # these are the cases when the next activity would be the current lap's activity so we fill up the diagonal
                    pf_lap_dict["sum activity[i+1] (norm., z-scored, shuffled)"] = np.nan
                else:
                    shuffled_lap_dict = pf_lap_dicts[shuffled_idx]  # select lap_dict belonging to said index
                    pf_lap_dict["sum activity[i+1] (norm., z-scored, shuffled)"] = shuffled_lap_dict["sum activity[i+1] (norm., z-scored)"]
                pf_lap_dicts_extended.append(pf_lap_dict)
            pf_lap_dicts = pf_lap_dicts_extended
            pf_lap_dicts_all += pf_lap_dicts

        self.pfs_laps_df = pd.DataFrame.from_dict(pf_lap_dicts_all)
        self.pfs_laps_df["log sum[i]"] = np.log10(self.pfs_laps_df["sum activity[i] (norm.)"])
        self.pfs_laps_df["log sum[i+1]"] = np.log10(self.pfs_laps_df["sum activity[i+1] (norm.)"])
        self.pfs_laps_df["log sum[i] (z)"] = np.log10(self.pfs_laps_df["sum activity[i] (norm., z-scored)"])
        self.pfs_laps_df["log sum[i+1] (z)"] = np.log10(self.pfs_laps_df["sum activity[i+1] (norm., z-scored)"])

    def plot_lap_by_lap_metrics(self):
        # bin activities into a list of act.ranges, so we can plot distributions
        step = 0.1
        act_range = np.arange(0.1, 1+step, step)  # 1+step is necessary to include 1
        max_find_nearest = lambda row: np.round(act_range[np.abs(act_range - row["max activity[i] (norm.)"]).argmin()],3)
        sum_find_nearest = lambda row: np.round(act_range[np.abs(act_range - row["sum activity[i] (norm.)"]).argmin()],3)
        self.pfs_laps_df["max activity range"] = self.pfs_laps_df.apply(max_find_nearest, axis=1)
        self.pfs_laps_df["sum activity range"] = self.pfs_laps_df.apply(sum_find_nearest, axis=1)

        """
        # TODO: avokádó plot
        plt.figure()
        sns.histplot(data=self.pfs_laps_df.reset_index(), x="log sum[i]", y="log sum[i+1]", binrange=[-3, 0], binwidth=1 / 30, cmap="viridis")
        plt.axline((0, 0), (1, 1), c="r")
        plt.xlim([-3, 0])
        plt.ylim([-3, 0])

        # TODO: űrhajó
        plt.figure()
        sns.histplot(data=self.pfs_laps_df.reset_index(), x="log sum[i] (z)", y="log sum[i+1] (z)", binrange=[-5, 2], binwidth=1 / 30, cmap="viridis")
        plt.axline((0, 0), (1, 1), c="r")
        plt.axhline(0, c="r", linestyle="--")
        plt.axvline(0, c="r", linestyle="--")

        # TODO: shuffled űrhajó; nem biztos hogy jó így a shuffling mert lehet sejteken belül kéne (most össze vannak kutyulva sejteken át)
        #self.pfs_laps_df["log sum[i+1] (z, shuff)"] = self.pfs_laps_df["log sum[i+1] (z)"].sample(frac=1).values
        self.pfs_laps_df["log sum[i+1] (z, pf-shuff)"] = np.log10(self.pfs_laps_df["sum activity[i+1] (norm., z-scored, shuffled)"])
        plt.figure()
        sns.histplot(data=self.pfs_laps_df.reset_index(), x="log sum[i] (z)", y="log sum[i+1] (z, pf-shuff)", binrange=[-5, 2], binwidth=1 / 30, cmap="viridis")
        plt.axline((0, 0), (1, 1), c="r")
        plt.axhline(0, c="r", linestyle="--")
        plt.axvline(0, c="r", linestyle="--")
        """

        # TODO: plot for next time (04/03/24)
        #bin_size = 0.025
        #bin_range = np.arange(0, 1+bin_size, bin_size)

        #plt.figure()
        #bin_data = lambda df: df.groupby(pd.cut(df["activity[i] (norm.)"], bin_range, labels=bin_range[1:], right=True))["dCOM"].median().reset_index()["dCOM"]
        #plt.plot(bin_range[1:], bin_data(self.pfs_laps_df[self.pfs_laps_df["category"] == "early"]), color=self.categories_colors_RGB[1], label="established")
        #plt.plot(bin_range[1:], bin_data(self.pfs_laps_df[self.pfs_laps_df["category"] == "transient"]), color=self.categories_colors_RGB[2], label="transient")
        #plt.plot(bin_range[1:], bin_data(self.pfs_laps_df[self.pfs_laps_df["category"] == "non-btsp"]), color=self.categories_colors_RGB[3], label="non-BTSP")
        #plt.plot(bin_range[1:], bin_data(self.pfs_laps_df[self.pfs_laps_df["category"] == "btsp"]), color=self.categories_colors_RGB[4], label="BTSP")
        #plt.legend()

        #plt.figure()
        #sns.histplot(data=self.pfs_laps_df.reset_index(), x="activity[i] (norm.)", y="activity[i+1] (norm.)",
        #             binrange=[0, 1], binwidth=1 / 30, cmap="viridis")
        #plt.axline((0, 0), slope=1, c="red")
        #plt.xlim([0.1, 1])
        #plt.ylim([0.1, 1])
        #plt.show()


        #plt.close()

        # TODO this is the shit
        sns.catplot(data=self.pfs_laps_df, x="max activity range", y="dCOM", kind="box", showfliers=False, linewidth=2, fill=False)
        plt.axhline(0, c="r", linestyle="--")

        #plt.figure()
        #sns.histplot(data=self.pfs_laps_df, x="max activity[i] (norm., z-scored)", y="max activity ratio (z)")
        #plt.show()
        #sns.jointplot(data=self.pfs_laps_df[(self.pfs_laps_df["max activity ratio (z)"] > 0) & (self.pfs_laps_df["max activity ratio (z)"] < 2)],
        #              kind="hist", x="max activity[i] (norm., z-scored)", y="max activity ratio (z)")
        #plt.show()
        #sns.catplot(data=self.pfs_laps_df, x="max activity range", y="max activity ratio (z)", kind="box",
        #            showfliers=False)
        #plt.axhline(1, c="r", linestyle="--")

        sns.catplot(data=self.pfs_laps_df, x="max activity range", y="max activity ratio (z)", kind="box", showfliers=False, linewidth=2, fill=False)
        plt.axhline(1, c="r", linestyle="--")

        # TODO subsample to CA3, n=4892 -- this number is old
        #if self.area == "CA1":
        #    subsamples = []
        #    for i in range(1000):
        #        subsample = self.pfs_laps_df.sample(n=4892).groupby(["max activity range"])["max activity ratio (z)"].median()
        #        subsamples.append(subsample)
        #    subsample_df = pd.DataFrame(subsamples)
        #    sns.catplot(data=subsample_df, kind="violin", color="blue", fill=False)
        #    plt.axhline(1, c="r", linestyle="--"); plt.ylim([0,4.5])
        plt.show()


        # (TODO) run this from debugger -- distribution of p-values under 100 sampling of all pfs where subsample is the size of the last (smallest) activity group
        #plt.figure(); [plt.plot([scipy.stats.wilcoxon(self.pfs_laps_df.sample(6583)[self.pfs_laps_df["activity range"] == val]["dCOM"]).pvalue for val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], marker="o", c="k", alpha=0.05) for i in range(100)]; plt.axhline(0.05, c="r");

        #df_all = pd.DataFrame(act_norm_act_all[1:], columns=["activity(i)", "activity(i+1)"])
        #bin_data = lambda ddf: ddf.groupby(pd.cut(ddf["activity(i)"], bin_range, labels=bin_range[1:], right=True))["activity(i+1)"].median().reset_index()["activity(i+1)"]
        #df_all_filt = df_all[(df_all["activity(i)"] != 1) & (df_all["activity(i+1)"] != 1)]
        #plt.figure()
        #plt.plot(bin_range, bin_data(df_all_filt), color="k")
        #plt.plot(bin_range, bin_data(df_all), color="b")
        #plt.axline((0, 0), slope=1, c="red")
        #plt.xlim([0.1, 1])
        #plt.ylim([0.1, 1])
        #plt.show()

    def calc_place_field_quality_all_laps(self):

        def get_tc(session_id, cellid, corridor):
            return [tc for tc in tcl_all_sessions if tc.sessionID == session_id and tc.cellid == cellid and tc.corridor == corridor][0]

        sessions = np.unique(self.pfs_df["session id"].values)
        tcl_all_sessions = []
        for session in sessions:
            print(session)
            tcl_path = f"{self.data_path}/tuned_cells/{self.area}{self.extra_info}/tuned_cells_{session}.pickle"
            with open(tcl_path, "rb") as tcl_file:
                tcl = pickle.load(tcl_file)
                tcl_all_sessions += tcl
        tc_df = pd.DataFrame([{"sessionID": tc.sessionID, "cellid": tc.cellid, "corridor": tc.corridor, "tc": tc} for tc in tcl_all_sessions])


        self.pfs_df = self.pfs_df.sample(frac=1).reset_index(drop=True)  # shuffle place fields
        pf_lap_dicts_all = []
        for i_pf, pf in self.pfs_df.iterrows():
            if i_pf % 100 == 0:
                print(f"{i_pf} / {len(self.pfs_df)}")
            tc = tc_df[(tc_df["sessionID"] == pf["session id"]) & (tc_df["cellid"] == pf["cell id"]) & (tc_df["corridor"] == pf["corridor"])]["tc"].iloc[0]

            #rate_matrix_pf = tc.rate_matrix[pf["lower bound"]:pf["upper bound"], pf["formation lap"]:pf["end lap"]]
            lb, ub, fl, el = pf["lower bound"], pf["upper bound"], pf["formation lap"], pf["end lap"]

            rate_sums_laps = tc.rate_matrix[lb:ub].sum(axis=0)
            rate_sums_laps = rate_sums_laps / rate_sums_laps.max()
            #rate_sums_laps = (rate_sums_laps - rate_sums_laps.mean())/rate_sums_laps.std()  # z-scoring
            for i_lap, lap in enumerate(rate_sums_laps):
                if i_lap == 0 or i_lap == len(rate_sums_laps)-1:
                    continue  # first lap has no prev.lap, last lap has no subseq.lap
                rate_sum_diff = rate_sums_laps[i_lap+1] - rate_sums_laps[i_lap-1]
                pf_lap_dict = {
                    "session id": pf["session id"],
                    "cell id": pf["cell id"],
                    "corridor": pf["corridor"],
                    "category": pf["category"],
                    "lap number": i_lap,
                    "rate sum[i]": rate_sums_laps[i_lap],
                    "rate sum diff": rate_sum_diff
                }
                pf_lap_dicts_all.append(pf_lap_dict)
        self.pfs_laps_df = pd.DataFrame.from_dict(pf_lap_dicts_all)
        cat_condition = (self.pfs_laps_df["category"] == "btsp") | (self.pfs_laps_df["category"] == "non-btsp") | (self.pfs_laps_df["category"] == "early")
        sns.jointplot(data=self.pfs_laps_df, x="rate sum[i]", y="rate sum diff", kind="hist")
        sns.jointplot(data=self.pfs_laps_df[(np.abs(self.pfs_laps_df["rate sum diff"])>0) & (self.pfs_laps_df["rate sum[i]"]>0)][cat_condition], x="rate sum[i]", y="rate sum diff", kind="hist")
        plt.axhline(0, c="r")


        plt.figure()
        # bin activities into a list of act.ranges, so we can plot distributions
        step = 0.05
        act_range = np.arange(0.1, 1+step, 0.05)  # 1+step is necessary to include 1
        max_find_nearest = lambda row: np.round(act_range[np.abs(act_range - row["rate sum[i]"]).argmin()],3)
        #sum_find_nearest = lambda row: np.round(act_range[np.abs(act_range - row["sum activity[i] (norm.)"]).argmin()],3)
        self.pfs_laps_df["max activity range"] = self.pfs_laps_df.apply(max_find_nearest, axis=1)
        #self.pfs_laps_df["sum activity range"] = self.pfs_laps_df.apply(sum_find_nearest, axis=1)
        sns.catplot(data=self.pfs_laps_df[(np.abs(self.pfs_laps_df["rate sum diff"])>0) & (self.pfs_laps_df["rate sum[i]"]>0)][cat_condition], x="max activity range", y="rate sum diff", kind="box", showfliers=False)
        plt.axhline(0, c="r")
        plt.show()

        #pfs_laps_df = self.pfs_laps_df[cat_condition]
        #pfs_laps_df = pfs_laps_df[np.abs(self.pfs_laps_df["rate sum diff"]) > 0]
        #sns.jointplot(data=pfs_laps_df.sample(30000), x="rate sum[i]", y="rate sum diff", hue="category", kind="kde", marginal_kws={"common_norm": False})
        pass

    def plot_lifespan(self):
        palette = ["#00B0F0", "#F5B800"]

        # filter pfs
        pfs = self.pfs_df[self.pfs_df["category"] != "unreliable"]
        #pfs = pfs[pfs["category"] != "transient"]

        # calculate lifespan
        pfs["lifespan"] = pfs["end lap"] - pfs["formation lap"]
        pfs["width"] = pfs["upper bound"] - pfs["lower bound"]
        pfs_w_norm_all = None
        for animal in self.animals:
            sessions_animal = list(set(pfs[pfs["animal id"] == animal]["session id"]))
            #fig, axs = plt.subplots(1,len(sessions_animal))
            pfs_w_norm = None
            for i_sess, session in enumerate(sessions_animal):
                pfs_sess = pfs[pfs["session id"] == session]
                pfs_sess["end lap (norm.)"] = pfs_sess["end lap"] / pfs_sess["end lap"].max()
                pfs_sess["lifespan (norm.)"] = pfs_sess["lifespan"] / pfs_sess["end lap"].max()
                pfs_sess["formation lap (norm.)"] = pfs_sess["formation lap"] / pfs_sess["end lap"].max()
                #if pfs_sess["end lap"].max() < 50:
                #    continue
                if i_sess == len(sessions_animal)-1:
                    legend = True
                else:
                    legend = False
                pfs_w_norm = grow_df(pfs_w_norm, pfs_sess)
                #sns.histplot(data=pfs_sess, x="end lap (norm.)", hue="newly formed", ax=axs[i_sess],
                #             palette=sns.color_palette(palette, 2), legend=legend, kde=True)
                #sns.jointplot(data=pfs_sess, x="formation lap", y="end lap", hue="newly formed")
                #pfs_sess["lived until end"] = True if pfs_sess["end lap (norm.)"] == 1 else False
            pfs_w_norm_all = grow_df(pfs_w_norm_all, pfs_w_norm)
        palette_cats = self.categories_colors_RGB[1:] #[self.categories_colors_RGB[1]]+ self.categories_colors_RGB[3:]
        thresh = 1.0
        fig, axs = plt.subplots(3,2)
        sns.histplot(data=pfs_w_norm_all, x="lifespan (norm.)", hue="newly formed", palette=palette, ax=axs[0,0])
        sns.histplot(data=pfs_w_norm_all[pfs_w_norm_all["end lap (norm.)"] >= thresh], x="lifespan (norm.)", hue="newly formed", palette=palette, ax=axs[1,0])
        sns.histplot(data=pfs_w_norm_all[pfs_w_norm_all["end lap (norm.)"] < thresh], x="lifespan (norm.)", hue="newly formed", palette=palette, ax=axs[1,1])
        sns.histplot(data=pfs_w_norm_all[pfs_w_norm_all["end lap (norm.)"] >= thresh], x="lifespan", hue="newly formed", palette=palette, ax=axs[2,0])
        sns.histplot(data=pfs_w_norm_all[pfs_w_norm_all["end lap (norm.)"] < thresh], x="lifespan", hue="newly formed", palette=palette, ax=axs[2,1])
        plt.show()

    def depth_analysis(self, depth_folder):
        if self.area == "CA1":
            print("this only works for CA3; aborting")
            return
        sessions = np.unique(self.pfs_df["session id"].values)
        depths_df_all = None
        for session in sessions:
            try:
                depths_df = pd.read_excel(f"{depth_folder}/cellDepth_depths_{session}.xlsx")
            except FileNotFoundError:
                print(f"failed to load depths for {session}; skippping")
                continue
            depths_df = depths_df.reset_index().rename(columns={"index": "cell id"})  # create cell ids from suite2p roi indices
            animal = session.partition("_")[0]
            depths_df["animal id"] = animal
            depths_df["session id"] = session
            depths_df_all = grow_df(depths_df_all, depths_df)
        pfs_df = self.pfs_df.set_index(["animal id", "session id", "cell id"])
        depths_df_all = depths_df_all.set_index(["animal id", "session id", "cell id"])
        pfs_df = pfs_df.join(depths_df_all, on=["animal id", "session id", "cell id"]).reset_index()
        depths_df_all = depths_df_all.reset_index()

        tc_df = self.tc_df.rename(columns={"sessionID": "session id", "cellid": "cell id"})
        tc_df["Ca2+ event rate"] = tc_df.apply(lambda row: 60 * row["tc"].n_events / row["tc"].total_time, axis=1)
        tc_df = tc_df[tc_df["corridor"] == 14].set_index(["session id", "cell id"])  # we can select either corridor bc. event rate is a cell-level property

        # load all cells (i.e not just tuned ones), calculate event rates for all of them
        sessions = np.unique(self.pfs_df["session id"].values)
        cells_all_sessions = []
        for session in sessions:
            print(session)
            cell_list_path = f"{self.data_path}/tuned_cells/{self.area}{self.extra_info}/all_cells_{session}.pickle"
            with open(cell_list_path, "rb") as cell_list_file:
                cell_list = pickle.load(cell_list_file)
                cells_all_sessions += cell_list
        cells_df = pd.DataFrame([{"sessionID": c.sessionID, "cellid": c.cellid, "cell": c} for c in cells_all_sessions])
        cells_df = cells_df.rename(columns={"sessionID": "session id", "cellid": "cell id"})
        cells_df["Ca2+ event rate"] = cells_df.apply(lambda row: 60 * row["cell"].n_events / row["cell"].total_time, axis=1)
        cells_df = cells_df.set_index(["session id", "cell id"])

        # join cell tables with depth tables
        depths_df_tuned = depths_df_all.set_index(["session id", "cell id"])
        depths_df_tuned = depths_df_all.join(tc_df, on=["session id", "cell id"]).reset_index()
        depths_df_all = depths_df_all.set_index(["session id", "cell id"])
        depths_df_all = depths_df_all.join(cells_df, on=["session id", "cell id"]).reset_index()

        plt.figure()
        sns.histplot(data=depths_df_tuned, x="depth")

        # discard ROIs that are clearly out of the SP
        depths_df_tuned = depths_df_tuned[(depths_df_tuned["depth"] >= -1) & (depths_df_tuned["depth"] < 1.5)]
        depths_df_all = depths_df_all[(depths_df_all["depth"] >= -1) & (depths_df_all["depth"] < 1.5)]
        pass

        def create_depth_bins(bounds, step):
            lb, ub = bounds
            depth_range = np.arange(lb, ub+step, step)  # 1+step is necessary to include 1
            find_nearest = lambda row: np.round(depth_range[np.abs(depth_range - row["depth"]).argmin()],3)
            depths_df_tuned["depth range"] = depths_df_tuned.apply(find_nearest, axis=1)

        ######## PLOTS
        # distribution of depths
        plt.figure()
        #sns.kdeplot(data=depths_df_tuned, x="depth")
        sns.kdeplot(data=depths_df_tuned, x="depth", hue="animal id", common_norm=False)
        plt.show()

        # event rates
        sns.jointplot(data=depths_df_all, x="depth", y="Ca2+ event rate", kind='hex', ylim=(0, 5))
        sns.jointplot(data=depths_df_tuned, x="depth", y="Ca2+ event rate", kind="hist")

        create_depth_bins(bounds=[-1,1.5], step=0.1)
        g = sns.catplot(data=depths_df_tuned, x="depth range", y="Ca2+ event rate", kind="box", fill=False, showfliers=False)
        g.map_dataframe(sns.stripplot, data=depths_df_tuned, x="depth range", y="Ca2+ event rate", alpha=0.5, dodge=True)

        # TODO: expand this to all cells, not just tuned cells
        #create_depth_bins(bounds=[-0.6, 1], step=0.2)
        #g = sns.catplot(data=depths_df_tuned, x="depth range", y="Ca2+ event rate", kind="violin", fill=False, cut=0, density_norm="width", width=0.9)
        #g.map_dataframe(sns.swarmplot, data=depths_df_tuned, x="depth range", y="Ca2+ event rate", alpha=0.5, dodge=True, color="black")

        # place field distributions
        plt.figure()
        sns.histplot(data=pfs_df.sort_values(by="category_order"), x="depth", hue="category", multiple="stack",
                     binrange=(-1, 1.5), binwidth=0.1, palette=self.categories_colors_RGB)
        plt.figure()
        sns.histplot(data=pfs_df.sort_values(by="category_order"), x="depth", hue="category", multiple="fill",
                     binrange=(0, 1), binwidth=0.1, palette=self.categories_colors_RGB)
        plt.figure()
        sns.histplot(data=pfs_df.sort_values(by="category_order"), x="depth", hue="category", multiple="fill",
                     binrange=(0, 1), binwidth=0.25, palette=self.categories_colors_RGB)
        plt.show()
        plt.close()



    def plot_session_length_dependence(self):
        pf_counts = self.pfs_df.groupby(["session length", "category"]).count().rename(columns={"animal id": "pf count"})
        pf_counts = pf_counts.reset_index()[["session length", "category", "pf count"]]
        pf_counts["pf prop"] = pf_counts["pf count"] / pf_counts.groupby(["session length"])["pf count"].transform("sum")

        CATEGORIES_DICT = {
            "unreliable": "#a2a1a1",
            "early": "#00B0F0",
            "transient": "#77D600",
            "non-btsp": "#FF0000",
            "btsp": "#AA5EFF",
        }
        pf_props_wide = pf_counts.pivot(index="session length", columns="category", values="pf prop")
        cats = ["unreliable", "early", "transient", "non-btsp", "btsp"]  # so that they are in the desired order
        pf_props_wide[cats].plot.bar(stacked=True, color=CATEGORIES_DICT, width=1)

        ### effect of animal vs session length
        pf_counts = self.pfs_df.groupby(["animal id", "session length"]).count().rename(columns={"cell id": "pf count"})
        pf_counts = pf_counts.reset_index()[["animal id", "session length", "pf count"]]
        pf_counts["animal idx"] = pf_counts["animal id"].factorize()[0]
        parcorr = pingouin.partial_corr(data=pf_counts, x="session length", y="pf count", covar="animal idx", method="spearman")
        print("partial correlation coeff. between session length, pf count and animal id")
        print(parcorr)

    def plot_formation_lap_history_dependence(self):
        pfs_filt = self.pfs_df[(self.pfs_df["category"] != "early") & (self.pfs_df["category"] != "unreliable")]
        pf_counts = pfs_filt.groupby(["formation lap history", "category"]).count().rename(columns={"animal id": "pf count"})
        pf_counts = pf_counts.reset_index()[["formation lap history", "category", "pf count"]]
        pf_counts["pf prop"] = pf_counts["pf count"] / pf_counts.groupby(["formation lap history"])["pf count"].transform("sum")

        CATEGORIES_DICT = {
            "transient": "#77D600",
            "non-btsp": "#FF0000",
            "btsp": "#AA5EFF",
        }
        pf_props_wide = pf_counts.pivot(index="formation lap history", columns="category", values="pf prop")
        cats = ["transient", "non-btsp", "btsp"]  # so that they are in the desired order
        pf_props_wide[cats].plot.bar(stacked=True, color=CATEGORIES_DICT, width=0.8)

    def plot_formations(self):
        makedir_if_needed(f"{self.output_root}/formations")
        for session_id in np.unique(self.pfs_df["session id"]):
            fig, axs = plt.subplots(1,2)
            pfs = self.pfs_df[self.pfs_df["category"] != "unreliable"]
            pfs_14 = pfs[(pfs["session id"] == session_id) & (pfs["corridor"] == 14)]
            pfs_15 = pfs[(pfs["session id"] == session_id) & (pfs["corridor"] == 15)]
            sns.histplot(ax=axs[0], data=pfs_14, x="formation lap", hue="category", multiple="stack", binwidth=1, palette=self.categories_colors_RGB[1:])
            sns.histplot(ax=axs[1], data=pfs_15, x="formation lap", hue="category", multiple="stack", binwidth=1, palette=self.categories_colors_RGB[1:])
            axs[0].set_title("corridor 14")
            axs[1].set_title("corridor 15")
            plt.suptitle(f"distribution of formation laps of PFs in session {session_id}")
            plt.tight_layout()
            plt.savefig(f"{self.output_root}/formations/{session_id}_formations.pdf")

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

    def export_data(self):
        sg_df = self.shift_gain_df
        sg_df_NF = sg_df[sg_df["newly formed"] == True]
        sg_df_NF[["initial shift", "log10(formation gain)"]].to_csv(f"{self.output_root}/shift_gain_NF.csv", index=False)
        sg_df_ES = sg_df[sg_df["newly formed"] == False]
        sg_df_ES[["initial shift", "log10(formation gain)"]].to_csv(f"{self.output_root}/shift_gain_ES.csv", index=False)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
    parser.add_argument("-dp", "--data-path", required=True)
    parser.add_argument("-op", "--output-path")
    parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
    parser.add_argument("-hi", "--history", choices=["ALL", "STAY", "CHANGE"])
    #parser.add_argument("-np", "--no-shift-path")
    args = parser.parse_args()

    area = args.area
    data_path = args.data_path
    output_path = args.output_path
    extra_info = args.extra_info
    history = args.history
    #noshift_path = args.no_shift_path

    #run analysis
    is_shift_criterion_on = True
    is_notebook = False
    depth_folder = f"{data_path}/depths"

    btsp_statistics = BtspStatistics(area, data_path, output_path, extra_info, is_shift_criterion_on, is_notebook, history)
    btsp_statistics.load_data()
    btsp_statistics.load_cell_data()
    btsp_statistics.filter_low_behavior_score()
    btsp_statistics.calc_place_field_proportions()
    btsp_statistics.calc_shift_gain_distribution(unit="cm")
    btsp_statistics.calc_place_field_quality()
    #btsp_statistics.calc_lap_by_lap_metrics()
    #btsp_statistics.calc_place_field_quality_all_laps()

    #btsp_statistics.plot_formations()
    #btsp_statistics.plot_session_length_dependence()
    #btsp_statistics.plot_formation_lap_history_dependence()
    #plt.show()

    #btsp_statistics.plot_cells(swarmplot=False)
    #btsp_statistics.plot_place_fields()
    #btsp_statistics.plot_place_fields_by_session()
    #btsp_statistics.plot_place_field_proportions()
    #btsp_statistics.plot_place_fields_criteria()
    #btsp_statistics.plot_place_field_properties()
    #btsp_statistics.plot_place_field_heatmap()
    #btsp_statistics.plot_place_field_distro()
    #btsp_statistics.plot_place_fields_criteria_venn_diagram(with_btsp=True)
    #btsp_statistics.plot_shift_gain_distribution(without_transient=False)
    #btsp_statistics.plot_place_field_quality()
    #btsp_statistics.plot_lap_by_lap_metrics()
    btsp_statistics.depth_analysis(depth_folder)
    #btsp_statistics.plot_shift_scores()
    #btsp_statistics.plot_slopes_like_Madar()
    #btsp_statistics.plot_highgain_vs_lowgain_shift_diffs()
    #btsp_statistics.shift_gain_pca_sklearn()
    #btsp_statistics.shift_gain_pca()
    #btsp_statistics.plot_shift_scores()
    #btsp_statistics.dF_F_shifts()
    #btsp_statistics.plot_distance_from_RZ_distribution()  # TODO: ZeroDivisionError
    #btsp_statistics.plot_no_shift_criterion(noshift_path=noshift_path)
    #btsp_statistics.plot_cv_com_distro()
    #btsp_statistics.plot_shift_gain_dependence()
    #btsp_statistics.plot_example_place_fields_by_cv_com()
    #btsp_statistics.plot_lifespan()
    #btsp_statistics.export_results()
    #btsp_statistics.plot_formation_rate_sum()
    #btsp_statistics.plot_formation_rate_sum_hist_by_categories()
    #btsp_statistics.plot_formation_rate_sum(log=False)
    #btsp_statistics.plot_ratemaps()
    #btsp_statistics.calc_place_field_quality()
    #btsp_statistics.export_data()
