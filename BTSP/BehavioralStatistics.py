import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from BTSP.constants import (ANIMALS, SESSIONS_TO_IGNORE, ANIMALS_PALETTE, AREA_PALETTE,
                       VSEL_NORMALIZATION, BEHAVIOR_SCORE_THRESHOLD)
from utils import makedir_if_needed


class BehaviorStatistics:
    def __init__(self, area, data_path, output_path, extra_info="", makedir=True):
        self.area = area
        self.animals = ANIMALS[area]
        self.data_path = data_path

        # set output folder
        self.extra_info = "" if len(extra_info) == 0 else f"_{extra_info}"
        self.output_root = f"{output_path}/statistics/behavior_{self.area}{self.extra_info}"
        if makedir:
            makedir_if_needed(self.output_root)

        # load sessions data
        self.behavior_df = pd.read_pickle(f"{self.data_path}/behavior/{self.area}{self.extra_info}/behavior_df_{self.area}.pickle")
        self._use_font()

    def _use_font(self):
        from matplotlib import font_manager
        font_dirs = ['C:\\Users\\martin\\home\\phd\\misc']
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = 'Roboto'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Roboto'
        plt.rcParams['mathtext.it'] = 'Roboto'

    def calc_behavior_score(self):
        y_cols = ['area', 'animalID', 'sessionID', 'P-correct (14)', 'P-correct (15)', 'Speed index (14)', 'Speed index (15)',
                  'Speed selectivity', 'Lick index (14)', 'Lick index (15)', 'Lick selectivity']
        df = self.behavior_df[y_cols].set_index(["area", "animalID", "sessionID"])
        df[["Speed index (14)", "Speed index (15)", "Speed selectivity"]] = df[["Speed index (14)", "Speed index (15)", "Speed selectivity"]] / VSEL_NORMALIZATION
        df = df.sum(axis=1).to_frame().rename(columns={0: "behavior score"})

        # append behavior score as new column to behavior dataframe
        self.behavior_df = self.behavior_df.set_index(["area", "animalID", "sessionID"]).join(df).reset_index()

    def filter_low_behavior_score(self):
        self.behavior_df = self.behavior_df[self.behavior_df["behavior score"] > BEHAVIOR_SCORE_THRESHOLD]

    def plot_behavioral_metrics_for_each_animal(self):
        for animal in self.animals:
            beh_df_animal = self.behavior_df[self.behavior_df["animalID"] == animal]
            #palette = ["#000000",     "#9B9B9B",      "#00CEFF",  "#004EFF",  "#00D95F",      "#B5005B",  "#FF3D3D",  "#ED6FFF"]
            palette = ["#000000",     "#9B9B9B",      "#4A20FF",  "#5FA4FF",  "#00E874",      "#B5005B",  "#FF3D3D",  "#ED6FFF"]
            y_cols = ["P-correct (14)", "P-correct (15)", "Speed index (14)", "Speed index (15)", "Speed selectivity", "Lick index (14)", "Lick index (15)", "Lick selectivity"]
            beh_df_animal_selcols = beh_df_animal[["sessionID", "protocol", *y_cols]]
            beh_df_animal_selcols_melted = beh_df_animal_selcols.melt(["sessionID", "protocol"], var_name="cols", value_name="vals")

            plt.figure()
            ax = sns.lineplot(x="sessionID", y="vals", hue="cols", palette=palette, data=beh_df_animal_selcols_melted)
            sns.scatterplot(x="sessionID", style="protocol", y="vals", hue="cols", palette=palette, data=beh_df_animal_selcols_melted, ax=ax, s=60)
            handles, labels = ax.get_legend_handles_labels()
            plt.axhline(0, c="k", linestyle="--")
            plt.ylim([-1.1, 1.1])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            plt.legend(handles[len(y_cols):], labels[len(y_cols):], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            plt.savefig(f"{self.output_root}/{animal}_behavior.pdf")
            plt.close()

    def plot_behavioral_metrics_for_all_sessions(self):

        def norm(df):
            return df #(df - df.mean()) / df.std()

        plt.figure()

        palette = ["#FFFFFF", "#9B9B9B", "#4A20FF", "#5FA4FF", "#00E874", "#B5005B", "#FF3D3D", "#ED6FFF"]
        y_cols = ["sessionID", "P-correct (14)", "P-correct (15)", "Speed index (14)", "Speed index (15)", "Speed selectivity", "Lick index (14)", "Lick index (15)", "Lick selectivity"]
        #y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Speed index (14)", "Lick index (14)", "Speed index (15)", "Lick index (15)", "Speed selectivity", "Lick selectivity"]
        disengaged_sessions = ["srb251_221027_T1", "srb251_221027_T2"]

        # plot all sessions
        behavior_df_selcols = self.behavior_df[y_cols].set_index("sessionID")
        #behavior_df_selcols[["Speed index (14)", "Speed index (15)", "Speed selectivity"]] = -1 * behavior_df_selcols[["Speed index (14)", "Speed index (15)", "Speed selectivity"]]
        #behavior_df_selcols = norm(behavior_df_selcols)
        fig, ax = plt.subplots(figsize=(9,6))
        #ax = sns.boxplot(data=behavior_df_selcols, palette=palette, showfliers=False)
        ax.set_xticks(list(range(len(y_cols[1:]))))
        ax.set_xticklabels(y_cols[1:], rotation=45, ha="right")

        colors = [plt.cm.tab20(i) for i in range(20)]
        index = behavior_df_selcols.index
        for i, row in behavior_df_selcols.reset_index(drop=True).iterrows():
            x = 0 # 0.25*i/len(behavior_df_selcols)-0.125
            if i<20:
                plt.plot([0 + x, 1 + x], [row[0], row[1]], marker="o", color=colors[i % 20], linewidth=1.5, markersize=4, label=index[i])
                plt.plot([2 + x, 3 + x, 4 + x], [row[2], row[3], row[4]], marker="o", color=colors[i % 20], linewidth=1.5, markersize=4)
                plt.plot([5 + x, 6 + x, 7 + x], [row[5], row[6], row[7]], marker="o", color=colors[i % 20], linewidth=1.5, markersize=4)
            else:
                plt.plot([0 + x, 1 + x], [row[0], row[1]], marker="o", color=colors[i % 20], linewidth=1.5, markersize=4, label=index[i], markerfacecolor="white")
                plt.plot([2 + x, 3 + x, 4 + x], [row[2], row[3], row[4]], marker="o", color=colors[i % 20], linewidth=1.5, markersize=4, markerfacecolor="white")
                plt.plot([5 + x, 6 + x, 7 + x], [row[5], row[6], row[7]], marker="o", color=colors[i % 20], linewidth=1.5, markersize=4, markerfacecolor="white")


        # filter out disengaged sessions for scatterplot
        #behavior_df_selcols_scatter = self.behavior_df[~self.behavior_df["sessionID"].isin(disengaged_sessions)][y_cols].abs()
        #behavior_df_selcols_scatter = norm(behavior_df_selcols_scatter)
        #ax = sns.swarmplot(data=behavior_df_selcols_scatter, color="black")
        #handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[len(y_cols):], labels[len(y_cols):], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncols=2)
        plt.axhline(0, c="k", linestyle="--")
        plt.tight_layout()

        # plot disengaged sessions
        #de_sess_df = self.behavior_df[self.behavior_df["sessionID"].isin(disengaged_sessions)]
        #de_sess_df_selcols = de_sess_df[y_cols].abs()
        #de_sess_df_selcols = norm(de_sess_df_selcols)
        #sns.stripplot(data=de_sess_df_selcols, ax=ax, color="red")

        #import matplotlib as mpl
        #def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        #    c1 = np.array(mpl.colors.to_rgb(c1))
        #    c2 = np.array(mpl.colors.to_rgb(c2))
        #    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
        #c1 = 'red'  # blue
        #c2 = 'blue'  # green
        #behavior_df_selcols_scatter = behavior_df_selcols_scatter.sort_values(by=y_cols[0])
        #for i in range(behavior_df_selcols_scatter.shape[0]):
        #    plt.plot(behavior_df_selcols_scatter.values[i, :], color=colorFader(c1,c2,i/behavior_df_selcols_scatter.shape[0]))

        plt.savefig(f"{self.output_root}/behavioral_measures.pdf")
        plt.close()

    def plot_correlation_matrix(self):
        y_cols = ["P-correct (14)", "P-correct (15)", "Speed index (14)", "Speed index (15)", "Speed selectivity", "Lick index (14)", "Lick index (15)", "Lick selectivity"]
        behavior_df_selcols = self.behavior_df[y_cols].abs()  # we take absolute value, since selectivity in either direction is fine
        corr_matrix = behavior_df_selcols.corr()
        plt.figure()
        sns.heatmap(corr_matrix, annot=True)
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/corr_matrix.pdf")
        plt.close()

    def export_to_excel(self):
        with pd.ExcelWriter(f"{self.output_root}/behavior_{self.area}{self.extra_info}.xlsx") as xlsx_writer:
            self.behavior_df.to_excel(xlsx_writer, index=False)

    def plot_CA1_v_CA3(self, extra_info_CA1, extra_info_CA3):
        behav_CA1 = pd.read_pickle(f"{self.data_path}/behavior/CA1{extra_info_CA1}/behavior_df_CA1.pickle").reset_index(drop=True)
        behav_CA3 = pd.read_pickle(f"{self.data_path}/behavior/CA3{extra_info_CA3}/behavior_df_CA3.pickle").reset_index(drop=True)
        behav = pd.concat([behav_CA1, behav_CA3])  # merge dataframes
        behav["area"] = pd.Categorical(behav["area"], ["CA1", "CA3"])

        # filter low behavior score TODO: CODE DUPLICATION (see filter function above); FIX THIS
        y_cols = ['area', 'animalID', 'sessionID', 'P-correct (14)', 'P-correct (15)', 'Speed index (14)', 'Speed index (15)',
                  'Speed selectivity', 'Lick index (14)', 'Lick index (15)', 'Lick selectivity']
        df = behav[y_cols].set_index(["area", "animalID", "sessionID"])
        df[["Speed index (14)", "Speed index (15)", "Speed selectivity"]] = df[["Speed index (14)", "Speed index (15)", "Speed selectivity"]] / VSEL_NORMALIZATION
        df = df.sum(axis=1).to_frame().rename(columns={0: "behavior score"})
        behav = behav.set_index(["area", "animalID", "sessionID"]).join(df).reset_index()

        behav = behav[behav["behavior score"] > BEHAVIOR_SCORE_THRESHOLD]
        behav = behav[["area", "animalID", "Vmax [cm/s]", "Speed selectivity", "Lick selectivity"]]  # select columns
        behav = behav.melt(id_vars=["area", "animalID"])  # convert to long format


        #df = behav[behav["variable"] == "Vmax [cm/s]"]
        #g = sns.catplot(kind="box", data=df, x="area", y="value", showfliers=False)
        #g.map_dataframe(sns.swarmplot, x="area", y="value", hue="animalID", dodge=False, palette=ANIMALS_PALETTE)

        df = behav#[behav["variable"] != "Vmax [cm/s]"]
        sns.set_context("poster")
        g = sns.catplot(kind="box", data=df, x="area", y="value", col="variable", palette=AREA_PALETTE, gap=0.2,
                        showfliers=False, sharey=False, width=1, legend=False, linewidth=3, height=6, aspect=0.7, linecolor="k")
        palette = ANIMALS_PALETTE["CA1"] + ANIMALS_PALETTE["CA3"]
        #g.map_dataframe(sns.swarmplot, x="area", y="value", hue="animalID", dodge=False, size=8.25,
        #                palette=palette, linewidth=1.5, alpha=0.8, legend=False)
        g.axes[0, 0].set_ylim([0, 70])
        g.axes[0, 1].set_ylim([-1, 1.05])
        g.axes[0, 2].set_ylim([-1, 1.05])
        #g.axes[0, 2].spines['left'].set_visible(False)
        #g.axes[0, 2].tick_params(axis="y", left=False)
        #g.axes[0, 2].set_yticklabels([""]*len(g.axes[0, 2].get_yticklabels()))
        for ax in g.axes[0]:
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
        [ax.set_title("") for ax in g.axes[0]]
        xmin, xmax = g.axes[0, 2].get_xlim()
        g.axes[0, 2].set_xlim(xmin - 0.2, xmax + 0.2)  # add a bit more spacing between the groups
        #scale = 2.7
        #g.figure.set_size_inches(scale*3, scale*1)
        #g.axes[0,0].legend()
        #g.axes[0,2].legend(ncol=2, title="animals")
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/plot_CA1_v_CA3.pdf")
        plt.savefig(f"{self.output_root}/plot_CA1_v_CA3.svg")
        plt.close()

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
    parser.add_argument("-dp", "--data-path", required=True)
    parser.add_argument("-op", "--output-path", default=os.getcwd())
    parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
    args = parser.parse_args()

    area = args.area
    data_path = args.data_path
    output_path = args.output_path
    extra_info = args.extra_info

    # run analysis
    behav_stats = BehaviorStatistics(area, data_path, output_path, extra_info)
    behav_stats.calc_behavior_score()
    behav_stats.plot_behavioral_metrics_for_each_animal()
    behav_stats.plot_behavioral_metrics_for_all_sessions()
    #behav_stats.plot_correlation_matrix()
    #behav_stats.export_to_excel()

    extra_info_CA1 = ""
    extra_info_CA3 = ""
    behav_stats.plot_CA1_v_CA3(extra_info_CA1, extra_info_CA3)
