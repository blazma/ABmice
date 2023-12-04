import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from constants import ANIMALS, SESSIONS_TO_IGNORE
from utils import makedir_if_needed


class BehaviorStatistics:
    def __init__(self, area, data_path, output_path, extra_info):
        self.area = area
        self.animals = ANIMALS[area]
        self.sessions_to_ignore = SESSIONS_TO_IGNORE[area]
        self.meta_xlsx = f"{area}_meta.xlsx"
        self.data_path = data_path
        self.date = datetime.today().strftime("%y%m%d")

        # set output folder
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/behavioral_plots_{self.area}_{self.date}{self.extra_info}"
        makedir_if_needed(self.output_root)

        # load sessions data
        self.behavior_df = pd.read_pickle(f"{self.data_path}/behavior_df_{self.area}.pickle")

    def plot_behavioral_metrics_for_each_animal(self):
        for animal in self.animals:
            beh_df_animal = self.behavior_df[self.behavior_df["animalID"] == animal]
            y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Vsel(14)", "Vsel(15)", "Vsel(X-corr)", "Lsel(14)", "Lsel(15)", "Lsel(X-corr)"]
            beh_df_animal_selcols = beh_df_animal[["sessionID", "protocol", *y_cols]]
            beh_df_animal_selcols_melted = beh_df_animal_selcols.melt(["sessionID", "protocol"], var_name="cols", value_name="vals")

            plt.figure()
            ax = sns.lineplot(x="sessionID", y="vals", hue="cols", data=beh_df_animal_selcols_melted)
            sns.scatterplot(x="sessionID", style="protocol", y="vals", hue="cols", data=beh_df_animal_selcols_melted, ax=ax, s=60)
            handles, labels = ax.get_legend_handles_labels()
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

        y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Vsel(14)", "Vsel(15)", "Vsel(X-corr)", "Lsel(14)", "Lsel(15)", "Lsel(X-corr)"]
        #y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Vsel(14)", "Lsel(14)", "Vsel(15)", "Lsel(15)", "Vsel(X-corr)", "Lsel(X-corr)"]
        disengaged_sessions = ["srb251_221027_T1", "srb251_221027_T2"]

        # plot all sessions
        behavior_df_selcols = self.behavior_df[y_cols].abs()  # we take absolute value, since selectivity in either direction is fine
        behavior_df_selcols = norm(behavior_df_selcols)
        ax = sns.boxplot(data=behavior_df_selcols, showfliers=False, color="white")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # filter out disengaged sessions for scatterplot
        behavior_df_selcols_scatter = self.behavior_df[~self.behavior_df["sessionID"].isin(disengaged_sessions)][y_cols].abs()
        behavior_df_selcols_scatter = norm(behavior_df_selcols_scatter)
        #ax = sns.swarmplot(data=behavior_df_selcols_scatter, color="black")
        #handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[len(y_cols):], labels[len(y_cols):], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()

        # plot disengaged sessions
        #de_sess_df = self.behavior_df[self.behavior_df["sessionID"].isin(disengaged_sessions)]
        #de_sess_df_selcols = de_sess_df[y_cols].abs()
        #de_sess_df_selcols = norm(de_sess_df_selcols)
        #sns.stripplot(data=de_sess_df_selcols, ax=ax, color="red")

        import matplotlib as mpl
        def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
            c1 = np.array(mpl.colors.to_rgb(c1))
            c2 = np.array(mpl.colors.to_rgb(c2))
            return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
        c1 = 'red'  # blue
        c2 = 'blue'  # green
        behavior_df_selcols_scatter = behavior_df_selcols_scatter.sort_values(by=y_cols[0])
        for i in range(behavior_df_selcols_scatter.shape[0]):
            plt.plot(behavior_df_selcols_scatter.values[i, :], color=colorFader(c1,c2,i/behavior_df_selcols_scatter.shape[0]))

        #plt.savefig(f"{self.output_root}/behavioral_measures.pdf")
        plt.show()
        plt.close()

    def plot_correlation_matrix(self):
        y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Vsel(14)", "Vsel(15)", "Vsel(X-corr)", "Lsel(14)", "Lsel(15)", "Lsel(X-corr)"]
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
    behav_stats.plot_behavioral_metrics_for_each_animal()
    behav_stats.plot_behavioral_metrics_for_all_sessions()
    behav_stats.plot_correlation_matrix()
    behav_stats.export_to_excel()
