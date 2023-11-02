import os
import pandas
import numpy as np
import openpyxl
import logging
from ImageAnal import *
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import argparse
import seaborn as sns


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
    "CA1": ['srb131_211019'], # reshuffles
    "CA3": []
}

CORRIDORS = [14, 15,  # random
             16, 18,  # block
             17]  # new environment


def grow_df(df_a, df_b):
    if df_a is None:
        df_a = df_b
    else:
        df_a = pandas.concat((df_a, df_b))
    return df_a


class BehaviorAnalysis():
    def __init__(self, area, generate_plots=False, extra_info=""):
        self.area = area
        self.animals = ANIMALS[area]
        self.sessions_to_ignore = SESSIONS_TO_IGNORE[area]
        self.meta_xlsx = f"{area}_meta.xlsx"
        self.generate_plots = generate_plots  # if True, generates ImageAnal plots for each session

        # set output folder
        self.output_root = "behavioral_analysis"
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

        # set up logging
        self.date = datetime.today().strftime("%y%m%d")
        logging_format = "%(asctime)s [%(levelname)s] %(message)s"
        logging.basicConfig(filename=f"{self.output_root}/behavlog_{self.date}{self.extra_info}.log",
                            level=logging.INFO, format=logging_format)

        ### main dataframe for collecting statistics; to be exported to excel
        self.behavior_df = None

        # dict of behavioral and session properties, updated for each session (~dummy variable)
        self.behavior_dict_sess = {}

    def _read_meta(self, animal):
        logging.info(f"reading meta file at data/{self.meta_xlsx}")
        wb = openpyxl.load_workbook(filename=f"data/{self.meta_xlsx}")
        ws = wb.worksheets[0]
        sessions = {}
        for row in ws.rows:
            animal_name = row[2].value
            session_name = row[4].value
            if session_name is None:
                continue
            if animal == animal_name and session_name not in sessions:
                sessions[session_name] = {
                    "date_time": row[1].value,
                    "name": animal_name,
                    "task": row[3].value,
                    "suite2p_folder": session_name,
                    "imaging_logfile_name": row[5].value,
                    "TRIGGER_VOLTAGE_FILENAME": row[6].value,
                }
        return sessions

    def _load_session(self, sessions_all, current_session):
        if current_session in SESSIONS_TO_IGNORE[self.area]:
            logging.info(f"{current_session} part of SESSIONS_TO_IGNORE list; skipping")
            return None

        # set up ImagingSessionData parameters
        date_time = sessions_all[current_session]["date_time"]
        name = sessions_all[current_session]["name"]
        task = sessions_all[current_session]["task"]
        datapath = os.getcwd() + '/'
        suite2p_folder = datapath + 'data/' + f"{name}_imaging/" + sessions_all[current_session]["suite2p_folder"] + "/"
        imaging_logfile_name = suite2p_folder + sessions_all[current_session]["imaging_logfile_name"]
        TRIGGER_VOLTAGE_FILENAME = suite2p_folder + sessions_all[current_session]["TRIGGER_VOLTAGE_FILENAME"]

        # check if data folder exists
        if not os.path.exists(suite2p_folder):
            logging.warning(f"failed to find suite2p folder for session {current_session} at path {suite2p_folder}; skipping")
            return None

        logging.info(f"creating ImagingSessionData with the following parametrization: {datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, current_session}")
        ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=current_session)
        ISD.sessionID = current_session
        return ISD

    def _find_lap_numbers(self, ISD):
        for corridor in CORRIDORS:
            query_result = np.where(ISD.i_corridors[ISD.i_Laps_ImData] == corridor)
            if query_result:
                N_corridor = len(query_result[0])
            else:
                N_corridor = ""
            self.behavior_dict_sess[f"#laps({corridor})"] = N_corridor

    def _determine_experimental_protocol(self, ISD):
        if 17 in ISD.corridors:
            self.behavior_dict_sess["protocol"] = "new_env"
        elif 16 in ISD.corridors or 18 in ISD.corridors:
            self.behavior_dict_sess["protocol"] = "block"
        else:
            self.behavior_dict_sess["protocol"] = "random"

    def _find_Pcorrect(self, ISD):
        for corridor in CORRIDORS:
            if corridor in ISD.Ps_correct:
                self.behavior_dict_sess[f"Pcorrect({corridor})"] = np.round(ISD.Ps_correct[corridor], 2)
            else:
                self.behavior_dict_sess[f"Pcorrect({corridor})"] = ""

    def _find_speed_selectivities(self, ISD):
        # within corridor
        for corridor in CORRIDORS:
            if corridor in ISD.speed_selectivity_laps:
                self.behavior_dict_sess[f"Vsel({corridor})"] = np.round(np.nanmean(ISD.speed_selectivity_laps[corridor]), 2)
            else:
                self.behavior_dict_sess[f"Vsel({corridor})"] = ""

        # across corridors
        self.behavior_dict_sess[f"Vsel(X-corr)"] = np.round(ISD.speed_selectivity_cross_corridor, 2)

    def _find_lick_selectivities(self, ISD):
        for corridor in CORRIDORS:
            if corridor in ISD.lick_selectivity_laps:
                self.behavior_dict_sess[f"Lsel({corridor})"] = np.round(np.nanmean(ISD.lick_selectivity_laps[corridor]), 2)
            else:
                self.behavior_dict_sess[f"Lsel({corridor})"] = ""

        self.behavior_dict_sess[f"Lsel(X-corr)"] = np.round(ISD.lick_selectivity_cross_corridor, 2)

    def _plot_sessions(self, animal):
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

    def _plot_correlation_matrix(self):
        y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Vsel(14)", "Vsel(15)", "Vsel(X-corr)", "Lsel(14)", "Lsel(15)", "Lsel(X-corr)"]
        behavior_df_selcols = self.behavior_df[y_cols].abs()  # we take absolute value, since selectivity in either direction is fine
        corr_matrix = behavior_df_selcols.corr()
        plt.figure()
        sns.heatmap(corr_matrix, annot=True)
        plt.tight_layout()
        plt.savefig(f"{self.output_root}/corr_matrix.pdf")
        plt.close()

    def _plot_behavioral_metrics(self):
        plt.figure()
        y_cols = ["Pcorrect(14)", "Pcorrect(15)", "Vsel(14)", "Vsel(15)", "Vsel(X-corr)", "Lsel(14)", "Lsel(15)", "Lsel(X-corr)"]
        behavior_df_selcols = self.behavior_df[y_cols].abs()  # we take absolute value, since selectivity in either direction is fine

        # plot all sessions
        ax = sns.boxplot(data=behavior_df_selcols)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        sns.stripplot(data=behavior_df_selcols, ax=ax, color="black")
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[len(y_cols):], labels[len(y_cols):], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()

        # plot disengaged sessions
        disengaged_sessions = ["srb251_221027_T1", "srb251_221027_T2"]
        de_sess_df = self.behavior_df[self.behavior_df["sessionID"].isin(disengaged_sessions)]
        de_sess_df_selcols = de_sess_df[y_cols].abs()
        sns.stripplot(data=de_sess_df_selcols, ax=ax, color="red")

        plt.savefig(f"{self.output_root}/behavioral_measures.pdf")
        plt.close()

    def _export_to_excel(self):
        with pd.ExcelWriter(f"{self.output_root}/behavior_{self.area}{self.extra_info}.xlsx") as xlsx_writer:
            self.behavior_df.to_excel(xlsx_writer, index=False)

    def analyze_behavior(self):
        sessions_meta_df = None
        for animal in self.animals:
            logging.info(f"running analysis for animal {animal}")
            sessions_all = self._read_meta(animal)
            for current_session in sessions_all:
                ISD = self._load_session(sessions_all, current_session)
 
                # if failed to load ISD, skip
                if ISD == None:
                    continue

                # load shuffle data
                try:
                    logging.info("loading/calculating shuffle data")
                    ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=12)
                except Exception:
                    logging.exception(f"loading/calculating shuffle data failed for session {current_session}; skipping")
                    continue

                avgspeed = np.round(np.nanmean([np.nanmean(lap.ave_speed) for lap in ISD.ImLaps]), 2)  # belül: bineken át vett, kívül: lap-eken át vett átlag
                maxspeed = np.round(np.nanmean([np.nanmax(lap.ave_speed) for lap in ISD.ImLaps]), 2)

                mean_SNR = np.round(np.nanmean(ISD.cell_SNR), 2)
                std_SNR = np.round(np.nanstd(ISD.cell_SNR), 2)

                self.behavior_dict_sess = {
                    "area": area,
                    "animalID": animal,
                    "sessionID": current_session,
                    "Vmax [cm/s]": maxspeed,
                    "Vavg [cm/s]": avgspeed,
                    "mean SNR": mean_SNR,
                    "std SNR": std_SNR,
                    "#cells": ISD.N_cells,
                    "#activeCs": len(ISD.active_cells),
                    "#tunedCs": len(functools.reduce(np.union1d, ISD.tuned_cells)),
                    "#corrs": ISD.N_corridors,
                    "corrIDs": ISD.corridors,
                    "#laps(total)": ISD.N_ImLaps,
                }
                self._determine_experimental_protocol(ISD)
                self._find_lap_numbers(ISD)
                self._find_Pcorrect(ISD)
                self._find_speed_selectivities(ISD)
                self._find_lick_selectivities(ISD)

                if self.generate_plots:
                    # plot selectivity histograms
                    if not os.path.exists(f"{self.output_root}/selectivities"):
                        os.makedirs(f"{self.output_root}/selectivities")
                    ISD.plot_speed_and_lick_selectivity(f"{self.output_root}/selectivities/{current_session}_selectivity.pdf")

                    # plot lap-by-lap behavior
                    if not os.path.exists(f"{self.output_root}/behavior_lap_by_lap"):
                        os.makedirs(f"{self.output_root}/behavior_lap_by_lap")
                    save_path = f"{self.output_root}/behavior_lap_by_lap/{current_session}_lap_by_lap.pdf"
                    ISD.plot_session(average=False, filename=save_path, selected_laps=ISD.i_Laps_ImData)

                    # plot dF/F for problematic (high) tau sessions
                    if current_session in ["srb251_221027_T1", "srb251_221027_T2"]:
                        if not os.path.exists(f"{self.output_root}/dF_F"):
                            os.makedirs(f"{self.output_root}/dF_F")
                        for cellid in ISD.tuned_cells[0][0:3]:
                            save_path = f"{self.output_root}/dF_F/dF_F_{current_session}_cell_{cellid}.pdf"
                            ISD.plot_cell_laps(cellid, multipdf_object=save_path, write_pdf=True)

                behavior_df_sess = pd.DataFrame.from_dict([self.behavior_dict_sess])
                self.behavior_df = grow_df(self.behavior_df, behavior_df_sess)

            # sort sessions alphabetically (~ by date)
            self.behavior_df = self.behavior_df.sort_values(by=["area", "animalID", "sessionID"])
            self._plot_sessions(animal)

        # sort sessions alphabetically (~ by date)
        self.behavior_df = self.behavior_df.sort_values(by=["area", "animalID", "sessionID"])
        self._export_to_excel()
        self._plot_correlation_matrix()
        self._plot_behavioral_metrics()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
parser.add_argument("-g", "--generate-plots", type=bool)
parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
args = parser.parse_args()

area = args.area
generate_plots = args.generate_plots
extra_info = args.extra_info

# run analysis
analysis = BehaviorAnalysis(area, generate_plots, extra_info)
analysis.analyze_behavior()
