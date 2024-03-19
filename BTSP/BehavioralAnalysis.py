import os
import json
import numpy as np
import openpyxl
import logging
from datetime import datetime
from matplotlib import pyplot as plt
import argparse
import seaborn as sns
from ImageAnal import *
from constants import ANIMALS, SESSIONS_TO_IGNORE, CORRIDORS, DISENGAGEMENT
from utils import grow_df, truncate_dict, makedir_if_needed
import warnings

warnings.filterwarnings("ignore")


class BehaviorAnalysis:
    def __init__(self, area, data_path, output_path, extra_info="", generate_all_plots=False):
        self.area = area
        self.animals = ANIMALS[area]
        self.sessions_to_ignore = SESSIONS_TO_IGNORE[area]
        self.meta_xlsx = f"{area}_meta.xlsx"
        self.data_path = data_path
        self.generate_all_plots = generate_all_plots

        # load json containing session-protocol pairs (to filter sessions by protocol later)
        with open("sessions.json") as session_protocol_file:
            self.session_protocol_dict = json.load(session_protocol_file)[self.area]

        # set output folder
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/behavior/{self.area}{self.extra_info}"
        makedir_if_needed(self.output_root)

        # set up logging
        logging_format = "%(asctime)s [%(levelname)s] %(message)s"
        logging.basicConfig(filename=f"{self.output_root}/behavlog{self.extra_info}.log",
                            level=logging.INFO, format=logging_format)

        ### main dataframe for collecting statistics; to be exported to excel
        self.behavior_df = None

        # dict of behavioral and session properties, updated for each session (~dummy variable)
        self.behavior_dict_sess = {}

    def _read_meta(self, animal):
        logging.info(f"reading meta file at data/{self.meta_xlsx}")
        wb = openpyxl.load_workbook(filename=f"{self.data_path}/data/{self.meta_xlsx}")
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
        if self.session_protocol_dict[current_session] != "random":
            logging.info(f"{current_session} has protocol {self.session_protocol_dict[current_session]}; skipping")
            return None
        if current_session in DISENGAGEMENT[self.area]:
            if DISENGAGEMENT[self.area][current_session] == [0, -1]:  # i.e. DE from start to end
                logging.info(f"{current_session} shows disengagement throughout; skipping")
                return None

        # set up ImagingSessionData parameters
        date_time = sessions_all[current_session]["date_time"]
        name = sessions_all[current_session]["name"]
        task = sessions_all[current_session]["task"]
        datapath = f"{self.data_path}/"
        suite2p_folder = f"{datapath}/data/{name}_imaging/{sessions_all[current_session]["suite2p_folder"]}/"
        imaging_logfile_name = suite2p_folder + sessions_all[current_session]["imaging_logfile_name"]
        TRIGGER_VOLTAGE_FILENAME = suite2p_folder + sessions_all[current_session]["TRIGGER_VOLTAGE_FILENAME"]

        # check if data folder exists
        if not os.path.exists(suite2p_folder):
            logging.warning(f"failed to find suite2p folder for session {current_session} at path {suite2p_folder}; skipping")
            return None

        logging.info(f"creating ImagingSessionData with the following parametrization: {datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, current_session}")
        ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name,
                                 TRIGGER_VOLTAGE_FILENAME, sessionID=current_session)
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
                    logging.exception(
                        f"loading/calculating shuffle data failed for session {current_session}; skipping")
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

                if self.generate_all_plots:
                    # plot selectivity histograms
                    makedir_if_needed(f"{self.output_root}/selectivities")
                    ISD.plot_speed_and_lick_selectivity(f"{self.output_root}/selectivities/{current_session}_selectivity.pdf")

                    # plot lap-by-lap behavior
                    makedir_if_needed(f"{self.output_root}/behavior_lap_by_lap")
                    save_path = f"{self.output_root}/behavior_lap_by_lap/{current_session}_lap_by_lap.pdf"
                    ISD.plot_session(average=False, filename=save_path, selected_laps=ISD.i_Laps_ImData)

                    # plot dF/F for problematic (high) tau sessions
                    #if current_session in ["srb251_221027_T1", "srb251_221027_T2"]:
                    #    makedir_if_needed(f"{self.output_root}/dF_F")
                    #    for cellid in ISD.tuned_cells[0][0:3]:
                    #        save_path = f"{self.output_root}/dF_F/dF_F_{current_session}_cell_{cellid}.pdf"
                    #        ISD.plot_cell_laps(cellid, multipdf_object=save_path, write_pdf=True)

                behavior_df_sess = pd.DataFrame.from_dict([self.behavior_dict_sess])
                self.behavior_df = grow_df(self.behavior_df, behavior_df_sess)

            # sort sessions alphabetically (~ by date)
            self.behavior_df = self.behavior_df.sort_values(by=["area", "animalID", "sessionID"])

        # sort sessions alphabetically (~ by date)
        self.behavior_df = self.behavior_df.sort_values(by=["area", "animalID", "sessionID"])
        self.behavior_df.to_pickle(f"{self.output_root}/behavior_df_{area}.pickle")


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

    generate_all_plots = True

    # run analysis
    analysis = BehaviorAnalysis(area, data_path, output_path, extra_info, generate_all_plots)
    analysis.analyze_behavior()
