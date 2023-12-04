import os
import openpyxl
import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

from ImageAnal import *
from constants import ANIMALS, SESSIONS_TO_IGNORE, DISENGAGEMENT
from utils import makedir_if_needed


class TunedCell:
    def __init__(self, sessionID, cellid, rate_matrix, corridor, bins_p95_geq_afr):
        self.sessionID = sessionID
        self.cellid = cellid
        self.rate_matrix = rate_matrix
        self.corridor = corridor

        # list of spatial bins where tuning curve > 95th percentile of shuffled activity
        self.bins_p95_geq_afr = bins_p95_geq_afr


class TunedCellList:
    def __init__(self, area, data_path, output_path, extra_info="", shuffled_laps=False):
        self.area = area
        self.animals = ANIMALS[area]
        self.sessions_to_ignore = SESSIONS_TO_IGNORE[area]
        self.meta_xlsx = f"{area}_meta.xlsx"
        self.data_path = data_path

        # load json containing session-protocol pairs (to filter sessions by protocol later)
        with open("sessions.json") as session_protocol_file:
            self.session_protocol_dict = json.load(session_protocol_file)[self.area]

        # set output folder
        self.shuffled_laps = shuffled_laps
        if self.shuffled_laps:
            self.extra_info = "_shuffled_laps" if not extra_info else f"_{extra_info}_shuffled_laps"
        else:
            self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/tuned_cells/{self.area}{self.extra_info}"
        makedir_if_needed(f"{output_path}/tuned_cells")
        makedir_if_needed(self.output_root)

        # save run parameters
        with open(f"{self.output_root}/params.txt", "w") as params_file:
            params_file.write(f"--area = {self.area}\n"
                              f"--data-path = {self.data_path}\n"
                              f"--output-path = {output_path}\n"
                              f"--extra-info = {self.data_path}\n"
                              f"shuffled = {self.shuffled_laps}\n")

        # set up logging
        logging_format = "%(asctime)s [%(levelname)s] %(message)s"
        logging.basicConfig(filename=f"{self.output_root}/tunedcells{self.extra_info}.log",
                            level=logging.INFO, format=logging_format)
        logging.info(f"************* STARTING ANALYSIS *************")
        logging.info(f"PARAMETERS:\n\t\t\t* area={area}\n\t\t\t* data_path={data_path}\n\t\t\t* output_path={output_path}\n"
                                  f"\t\t\t* extra_info={extra_info}\n\t\t\t*")

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
            logging.info(f"{current_session} part of DISENGAGEMENT list; skipping")
            return None

        # set up ImagingSessionData parameters
        logging.info(f"running analysis for session {current_session}")
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
        ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=current_session)
        ISD.sessionID = current_session
        return ISD

    def create_tuned_cell_list(self):
        for animal in self.animals:
            logging.info(f"creating tuned cell list for animal {animal}")
            sessions_all = self._read_meta(animal)
            for current_session in sessions_all:
                ISD = self._load_session(sessions_all, current_session)
                if ISD == None:  # if failed to load ISD, skip
                    continue

                # load shuffle data
                try:
                    logging.info("loading/calculating shuffle data")
                    ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=12)
                except Exception:
                    logging.exception(f"loading/calculating shuffle data failed for session {current_session}; skipping")
                    continue

                tuned_cell_list = ISD.prepare_btsp_analysis(shuffled_laps=self.shuffled_laps)
                with open(f"{self.output_root}/tuned_cells_{current_session}.pickle", "wb") as tuned_cells_file:
                    pickle.dump(tuned_cell_list, tuned_cells_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
    parser.add_argument("-dp", "--data-path", required=True)
    parser.add_argument("-op", "--output-path", default=os.getcwd())
    parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
    args = parser.parse_args()

    shuffled_laps = True
    tcl = TunedCellList(args.area, args.data_path, args.output_path, extra_info=args.extra_info, shuffled_laps=shuffled_laps)
    tcl.create_tuned_cell_list()
