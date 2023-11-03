import os
from copy import deepcopy
import pandas
import numpy as np
import pickle
import openpyxl
import logging
from ImageAnal import *
from datetime import datetime
from collections import defaultdict, OrderedDict
import warnings
warnings.filterwarnings("ignore")
import argparse

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


def grow_df(df_a, df_b):
    if df_a is None:
        df_a = df_b
    else:
        df_a = pandas.concat((df_a, df_b))
    return df_a


def makedir_if_needed(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class BtspAnalysis:
    def __init__(self, area, data_path, output_path, extra_info="",
                 is_shift_criterion_on=True, is_create_ratemaps=False, is_masks=False, is_poster=False):
        self.area = area
        self.animals = ANIMALS[area]
        self.sessions_to_ignore = SESSIONS_TO_IGNORE[area]
        self.meta_xlsx = f"{area}_meta.xlsx"
        self.data_path = data_path
        self.date = datetime.today().strftime("%y%m%d")

        # set output folder
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.output_root = f"{output_path}/BTSP_analysis_{self.area}_{self.date}{self.extra_info}"
        makedir_if_needed(self.output_root)

        # read extra arguments
        self.is_shift_criterion_on = is_shift_criterion_on
        self.is_create_ratemaps = is_create_ratemaps
        self.is_masks = is_masks
        self.is_poster = is_poster

        # set up logging
        logging_format = "%(asctime)s [%(levelname)s] %(message)s"
        logging.basicConfig(filename=f"{self.output_root}/btsplog_{self.date}{self.extra_info}.log",
                            level=logging.INFO, format=logging_format)
        logging.info(f"************* STARTING ANALYSIS *************")
        logging.info(f"PARAMETERS:\n\t\t\t* area={area}\n\t\t\t* data_path={data_path}\n\t\t\t* output_path={output_path}\n"
                                  f"\t\t\t* extra_info={extra_info}\n\t\t\t* shift={self.is_shift_criterion_on}\n"
                                  f"\t\t\t* masks={self.is_masks}\n\t\t\t* ratemaps={self.is_create_ratemaps}\n"
                                  f"\t\t\t* poster={self.is_poster}")

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
        ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=current_session)
        ISD.sessionID = current_session
        return ISD

    def _color_masks(self, suite2p_folder, D1):
        logging.info(f"creating color masks for session {D1.sessionID}")
        makedir_if_needed(f"{self.output_root}/masks")

        ops = np.load(D1.ops_string, allow_pickle=True).item()
        stat = np.load(D1.stat_string, allow_pickle=True)

        cellids_btsp = list(set([pf.cellid for pf in D1.btsp_analysis.btsp_place_fields]))
        cellids_tuned = np.union1d(D1.tuned_cells[0], D1.tuned_cells[1])
        cellids_act = D1.active_cells
        cellids = np.arange(D1.N_cells)

        print(cellids_act.size, cellids_tuned.size)
        im = np.ones((ops['Ly'], ops['Lx']))
        im[:] = np.nan
        im_tuning = np.ones((ops['Ly'], ops['Lx']))
        im_tuning[:] = np.nan

        # x_center = []
        # y_center = []
        # will have random coloring
        plt.figure('random')
        plt.matshow(np.log(ops['meanImg']), cmap='gray', fignum=False)
        # will color according to whether cell is active tuned or just "there"
        plt.figure('tuning')
        plt.matshow(np.log(ops['meanImg']), cmap='gray', fignum=False)
        colors = np.linspace(0., 1, cellids.size)
        for i in range(np.size(cellids)):

            im[:] = np.nan
            im_tuning[:] = np.nan

            cellid = cellids[i]
            n = D1.neuron_index[cellid]  # n is the suite2p ID for the given cell
            # print(cellids[i], n)
            ypix = stat[n]['ypix']  # [~stat[n]['overlap']]
            xpix = stat[n]['xpix']  # [~stat[n]['overlap']]
            # x_center.append(stat[n]['med'][1])
            # y_center.append(stat[n]['med'][0])

            # random coloring
            im[ypix, xpix] = colors[i]
            plt.figure('random')
            plt.matshow(im, fignum=False, alpha=0.5, cmap='gist_rainbow', vmin=0, vmax=1)

            # color according to whether cell is active tuned or just "there"
            plt.figure('tuning')
            if cellid in cellids_btsp:
                im_tuning[ypix, xpix] = 0.15
                plt.matshow(im_tuning, fignum=False, alpha=0.5, cmap='hsv', vmin=0, vmax=1)

            elif cellid in cellids_tuned:
                im_tuning[ypix, xpix] = 0
                plt.matshow(im_tuning, fignum=False, alpha=0.5, cmap='hsv', vmin=0, vmax=1)

            elif cellid in cellids_act:
                im_tuning[ypix, xpix] = 0.35
                plt.matshow(im_tuning, fignum=False, alpha=0.6, cmap='hsv', vmin=0, vmax=1)

            else:
                im_tuning[ypix, xpix] = 0.6
                plt.matshow(im_tuning, fignum=False, alpha=0.5, cmap='hsv', vmin=0, vmax=1)

        plt.figure('random')
        plt.savefig(f"{self.output_root}/masks/{D1.sessionID}_random.pdf")
        plt.close()

        plt.figure('tuning')
        plt.savefig(f"{self.output_root}/masks/{D1.sessionID}_tuning.pdf")
        plt.close()

    def run_btsp_analysis(self):
        for animal in self.animals:
            logging.info(f"running analysis for animal {animal}")
            sessions_all = self._read_meta(animal)

            df_joined_animal = None
            cell_stats_df_joined_animal = None
            nmf_matrices_animal = {
                "all": None,
                "non_btsp": None,
                "newly": None,
                "candidate_btsp": None,
                "btsp": None
            }
            for current_session in sessions_all:
                logging.info(f"running analysis for session {current_session}")
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

                # save cell number statistics to pandas DF
                cell_stats_df = ISD.save_cell_stats()
                cell_stats_df_joined_animal = grow_df(cell_stats_df_joined_animal, cell_stats_df)

                tuned_cells = np.union1d(ISD.tuned_cells[0], ISD.tuned_cells[1])
                if len(tuned_cells) == 0:
                    logging.info(f"no tuned cells found in session {current_session}; skipping")
                    continue

                # run BTSP analysis
                logging.info(f"running BTSP analysis for session {current_session}, create_ratemap_plots={self.is_create_ratemaps}")
                for cellid in tuned_cells:
                    ISD.run_btsp_analysis(cellid, self.is_shift_criterion_on)
                    if self.is_create_ratemaps:
                        makedir_if_needed(f"{self.output_root}/ratemaps")
                        makedir_if_needed(f"{self.output_root}/ratemaps/{current_session}")
                        if self.is_poster:
                            pass  # TODO: make specific sessions, cells, corridors selectable
                            #ISD.plot_cell_laps_poster(area, cellid, corridor_abs=selected_corridor)
                        else:
                            ratemap_dir = f"{self.output_root}/ratemaps/{current_session}"
                            ISD.plot_cell_laps(cellid, signal="rate", multipdf_object=ratemap_dir,
                                               write_pdf=True, plot_BTSP=True)
                if ISD.btsp_analysis is None:
                    logging.warning(f"BTSP analysis could not be run for session {current_session}; skipping")
                    continue

                # stack NMF matrices
                for nmf_type in nmf_matrices_animal:
                    nmf_matrix = ISD.btsp_analysis.create_nmf_matrix(nmf_type)
                    if nmf_matrix is not None:
                        if nmf_matrices_animal[nmf_type] is None:
                            nmf_matrices_animal[nmf_type] = nmf_matrix
                        else:
                            nmf_matrices_animal[nmf_type] = np.vstack((nmf_matrices_animal[nmf_type], nmf_matrix))

                # add place fields to dataframe for later analysis
                logging.info(f"adding session {current_session} to dataframe")
                df = ISD.btsp_analysis.combine_place_field_dataframes()
                df_joined_animal = grow_df(df_joined_animal, df)

                # create colored masks
                if self.is_masks:
                    suite2p_folder = f"{self.data_path}/data/{animal}_imaging/{sessions_all[current_session]["suite2p_folder"]}/"
                    self._color_masks(suite2p_folder, ISD)
                del ISD

            # save place field database and cell stats for a particular animal
            df_save_path_animal = f"{self.output_root}/{animal}_place_fields_df.pickle"
            logging.info(f"saving place field dataframe to {df_save_path_animal} for animal {animal}")
            df_joined_animal.to_pickle(df_save_path_animal)

            df_save_path_cells_animal = f"{self.output_root}/{animal}_cell_stats_df.pickle"
            logging.info(f"saving cell stats dataframe to {df_save_path_cells_animal} for animal {animal}")
            cell_stats_df_joined_animal.to_pickle(df_save_path_cells_animal)

            # save NMF matrices
            for nmf_type in nmf_matrices_animal:
                with open(f"{self.output_root}/{animal}_NMF_matrix_{nmf_type}.npy", "wb") as nmf_file:
                    np.save(nmf_file, nmf_matrices_animal[nmf_type])


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--area", required=True, choices=["CA1", "CA3"])
parser.add_argument("-dp", "--data-path", required=True)
parser.add_argument("-op", "--output-path", default=os.getcwd())
parser.add_argument("-x", "--extra-info")  # don't provide _ in the beginning
parser.add_argument("--shift", type=bool)
parser.add_argument("--ratemaps", type=bool)
parser.add_argument("--masks", type=bool)
parser.add_argument("--poster", type=bool)
args = parser.parse_args()

# run analysis
analysis = BtspAnalysis(args.area, args.data_path, args.output_path, extra_info=args.extra_info, is_masks=args.masks,
                        is_shift_criterion_on=args.shift, is_create_ratemaps=args.ratemaps, is_poster=args.poster)
analysis.run_btsp_analysis()
