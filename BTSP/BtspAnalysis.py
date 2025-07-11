import configparser
import os
import pickle
import logging
import argparse
import numpy as np
from configparser import ConfigParser
from constants import ANIMALS, SESSIONS_TO_IGNORE
from BtspAnalysisSingleCell import BtspAnalysisSingleCell
from utils import makedir_if_needed, grow_df
import tqdm


class BtspAnalysis:
    def __init__(self, area, data_path, output_path, params_dict=None, extra_info=""):
        self.area = area
        self.animals = ANIMALS[area]
        self.data_path = data_path
        self.extra_info = "" if not extra_info else f"_{extra_info}"
        self.history_dependent = True if "historyDependent" in extra_info else False
        self.without_fbe = True if "withoutFBE" in extra_info else False  # FBE = forwards bound extension
        self.output_root = f"{output_path}/place_fields/{self.area}{self.extra_info}"
        self.reverse = True if "reverse" in self.extra_info else False
        makedir_if_needed(f"{output_path}/place_fields")
        makedir_if_needed(self.output_root)

        # load parameters of analysis
        # allows for overwriting certain keys if params_dict is provided
        self.params = self.read_params()
        if params_dict is not None:
            self.update_params(params_dict)

        # save run parameters
        with open(f"{self.output_root}/script_params.txt", "w") as params_file:
            params_file.write(f"--area = {self.area}\n"
                              f"--data-path = {self.data_path}\n"
                              f"--output-path = {output_path}\n"
                              f"--extra-info = {self.data_path}\n")

        # set up logging
        logging_format = "%(asctime)s [%(levelname)s] %(message)s"
        logging.basicConfig(filename=f"{self.output_root}/placefield{self.extra_info}.log",
                            level=logging.INFO, format=logging_format)
        logging.info(f"************* STARTING ANALYSIS *************")
        logging.info(f"PARAMETERS:\n\t\t\t* area={area}\n\t\t\t* data_path={data_path}"
                     f"\n\t\t\t* output_path={output_path}\n\t\t\t* extra_info={extra_info}")

    def read_params(self):
        config = ConfigParser(inline_comment_prefixes="#")
        config.read("parameters.ini")
        config_dict = dict(config["PARAMETERS"])
        params = {}
        for key, value in config_dict.items():
            value_cast = float(value) if "." in value else int(value)
            params[key.upper()] = value_cast
        return params

    def update_params(self, params_dict):
        for key, value in params_dict.items():
            self.params[key] = value

    def run_btsp_analysis(self):
        tcl_root = f"{self.data_path}\\tuned_cells\\{self.area}{self.extra_info}\\"
        tcl_paths = [file for file in os.listdir(tcl_root) if "tuned_cells" in file]
        for animal in tqdm.tqdm(self.animals):
            pfs_df_animal = None
            tcl_paths_animal = [path for path in tcl_paths if f"_{animal}_" in path]
            for path in tcl_paths_animal:
                with open(f"{tcl_root}\\{path}", "rb") as tcl_file:
                    tcl = pickle.load(tcl_file)

                basc_all = None
                for tuned_cell in tcl:
                    if tuned_cell.sessionID in SESSIONS_TO_IGNORE[self.area]:
                        continue

                    #tuned_cell.frames_pos_bins = []
                    #tuned_cell.frames_dF_F = []
                    rate_matrix = tuned_cell.rate_matrix
                    if self.reverse:
                        rate_matrix = np.flip(tuned_cell.rate_matrix, axis=1)

                    # throw away cell if session has fewer than 10 laps (due to history split)
                    if self.history_dependent and rate_matrix.shape[1] <= 15:
                        continue

                    basc = BtspAnalysisSingleCell(tuned_cell.sessionID, tuned_cell.cellid, rate_matrix,
                                                  tuned_cell.frames_pos_bins, tuned_cell.frames_dF_F, tuned_cell.lap_histories,
                                                  tuned_cell.spike_matrix, i_laps=tuned_cell.corridor_laps,
                                                  params=self.params, bins_p95_geq_afr=tuned_cell.bins_p95_geq_afr)
                    if self.history_dependent:
                        basc.categorize_place_fields(cor_index=tuned_cell.corridor, history=tuned_cell.history,
                                                     without_fbe=self.without_fbe)
                    else:
                        basc.categorize_place_fields(cor_index=tuned_cell.corridor,
                                                     without_fbe=self.without_fbe)

                    # TODO: ezt az operator overloadingot dobd ki
                    if basc_all is None:
                        basc_all = basc
                    else:
                        basc_all += basc
                if basc_all is None:
                    logging.warning(f"BTSP analysis could not be run for session at {path}; skipping")
                    continue

                # add place fields to dataframe for later analysis
                pfs_df = basc_all.combine_place_field_dataframes()
                pfs_df_animal = grow_df(pfs_df_animal, pfs_df)

            # save place field database and cell stats for a particular animal
            pfs_df_save_path_animal = f"{self.output_root}/{animal}_place_fields_df.pickle"
            logging.info(f"saving place field dataframe to {pfs_df_save_path_animal} for animal {animal}")
            try:
                pfs_df_animal.to_pickle(pfs_df_save_path_animal)
            except AttributeError:
                # happens when pfs_df_animal is None, bc there were no place fields
                logging.warning(f"no place field found for animal {animal}")

    def export_parameters(self, export_path=""):
        path = f"{self.output_root}/parameters.ini"
        if export_path:
            path = export_path
        config = configparser.ConfigParser()
        config["PARAMETERS"] = self.params
        with open(path, "w") as params_file:
            config.write(params_file)

if __name__ == "__main__":
    area = "CA3"
    data_path = f"C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual"
    output_path = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual"
    extra_info = "NFafter5Laps"

    #run analysis
    analysis = BtspAnalysis(area, data_path, output_path, extra_info=extra_info)
    analysis.run_btsp_analysis()
    # TODO: hiányzik még NMF, cell stats és ratemaps
