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

#############
area = "CA1"
create_ratemap_plots = False
shift_criterion = True
extra = "_calcSGforEarly"

poster = False
selected_cells = [
    ["CA1", "KS028", "KS028_103121", 48, 15],
    ["CA1", "KS028", "KS028_110121", 19, 15],
    ["CA1", "KS028", "KS028_103121", 387, 14],
    ["CA1", "KS028", "KS028_103121", 123, 14],
    ["CA1", "KS028", "KS028_103121", 142, 15],
    ["CA3", "srb231", "srb231_220808", 16, 14],
    ["CA3", "srb231", "srb231_220808", 26, 14],
    ["CA3", "srb231", "srb231_220808", 33, 14],
    ["CA3", "srb231", "srb231_220809_002", 53, 15],
    ["CA3", "srb231", "srb231_220808", 1, 14],
]
#i = 4
selected_animal = False #selected_cells[i][1]
selected_session = False #selected_cells[i][2]
selected_cell = False #selected_cells[i][3]
selected_corridor = False #selected_cells[i][4]

write_stats_csv = True
write_placefield_categories_csv = True
create_correlations_plots = False
create_shift_histograms = False
create_shift_gain_scatterplots = False
create_maxrates_plots = False
create_shiftscores_plots = False
#############

animals_CA1 = ["KS028",
               "KS029",
               "KS030",
               "srb131"]
animals_CA3 = ["srb231",
               "srb251",
               "srb269",
               "srb270"]
sessions_to_ignore_CA1 = [#'KS028_110521',  # error
                          'KS029_110321',  # no p95
                          'KS029_110721',  # error
                          'KS029_110821',  # error
                          'KS029_110521',  # error
                          'KS030_110721',  # error
                          'srb131_211019'] # reshuffles
sessions_to_ignore_CA3 = []

# configure run
if area == "CA1":
    animals = animals_CA1
    sessions_to_ignore = sessions_to_ignore_CA1
elif area == "CA3":
    animals = animals_CA3
    sessions_to_ignore = sessions_to_ignore_CA3
else:
    raise Exception("choose CA1 or CA3")
xlsx_file = f"{area}_meta.xlsx"
date = datetime.today().strftime("%y%m%d")
if not shift_criterion:
    extra += "_noshift"
output_root = f"BTSP_analysis_{area}_{date}{extra}"
if not os.path.exists(output_root):
    os.makedirs(output_root)
logging_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(filename=f"{output_root}/BTSP_analysis.log", level=logging.INFO, format=logging_format)

def read_meta(xlsx_file, animal):
    logging.info(f"reading meta file at data/{xlsx_file}")
    wb = openpyxl.load_workbook(filename=f"data/{xlsx_file}")
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

def color_masks(suite2p_folder, D1):
    logging.info(f"creating color masks for session {D1.sessionID}")
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
    plt.savefig(suite2p_folder + 'FOV_with_masks_random.pdf')
    plt.close()

    plt.figure('tuning')
    plt.savefig(suite2p_folder + 'FOV_with_masks_tuning.pdf')
    plt.close()


place_fields_all_sessions = defaultdict(list)
sessions_meta_df = None
for animal in animals:
    if selected_animal:
        if animal != selected_animal:
            continue

    logging.info(f"running analysis for animal {animal}")
    # load session metadata
    sessions = read_meta(xlsx_file, animal)

    # create output folders
    logging.info(f"creating output folders")
    output_dir = f"{output_root}/{animal}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for session_name in sessions:
        if not os.path.exists(f"{output_dir}/{session_name}"):
            os.makedirs(f"{output_dir}/{session_name}")
            os.makedirs(f"{output_dir}/{session_name}/ratemaps")

    # analyze sessions
    first_run = True
    df_joined_animal = None
    cell_stats_df_joined_animal = None
    nmf_matrices_animal = {
        "all": None,
        "non_btsp": None,
        "newly": None,
        "candidate_btsp": None,
        "btsp": None
    }
    #first_session = True
    for session_name in sessions:
        #if not first_session:
        #    continue
        #first_session = False

        if selected_session:
            if selected_session != session_name:
                continue

        logging.info(f"running analysis for session {session_name}")
        if session_name in sessions_to_ignore:
            logging.info(f"{session_name} part of ignored session list; skipping")
            continue
        session_output_dir = f"{output_dir}/{session_name}"

        # create session
        date_time = sessions[session_name]["date_time"]
        name = sessions[session_name]["name"]
        task = sessions[session_name]["task"]
        datapath = os.getcwd() + '/'
        suite2p_folder = datapath + 'data/' + f"{name}_imaging/" + sessions[session_name]["suite2p_folder"] + "/"
        imaging_logfile_name = suite2p_folder + sessions[session_name]["imaging_logfile_name"]
        TRIGGER_VOLTAGE_FILENAME = suite2p_folder + sessions[session_name]["TRIGGER_VOLTAGE_FILENAME"]
        logging.info(f"creating ImagingSessionData with the following parametrization: {datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, session_name}")
        ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=session_name)
        ISD.sessionID = session_name
        ISD.plot_speed_and_lick_selectivity(f"{output_root}/{session_name}_selectivity.pdf")

        # load shuffle data
        try:
            logging.info("loading/calculating shuffle data")
            ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=12)
        except Exception:
            logging.exception(f"loading/calculating shuffle data failed for session {session_name}; skipping")
            continue

        # save cell number statistics to pandas DF
        cell_stats_df = ISD.save_cell_stats()
        if cell_stats_df_joined_animal is None:
            cell_stats_df_joined_animal = cell_stats_df
        else:
            cell_stats_df_joined_animal = pandas.concat((cell_stats_df_joined_animal, cell_stats_df))

        N_laps_corr3 = None
        if len(ISD.corridors) > 2:
            N_laps_corr3 = len(np.nonzero(ISD.i_corridors[ISD.i_Laps_ImData] == ISD.corridors[2])[0])
        avgspeed = np.nanmean([np.nanmean(lap.ave_speed) for lap in ISD.ImLaps])  # belül: bineken át vett, kívül: lap-eken át vett átlag
        maxspeed = np.nanmean([np.nanmax(lap.ave_speed) for lap in ISD.ImLaps])

        possible_corridors = [14, 15, # random
                              16, 18, # block
                              17] # new environment

        session_meta_dict = {
            "area": area,
            "animalID": animal,
            "sessionID": session_name,
            "maxspeed (cm/s)": maxspeed,
            "avgspeed (cm/s)": avgspeed,
            "SNR_mean": np.nanmean(ISD.cell_SNR),
            "SNR_std": np.nanstd(ISD.cell_SNR),
            "N_cells": ISD.N_cells,
            "N_active_cells": len(ISD.active_cells),
            "N_tuned_cells":  len(functools.reduce(np.union1d, ISD.tuned_cells)),
            "N_corridors": ISD.N_corridors,
            "corridors": ISD.corridors,
            "N_laps_total": ISD.N_ImLaps,
        }

        for corridor in possible_corridors:
            query_result = np.where(ISD.i_corridors[ISD.i_Laps_ImData] == corridor)
            if query_result:
                N_corridor = len(query_result[0])
            else:
                N_corridor = None
            session_meta_dict[f"N_laps_corr_{corridor}"] = N_corridor
        if 17 in ISD.corridors:
            session_meta_dict["protocol"] = "new_env"
        elif 16 in ISD.corridors or 18 in ISD.corridors:
            session_meta_dict["protocol"] = "block"
        else:
            session_meta_dict["protocol"] = "random"

        for corridor in possible_corridors:
            if corridor in ISD.Ps_correct:
                session_meta_dict[f"P_correct_corr_{corridor}"] = ISD.Ps_correct[corridor]
            else:
                session_meta_dict[f"P_correct_corr_{corridor}"] = "-"

        for corridor in possible_corridors:
            if corridor in ISD.speed_selectivity_laps:
                session_meta_dict[f"mean_speed_sel_corr_{corridor}"] = np.nanmean(ISD.speed_selectivity_laps[corridor])
            else:
                session_meta_dict[f"mean_speed_sel_corr_{corridor}"] = "-"

        for corridor in possible_corridors:
            if corridor in ISD.lick_selectivity_laps:
                session_meta_dict[f"mean_lick_sel_corr_{corridor}"] = np.nanmean(ISD.lick_selectivity_laps[corridor])
            else:
                session_meta_dict[f"mean_lick_sel_corr_{corridor}"] = "-"

        session_meta_dict[f"cross_corr_speed_sel"] = ISD.speed_selectivity_cross_corridor
        session_meta_dict[f"cross_corr_lick_sel"] = ISD.lick_selectivity_cross_corridor

        session_meta_df = pd.DataFrame.from_dict([session_meta_dict])
        if sessions_meta_df is None:
            sessions_meta_df = session_meta_df
        else:
            sessions_meta_df = pandas.concat((sessions_meta_df, session_meta_df))

        place_cells = np.union1d(ISD.tuned_cells[0], ISD.tuned_cells[1])  # TODO: handle third corridor
        if len(place_cells) == 0:
            logging.info(f"no tuned cells found in session {session_name}; skipping")
            continue

        # run BTSP analysis
        logging.info(f"running BTSP analysis for session {session_name}, create_ratemap_plots={create_ratemap_plots}")
        for cellid in place_cells:
            if selected_cell:
                if selected_cell != cellid:
                    continue
            ISD.run_btsp_analysis(cellid, shift_criterion)
            if create_ratemap_plots:
                if poster:
                    ISD.plot_cell_laps_poster(area, cellid, corridor_abs=selected_corridor)
                else:
                    #ISD.plot_cell_laps(cellid, signal="rate", multipdf_object="C:/home/phd/btsp_poster/pf_examples", write_pdf=True, plot_BTSP=True)
                    ISD.plot_cell_laps(cellid, signal="rate", multipdf_object=f"{session_output_dir}/ratemaps", write_pdf=True, plot_BTSP=True)
        if ISD.btsp_analysis is None:
            logging.warning(f"BTSP analysis could not be run for session {session_name}; skipping")
            continue

        # stack NMF matrices
        for nmf_type in nmf_matrices_animal:
            nmf_matrix = ISD.btsp_analysis.create_nmf_matrix(nmf_type)
            if nmf_matrix is not None:
                if nmf_matrices_animal[nmf_type] is None:
                    nmf_matrices_animal[nmf_type] = nmf_matrix
                else:
                    nmf_matrices_animal[nmf_type] = np.vstack((nmf_matrices_animal[nmf_type], nmf_matrix))

        try:
            if create_shift_histograms:
                logging.info(f"creating \"shift score histogram\" for session {session_name}")
                ISD.btsp_analysis.plot_BTSP_shift_by_lap_histograms(save_path=session_output_dir)
            if create_shift_gain_scatterplots:
                logging.info(f"creating \"shift-gain scatterplot\" for session {session_name}")
                ISD.btsp_analysis.plot_BTSP_shift_gain_by_lap_scatterplots(save_path=session_output_dir)
            if create_maxrates_plots:
                logging.info(f"creating \"maximum rates by lap\" plot for session {session_name}")
                ISD.btsp_analysis.plot_BTSP_maxrates_all_fields_from_formation(save_path=session_output_dir)
            if create_shiftscores_plots:
                logging.info(f"creating \"shift scores by lap plot\" for session {session_name}")
                ISD.btsp_analysis.plot_BTSP_shift_all_fields_from_formation(save_path=session_output_dir)
            if create_correlations_plots:
                logging.info(f"creating \"shift and drift \" plot for session {session_name}")
                ISD.btsp_analysis.plot_BTSP_shift_all_fields_sorted_by_correlation(save_path=session_output_dir)
        except Exception:
            logging.exception(f"Exception occurred for session {session_name} during plot creation")

        logging.info(f"saving place field statistics and category CSVs for {session_name}")
        ISD.btsp_analysis.save_BTSP_statistics(save_path=session_output_dir)
        ISD.btsp_analysis.save_BTSP_place_field_categories(save_path=session_output_dir)

        # add place fields to dataframe for later analysis
        logging.info(f"adding session {session_name} to dataframe")
        df = ISD.btsp_analysis.combine_place_field_dataframes()
        if df_joined_animal is None:
            df_joined_animal = df
        else:
            df_joined_animal = pandas.concat((df_joined_animal, df))

        # collect place fields into dict
        logging.info(f"adding session {session_name} to pool dict")
        place_fields_all_sessions["unreliable"] += ISD.btsp_analysis.unreliable_place_fields
        place_fields_all_sessions["early"] += ISD.btsp_analysis.early_place_fields
        place_fields_all_sessions["transient"] += ISD.btsp_analysis.transient_place_fields
        place_fields_all_sessions["cbtsp"] += ISD.btsp_analysis.candidate_btsp_place_fields
        place_fields_all_sessions["nonbtsp"] += ISD.btsp_analysis.nonbtsp_novel_place_fields
        place_fields_all_sessions["btsp"] += ISD.btsp_analysis.btsp_place_fields

        # create colored masks
        name = sessions[session_name]["name"]
        suite2p_folder = datapath + 'data/' + f"{name}_imaging/" + sessions[session_name]["suite2p_folder"] + "/"
        color_masks(suite2p_folder, ISD)
        del ISD

    # save place field database and cell stats for a particular animal
    df_save_path_animal = f"{output_dir}/{animal}_place_fields_df.pickle"
    logging.info(f"saving place field dataframe to {df_save_path_animal} for animal {animal}")
    df_joined_animal.to_pickle(df_save_path_animal)

    df_save_path_cells_animal = f"{output_dir}/{animal}_cell_stats_df.pickle"
    logging.info(f"saving cell stats dataframe to {df_save_path_cells_animal} for animal {animal}")
    cell_stats_df_joined_animal.to_pickle(df_save_path_cells_animal)

    # save NMF matrices
    for nmf_type in nmf_matrices_animal:
        with open(f"{output_dir}/{animal}_NMF_matrix_{nmf_type}.npy", "wb") as nmf_file:
            np.save(nmf_file, nmf_matrices_animal[nmf_type])

with pd.ExcelWriter(f"{output_root}/sessions_df_{area}.xlsx") as xlsx_writer:
    sessions_meta_df.to_excel(xlsx_writer)

# create plots for all sessions and animals pooled
logging.info("running analysis for all sessions and animals pooled")
btsp_analysis_pooled = BtspAnalysis(sessionID=None, cellid=None, rate_matrix=np.zeros((1,1)), place_field_bounds=[])
btsp_analysis_pooled.unreliable_place_fields = place_fields_all_sessions["unreliable"]
btsp_analysis_pooled.early_place_fields = place_fields_all_sessions["early"]
btsp_analysis_pooled.transient_place_fields = place_fields_all_sessions["transient"]
btsp_analysis_pooled.candidate_btsp_place_fields = place_fields_all_sessions["cbtsp"]
btsp_analysis_pooled.btsp_place_fields = place_fields_all_sessions["btsp"]
btsp_analysis_pooled.nonbtsp_novel_place_fields = place_fields_all_sessions["nonbtsp"]

if create_shift_histograms:
    logging.info(f"creating \"shift score histogram\" for all sessions and animals pooled")
    btsp_analysis_pooled.plot_BTSP_shift_by_lap_histograms(save_path=output_root)
if create_shift_gain_scatterplots:
    logging.info(f"creating \"shift-gain scatterplot\" for all sessions and animals pooled")
    btsp_analysis_pooled.plot_BTSP_shift_gain_by_lap_scatterplots(save_path=output_root)
if create_maxrates_plots:
    logging.info(f"creating \"maximum rates by lap\" plot for all sessions and animals pooled")
    btsp_analysis_pooled.plot_BTSP_maxrates_all_fields_from_formation(save_path=output_root)
if create_shiftscores_plots:
    logging.info(f"creating \"shift scores by lap plot\" for all sessions and animals pooled")
    btsp_analysis_pooled.plot_BTSP_shift_all_fields_from_formation(save_path=output_root)
if create_correlations_plots:
    logging.info(f"creating \"shift and drift \" plot for all sessions and animals pooled")
    btsp_analysis_pooled.plot_BTSP_shift_all_fields_sorted_by_correlation(save_path=output_root)
logging.info(f"saving place field statistics and category CSVs for all sessions and animals pooled")
btsp_analysis_pooled.save_BTSP_statistics(save_path=output_root)
