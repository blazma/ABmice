import os
from copy import deepcopy
import openpyxl
from ImageAnal import *


def read_meta(xlsx_file, session_names):
    wb = openpyxl.load_workbook(filename=f"data/{xlsx_file}")
    ws = wb.worksheets[0]
    sessions = {}
    for row in ws.rows:
        session_name = row[4].value
        if session_name in session_names:
            sessions[session_name] = {
                "date_time": row[1].value,
                "name": row[2].value,
                "task": row[3].value,
                "suite2p_folder": session_name,
                "imaging_logfile_name": row[5].value,
                "TRIGGER_VOLTAGE_FILENAME": row[6].value,
            }
    return sessions

# read session metadata for selected sessions
session_names = ['KS028_103121',
                 'KS028_110121',
                 'KS028_110221',
                 'KS028_110321',
                 'KS028_110421',]
                 #'KS028_110521',
                 #'KS028_110621',
                 #'KS028_110721',
                 #'KS028_110921']
xlsx_file = "CA1_meta.xlsx"
sessions = read_meta(xlsx_file, session_names)

# create data objects for each session
ISDs = {}
datapath = os.getcwd() + '/'
for session_name in sessions:
    date_time = sessions[session_name]["date_time"]
    name = sessions[session_name]["name"]
    task = sessions[session_name]["task"]
    suite2p_folder = datapath + 'data/' + f"{name}_imaging/" + sessions[session_name]["suite2p_folder"] + "/"
    imaging_logfile_name = suite2p_folder + sessions[session_name]["imaging_logfile_name"]
    TRIGGER_VOLTAGE_FILENAME = suite2p_folder + sessions[session_name]["TRIGGER_VOLTAGE_FILENAME"]
    ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=session_name)
    ISDs[session_name] = ISD

# create output folders
output_dir = "BTSP_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for session_name in sessions:
    if not os.path.exists(f"{output_dir}/{session_name}"):
        os.makedirs(f"{output_dir}/{session_name}")
        os.makedirs(f"{output_dir}/{session_name}/ratemaps")

# combine place fields from all the sessions into a dummy session
dummy_ISD = deepcopy(list(ISDs.items())[0][1])

# run BTSP search
for session_name in ISDs:
     ISD = ISDs[session_name]
     session_output_dir = f"{output_dir}/{session_name}"

     ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=10)
     place_cells = np.union1d(ISD.accepted_PCs[0], ISD.accepted_PCs[1])
     for cellid in place_cells:
         ISD.find_BTSP(cellid)
         ISD.plot_cell_laps(cellid, signal="rate", multipdf_object=f"{session_output_dir}/ratemaps", write_pdf=True, plot_BTSP=True)

     ISD.plot_BTSP_shift_by_lap_histograms(save_path=session_output_dir)
     ISD.plot_BTSP_shift_gain_by_lap_scatterplots(save_path=session_output_dir)
     ISD.plot_BTSP_maxrates_all_fields_from_formation(save_path=session_output_dir)
     ISD.plot_BTSP_shift_all_fields_from_formation(save_path=session_output_dir)
     ISD.plot_BTSP_shift_all_fields_sorted_by_correlation(save_path=session_output_dir)
     ISD.save_BTSP_statistics(save_path=session_output_dir)
     del ISD

# reset place field variables for dummy
dummy_ISD.unreliable_place_fields = []
dummy_ISD.early_place_fields = []
dummy_ISD.transient_place_fields = []
dummy_ISD.candidate_btsp_place_fields = []
dummy_ISD.btsp_place_fields = []
dummy_ISD.nonbtsp_novel_place_fields = []

for session_name in ISDs:
    ISD = ISDs[session_name]
    dummy_ISD.unreliable_place_fields += ISD.unreliable_place_fields
    dummy_ISD.early_place_fields += ISD.early_place_fields
    dummy_ISD.transient_place_fields += ISD.transient_place_fields
    dummy_ISD.candidate_btsp_place_fields += ISD.candidate_btsp_place_fields
    dummy_ISD.btsp_place_fields += ISD.btsp_place_fields
    dummy_ISD.nonbtsp_novel_place_fields += ISD.nonbtsp_novel_place_fields

dummy_ISD.plot_BTSP_shift_by_lap_histograms(save_path=output_dir)
dummy_ISD.plot_BTSP_shift_gain_by_lap_scatterplots(save_path=output_dir)
dummy_ISD.plot_BTSP_maxrates_all_fields_from_formation(save_path=output_dir)
dummy_ISD.plot_BTSP_shift_all_fields_from_formation(save_path=output_dir)
dummy_ISD.plot_BTSP_shift_all_fields_sorted_by_correlation(save_path=output_dir)
dummy_ISD.save_BTSP_statistics(save_path=output_dir)
