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

def color_masks(suite2p_folder, D1):
    ops = np.load(D1.ops_string, allow_pickle=True).item()
    stat = np.load(D1.stat_string, allow_pickle=True)

    cellids_btsp = list(set([pf.cellid for pf in D1.btsp_place_fields]))
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

# read session metadata for selected sessions
session_names = ['srb231_220809_002',
                 'srb251_221028_T1',
                 'srb269_230119',
                 'srb270_230118',
                 'srb270_230120']
xlsx_file = "CA3_meta.xlsx"
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
output_dir = "BTSP_analysis_CA3"
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

     ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=12)
     place_cells = np.union1d(ISD.accepted_PCs[0], ISD.accepted_PCs[1])
     for cellid in place_cells:
         ISD.find_BTSP(cellid)
         #ISD.plot_cell_laps(cellid, signal="rate", multipdf_object=f"{session_output_dir}/ratemaps", write_pdf=True, plot_BTSP=True)
     ISD.plot_BTSP_shift_by_lap_histograms(save_path=session_output_dir)
     ISD.plot_BTSP_shift_gain_by_lap_scatterplots(save_path=session_output_dir)
     ISD.plot_BTSP_maxrates_all_fields_from_formation(save_path=session_output_dir)
     ISD.plot_BTSP_shift_all_fields_from_formation(save_path=session_output_dir)
     ISD.plot_BTSP_shift_all_fields_sorted_by_correlation(save_path=session_output_dir)
     ISD.save_BTSP_statistics(save_path=session_output_dir)

     name = sessions[session_name]["name"]
     suite2p_folder = datapath + 'data/' + f"{name}_imaging/" + sessions[session_name]["suite2p_folder"] + "/"
     color_masks(suite2p_folder, ISD)
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
