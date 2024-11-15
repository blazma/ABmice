from ImageAnal import *


datapath = os.getcwd() + '/'  # current working directory - look for data and strings
date_time = '2021-11-02_17-02-32'  # date and time of the imaging session
name = 'KS028'  # mouse name
task = 'NearFarLong'  # task name
suite2p_folder = datapath + 'data/' + name + '_imaging/KS028_110221/' # locate the suite2p folder
imaging_logfile_name = suite2p_folder + 'KS028_TSeries-11022021-1017-001.xml' # the name and location of the imaging log file
TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'KS028_TSeries-11022021-1017-001_Cycle00001_VoltageRecording_001.csv' # the name and location of the trigger voltage file
D1 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME)
D1.sessionID = "KS028_110221"

output_dir = "BTSP_CA1_SingleInBounds"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(f"{output_dir}/ratemaps"):
    os.makedirs(f"{output_dir}/ratemaps")

D1.calc_shuffle(D1.active_cells, 1000, 'shift', batchsize=10)
place_cells = np.union1d(D1.accepted_PCs[0], D1.accepted_PCs[1])
for cellid in place_cells:
    D1.run_btsp_analysis(cellid)
    D1.plot_cell_laps(cellid, signal="rate", multipdf_object=f"{output_dir}/ratemaps/", write_pdf=True, plot_BTSP=True)

D1.btsp_analysis.plot_BTSP_shift_by_lap_histograms(save_path=output_dir)
D1.btsp_analysis.plot_BTSP_shift_gain_by_lap_scatterplots(save_path=output_dir)
D1.btsp_analysis.plot_BTSP_maxrates_all_fields_from_formation(save_path=output_dir)
D1.btsp_analysis.plot_BTSP_maxrates_all_fields_from_formation(save_path=output_dir, reverse=True)
D1.btsp_analysis.plot_BTSP_shift_all_fields_from_formation(save_path=output_dir)
D1.btsp_analysis.plot_BTSP_shift_all_fields_from_formation(save_path=output_dir, reverse=True)
D1.btsp_analysis.plot_BTSP_shift_all_fields_sorted_by_correlation(save_path=output_dir)
D1.btsp_analysis.save_BTSP_statistics(save_path=output_dir)
D1.btsp_analysis.save_BTSP_place_field_categories(save_path=output_dir)




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