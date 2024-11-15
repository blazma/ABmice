import numpy as np
import pandas as pd
import os
from ImageAnal import ImagingSessionData

# create dummy ImagingSessionData to fill up later
datapath = os.getcwd() + '/'  # current working directory - look for data and strings
date_time = '2021-11-02_17-02-32'  # date and time of the imaging session
name = 'KS028'  # mouse name
task = 'NearFarLong'  # task name
suite2p_folder = datapath + 'data/' + name + '_imaging/KS028_110221/' # locate the suite2p folder
imaging_logfile_name = suite2p_folder + 'KS028_TSeries-11022021-1017-001.xml' # the name and location of the imaging log file
TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'KS028_TSeries-11022021-1017-001_Cycle00001_VoltageRecording_001.csv' # the name and location of the trigger voltage file
ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME)
ISD.sessionID = "Mate_sesh"

# change dummy ISD with Mate's data
data = pd.read_pickle('data_Mate/all_spks_df.pickle').groupby(['Run','LapNo','SpaceBin']).mean()
cellids = list(data.columns)  # ROI IDs

ISD.N_cells = len(cellids)
ISD.N_pos_bins = 196
ISD.N_ImLaps = 56
ISD.corridors = np.zeros(1)  # let's say their only corridor is corridor number 0
ISD.i_Laps_ImData = np.arange(ISD.N_ImLaps)  # lap indices with imaging, let's assume it's all of them
ISD.i_corridors = np.zeros(ISD.N_ImLaps)  # corridor indices for each lap, which is 0 now assuming all from the same

# trplt_pf_df.roi.unique()
ISD.tuned_cells = [np.array(cellids), np.array(cellids)]  # cell indices in "both" corridors, TODO: lekezelni hogy nincs két korridor (vagy kettőnél több van

ISD.p95 = np.ones((1, ISD.N_pos_bins, len(cellids)))  # (corridors X bins X cells)
ISD.activity_tensor = np.zeros((ISD.N_pos_bins, ISD.N_cells, ISD.N_ImLaps))
ISD.activity_tensor_time = np.ones((ISD.N_pos_bins, ISD.N_ImLaps))  # frame time ide jöhet

run = 1
for i, cellid in enumerate(cellids):
    rate_matrix = np.transpose(data.loc[pd.IndexSlice[run,:,:], cellid].unstack('SpaceBin').values)
    ISD.activity_tensor[:,i,:] = rate_matrix

for i, cellid in enumerate(cellids):
    ISD.find_BTSP(i)
    ISD.plot_cell_laps(i, signal="rate", multipdf_object=f"BTSP_Mate/ratemaps/", write_pdf=True, plot_BTSP=True)