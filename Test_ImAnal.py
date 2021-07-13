# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:14:19 2020
@author: luko.balazs, ujfalussy.balazs
"""

# 1. load the class definitions
from ImageAnal import *

# 2. tell python where the data is 
datapath = os.getcwd() + '/' #current working directory - look for data and strings here!
date_time = '2021-02-03_10-15-50' # date and time of the imaging session
name = 'srb059' # mouse name
task = 'NearFar' # task name

## locate the suite2p folder
suite2p_folder = datapath + 'data/' + name + '_imaging/Suite2P_4_19-05-2021/'

## the name and location of the imaging log file
imaging_logfile_name = suite2p_folder + 'srb059_TSeries-02032021-1011-001.xml'

## the name and location of the trigger voltage file
TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb059_TSeries-02032021-1011-001_Cycle00001_VoltageRecording_001.csv'


# 3. load all the data - this taks ~20 secs in my computer
#    def __init__(self, datapath, date_time, name, task, suite2p_folder, TRIGGER_VOLTAGE_FILENAME):

D1 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, speed_threshold=5)#, startendlap=[27, 99])
try:
	D1 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, speed_threshold=5)#, startendlap=[27, 99])
	# D2 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, speed_threshold=5, selected_laps=np.arange(40, 50))#, startendlap=[27, 99])
except:
	print('Loading data failed! Check for errors!')


##################################################################
## check saving parameters and data into file
##################################################################

D2 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, speed_threshold=5, selected_laps=np.arange(40, 50))#, startendlap=[27, 99])
D1.write_params('aa_test_params1.csv')
D2.write_params('aa_test_params2.csv')

if (D1.check_params('aa_test_params1.csv') == False):
	print('Error: checking parameters failed!')

if (D2.check_params('aa_test_params2.csv') == False):
	print('Error: checking parameters failed!')

if (D1.check_params('aa_test_params2.csv')):
	print('Error: checking parameters failed!')

if (D2.check_params('aa_test_params1.csv')):
	print('Error: checking parameters failed!')

D1.save_data()
D1.check_params('lapdata_lap_162_N1022.csv')


###########################################
##


if (round(D1.imstart_time, 7) - 870.993125 != 0):
	print('matching imaging and behavioral times failed!')

if (D1.spks_all.shape != (3202, 26784)):
	print('Size of imaging array is not as expected!')

if (D1.n_laps != 178):
	print('Number of laps is not 178, as expected')

if (D1.n_laps != len(D1.i_corridors)):
	print('Number of laps is not 178, as expected')

if (np.min(D1.frame_laps) > min(D1.i_Laps_ImData)):
	print('Frame lap numbers mismatched')

if (np.max(D1.frame_laps) < max(D1.i_Laps_ImData)):
	print('frame lap numbers mismatched')

if (np.max(D1.frame_pos) > D1.corridor_length_roxel):
	print('Maze length mismatch')

if (round(np.min(D1.frame_times) - 870.985125, 6) != 0):
	print('Frame times mismatch')

if (D1.N_cells != 1022):
	print('Cell number is not 1022')

if (round(D1.cell_SNR[40] - 57.2069257, 6) != 0):
	print('SNR calculation is corrupted')

if (round(sum(D1.spks[40,:]) - 38648.113207, 6) != 0):
	print('Spikes calculation is incorrect')

if (round(sum(D1.ratemaps[0][40,:]) - 632.4532370509081, 6) != 0):
	print('Ratemap calculation is incorrect')

if (D1.ratemaps[0].shape[1] != D1.N_cells):
	print('Number of cells is not equal to the number of columns in the ratemaps!')


# test for all ImLaps that the 
for ilap in D1.i_Laps_ImData:
	N_rates = len(np.where(np.sum(D1.ImLaps[ilap].event_rate, axis=0) > 0)[0])
	# is only zero, where there are no valid frames - speed is too low
	N_speed = len(np.unique(D1.ImLaps[ilap].frames_pos[np.where(D1.ImLaps[ilap].frames_speed > 5)[0]] // 70))
	if (N_rates != N_speed):
		print ('Error: In lap' , ilap, 'firing rate is calculated in ', N_rates, 'spatial bin, but the speed is above threshold in ', N_speed, 'bins!')


D1.plot_session()
D1.plot_session(selected_laps = np.arange(115, 178))


cellids = np.nonzero(D1.cell_SNR > 30)[0]
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=3, normalized=True)
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=4, normalized=True)
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=3, normalized=False)
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=3, corridor=3, normalized=True)
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=3, corridor=4, normalized=True)


try:
	D1.plot_masks(cellids, D1.cell_SNR, 'signal to noise ratio')
except:
	print('Could not find the ops file (ops.npy).')


cellids = np.arange(500)
D1.plot_popact(cellids, name_string='high SNR neurons', bylaps=False)
D1.plot_popact(cellids, bylaps=True, corridor=3)


D1.plot_cell_laps(cellid=40, signal='rate') ## look at lap 20
D1.plot_cell_laps(cellid=40, signal='rate', corridor=3, plot_laps='correct') ## look at lap 20
D1.plot_cell_laps(cellid=40, signal='dF') ## look at lap 20
D1.plot_cell_laps(cellid=40, signal='dF', corridor=3) ## look at lap 20
D1.plot_cell_laps(cellid=40, signal='dF', corridor=4, reward=False) ## look at lap 20


D1.get_lap_indexes(corridor=3, i_lap=24) # print lap index for the 74th imaging lap in corridor 19
D1.get_lap_indexes(corridor=3) # print all lap indexes in corridor 19

## 5. plotting all cells in a single lap, and zoom in to see cell 106 - it actually matches the data well
D1.ImLaps[102].plot_tx(fluo=True)
D1.ImLaps[102].plot_xv() # around 1435 there is only one frame that passes the speed threshold. Usually there are at least 2...
D1.ImLaps[102].plot_txv()

