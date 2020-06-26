# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:14:19 2020
## folyoso hossza 106.5 cm
## 0: kozepetol elore van mintazat
## reward zone: mar eppen nem latszik a mintazat
@author: luko.balazs
"""

# 1. load the class definitions
from ImageAnal import *

# 2. tell python where the data is 
datapath = os.getcwd() + '/' #current working directory - look for data and strings here!
date_time = '2020-03-13_12-15-01' # date and time of the imaging session
name = 'rn013' # mouse name
task = 'contingency_learning' # task name

## locate the suite2p folder
suite2p_folder = datapath + 'data/' + name + '_imaging/rn013_TSeries-03132020-0939-003/'

## the name and location of the imaging log file
imaging_logfile_name = suite2p_folder + 'rn013_TSeries-03132020-0939-003.xml'

## the name and location of the trigger voltage file
TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'rn013_TSeries-03132020-0939-003_Cycle00001_VoltageRecording_001.csv'


# 3. load all the data - this taks ~20 secs in my computer
#    def __init__(self, datapath, date_time, name, task, suite2p_folder, TRIGGER_VOLTAGE_FILENAME):
D1 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME)

#########################################################
## PLOTTING
#########################################################
## 0. plotting the behavioral data in the different corridors
D1.plot_session(save_data=True)

## 1. select cells based on their properties
## a) selection based on signal-to noise ratio:
cellids = np.nonzero(D1.cell_SNR > 30)[0]
## b) candidate place cells in either corridor 0 OR corridor 1
cellids = np.nonzero(D1.candidate_PCs[0] + D1.candidate_PCs[1])[0]
## c) corridor selectivity - a matrix with 3 rows: 
## selectivity in the whole corridor - cells firing more in one of the corridors
cellids = np.nonzero(abs(D1.cell_corridor_selectivity[0,]) > 0.3)[0]
## selectivity in the pattern area - cell firing in coridor 19 pattern zone 
cellids = np.nonzero(D1.cell_corridor_selectivity[1,] < -0.3)[0]
## selectivity in the reward zone - cell more active in corridor 16 reward zone
cellids = np.nonzero(D1.cell_corridor_selectivity[2,] > 0.5)[0]
# the boundary between the pattern zone and the reward zone can be set in the function 'calculate_properties' in the ImageAnal.py
# rates_pattern = np.sum(total_spikes[5:40,:], axis=0) / np.sum(total_time[5:40])
# rates_reward = np.sum(total_spikes[40:50], axis=0) / np.sum(total_time[40:50])


## 1.1. plot the ratemaps of the selected cells
D1.plot_ratemaps(cellids = cellids)


## d) corridor selectivity - whether the ratemaps are similar in the two corridors
## first row - with 0 index: similarity in Pearson correlation
## second row - with 1 index: P-value
cellids = np.nonzero(D1.cell_corridor_similarity[0, ] > 0.75)[0]
D1.plot_ratemaps(cellids = cellids)

## e) other possibilities are:
# cell_rates  - event rates
# cell_relility - spatial reliability
# cell_Fano_fact - Fano factor
# cell_skaggs - Skaggs spatial info
# cell_activelaps - % active laps based on spikes
# cell_activelaps_df - % active laps (dF/F)
# cell_tuning_specificity - tuning specificity
#
# a single criterion is specified like this:
cellids = np.nonzero(D1.cell_tuning_specificity[0] > 0.5)[0]

# a combination of multiple criteria can be specified like this:
cellids = np.nonzero((D1.cell_tuning_specificity[0] + D1.cell_activelaps[0] > 0.7) + (D1.cell_tuning_specificity[1] + D1.cell_activelaps[1] > 0.7))[0]
D1.plot_ratemaps(cellids = cellids)

## 1.2 sorting ratemaps
D1.plot_ratemaps(cellids = cellids, sorted=True)
D1.plot_ratemaps(cellids = cellids, corridor=19, sorted=True)
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=19)

## 1.3 plotting the total population activity
cellids = np.nonzero((D1.cell_reliability[0] > 0.3) + (D1.cell_reliability[1] > 0.3))[0]
D1.plot_popact(cellids)


## 2. plot properties - you can select interactive mode to be True or False
cellids = np.nonzero(D1.candidate_PCs[0] + D1.candidate_PCs[1])[0]
D1.plot_properties(cellids=cellids, interactive=False)

# 3. plot masks - only works in python 3
D1.plot_masks(cellids)

# 4. plot the laps of a selected cell - there are two options:
# 4a) plot the event rates versus space
D1.plot_cell_laps(cellid=209, signal='rate', save_data=False) ## look at lap 20

# 4a) plot the dF/F and the spikes versus time
D1.plot_cell_laps(cellid=209, signal='dF') ## look at lap 20

# get the index of a lap in a specific corridor
D1.get_lap_indexes(corridor=16, i_lap=7) # print lap index for the 74th imaging lap in corridor 19
D1.get_lap_indexes(corridor=19) # print all lap indexes in corridor 19

## 5. plotting all cells in a single lap, and zoom in to see cell 106 - it actually matches the data well
D1.ImLaps[152].plot_tx(fluo=True)
D1.ImLaps[152].plot_xv()
D1.ImLaps[153].plot_xv()
D1.ImLaps[153].plot_txv()

# fig, axs = plt.subplots(2,2, sharex='row', sharey='row')
# axs[0,0].scatter(D1.cell_rates[0], D1.cell_skaggs[0], c='C0', alpha=0.5)
# axs[0,1].scatter(D1.cell_rates[1], D1.cell_skaggs[1], c='C0', alpha=0.5)
# axs[1,0].scatter(D1.cell_reliability[0], D1.cell_skaggs[0], c='C0', alpha=0.5)
# axs[1,1].scatter(D1.cell_reliability[1], D1.cell_skaggs[1], c='C0', alpha=0.5)
# plt.show(block=False)


## 6. calculate shuffle controls
cellids = np.nonzero(D1.candidate_PCs[0] + D1.candidate_PCs[1])[0]
D1.plot_properties(cellids=cellids, interactive=False)
D1.calc_shuffle(cellids, n=100, mode='shift')
D1.shuffle_stats.plot_properties_shuffle()


cellids = np.arange(100) + 600
D1.calc_shuffle(cellids, n=100, mode='shift')
D1.shuffle_stats.plot_properties_shuffle(maxNcells=100)


D1.calc_shuffle(cellids, n=100, mode='random')
D1.shuffle_stats.plot_properties_shuffle()


cellids = np.nonzero((D1.cell_reliability[0] > 0.25) + (D1.cell_reliability[1] > 0.25))[0]
D1.calc_shuffle(cellids, n=100, mode='random')
D1.shuffle_stats.plot_properties_shuffle(cellids[0:10])
D1.shuffle_stats.plot_properties_shuffle(cellids[10:20])

cellids = np.nonzero((D1.cell_reliability[0] < 0.15) & (D1.cell_activelaps[0] > 0.2))[0]
D1.calc_shuffle(cellids, n=100, mode='random')
D1.shuffle_stats.plot_properties_shuffle(cellids, maxNcells=50)
