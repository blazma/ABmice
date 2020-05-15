# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:14:19 2020

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

# 4. look at the data - plotting functions

## plotting the behavioral data in the different corridors
D1.plot_session()

## plotting cell properties - currently reliability as a function of event rate
cellids = np.nonzero(D1.cell_SNR > 30)[0]
D1.plot_properties(cellids=cellids)

cellids = np.nonzero((D1.cell_tuning_specificity[0] + D1.cell_activelaps[0] > 0.7) + (D1.cell_tuning_specificity[1] + D1.cell_activelaps[1] > 0.7))[0]
D1.plot_properties(cellids=cellids)

cellids = np.nonzero((D1.cell_reliability[0] > 0.15) + (D1.cell_reliability[1] > 0.15))[0]
D1.plot_properties(cellids=cellids)


D1.plot_ratemaps(cellids = cellids)


D1.plot_cell_laps(cellid=24, signal='rate') ## look at lap 20
D1.plot_cell_laps(cellid=24, signal='dF') ## look at lap 20

D1.get_lap_indexes(corridor=16, i_lap=7) # print lap index for the 74th imaging lap in corridor 19
D1.get_lap_indexes(corridor=19) # print all lap indexes in corridor 19

## we see that lap 74 in corridor 19 was the 297 imaging lap
## plotting a single lap, and zoom in to see cell 106 - it actually matches the data well
D1.ImLaps[152].plot_tx(fluo=True)
D1.ImLaps[152].plot_xv()
D1.ImLaps[152].plot_txv()


# D1.cell_rates[]

# D1.plot_ratemaps(corridor=19)

# fig, axs = plt.subplots(2,2, sharex='row', sharey='row')
# axs[0,0].scatter(D1.cell_rates[0], D1.cell_skaggs[0], c='C0', alpha=0.5)
# axs[0,1].scatter(D1.cell_rates[1], D1.cell_skaggs[1], c='C0', alpha=0.5)
# axs[1,0].scatter(D1.cell_reliability[0], D1.cell_skaggs[0], c='C0', alpha=0.5)
# axs[1,1].scatter(D1.cell_reliability[1], D1.cell_skaggs[1], c='C0', alpha=0.5)
# plt.show(block=False)


