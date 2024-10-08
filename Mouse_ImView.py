# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:14:19 2020
## folyoso hossza 106.5 cm
## 0: kozepetol elore van mintazat
## reward zone: mar eppen nem latszik a mintazat
@author: luko.balazs
"""

# function to save a vector
def save_vec(vec, filename):
   with open(filename, mode='w') as outfile:
       file_writer = csv.writer(outfile, delimiter=',')
       file_writer.writerow(vec)
   print('vector saved into file: ' + filename)


# use this function to save data into file
# np.savetxt("mat.csv", mat, delimiter=",",fmt='%10.5f', header='col1, col2')
# see docs here: https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
# mat can be any numpy array or vector

# 1. load t[]he class definitions
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
D1 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME)#, startendlap=[27, 99])
D1.i_Laps_ImData

# D2 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, selected_laps=np.array([140, 141, 142, 143, 144, 145]))
D2 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, selected_laps=np.arange(140, 225))
D2.i_Laps_ImData

#########################################################
## quick access to plotting functions...
#########################################################
cellids = np.nonzero((D1.cell_activelaps[0]>0.4) + (D1.cell_activelaps[1]>0.4))[0]
D1.calc_shuffle(cellids, n=100, mode='shift')
D1.shuffle_stats.plot_properties_shuffle(maxNcells=30)

D1.plot_session(save_data=False, selected_laps=np.arange(139, 316))

D1.plot_cell_laps(cellid=194, signal='rate', save_data=False) ## look at lap 20
D1.plot_cell_laps(cellid=19, signal='dF') ## look at lap 20

D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=19)

D1.plot_popact(cellids, bylaps=True)
D1.plot_popact(cellids, bylaps=False)


#########################################################
## PLOTTING
#########################################################
## 0. plotting the behavioral data in the different corridors
D1.plot_session(save_data=False)
D1.plot_session(save_data=False, selected_laps=np.arange(0, 139))
D1.plot_session(save_data=False, selected_laps=np.arange(139, 316))
D1.plot_session(save_data=False, selected_laps=np.arange(316, 350))

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
cellids = np.nonzero(D1.candidate_PCs[0] + D1.candidate_PCs[1])[0]

D1.plot_ratemaps(cellids = cellids)

D1.plot_cell_laps(cellid=1076, signal='rate', save_data=False) ## look at lap 20

# > 200:  1076, 351, 269


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
cellids = np.nonzero(((D1.cell_tuning_specificity[0]/40 + D1.cell_activelaps[0]) > 0.5) + ((D1.cell_tuning_specificity[1]/40 + D1.cell_activelaps[1]) > 0.5))[0]
D1.plot_ratemaps(cellids = cellids)

## 1.2 sorting ratemaps
D1.plot_ratemaps(cellids = cellids, sorted=True)
D1.plot_ratemaps(cellids = cellids, corridor=19, sorted=True)
D1.plot_ratemaps(cellids = cellids, sorted=True, corridor_sort=19)

## 1.3 plotting the total population activity
cellids = np.nonzero((D1.cell_reliability[0] > 0.3) + (D1.cell_reliability[1] > 0.3))[0]
D1.plot_popact(cellids)
D1.plot_popact(cellids, bylaps=True)

cellids = np.nonzero((D1.cell_reliability[0] > 0))[0]
D1.plot_popact(cellids, bylaps=True)
D1.plot_popact(cellids, bylaps=False)

## 2. plot properties - you can select interactive mode to be True or False
cellids = np.nonzero(D1.candidate_PCs[0] + D1.candidate_PCs[1])[0]
D1.plot_properties(cellids=cellids, interactive=False)

# 3. plot masks - only works in python 3
D1.plot_masks(cellids)

# 4. plot the laps of a selected cell - there are two options:
# 4a) plot the event rates versus space
D1.plot_cell_laps(cellid=110, signal='rate', save_data=False) ## look at lap 20

# 4a) plot the dF/F and the spikes versus time
D1.plot_cell_laps(cellid=106, signal='dF') ## look at lap 20

# get the index of a lap in a specific corridor
D1.get_lap_indexes(corridor=16, i_lap=77) # print lap index for the 74th imaging lap in corridor 19
D1.get_lap_indexes(corridor=19) # print all lap indexes in corridor 19

## 5. plotting all cells in a single lap, and zoom in to see cell 106 - it actually matches the data well
D1.ImLaps[285].plot_tx(fluo=True)
D1.ImLaps[285].plot_xv()
D1.ImLaps[153].plot_xv()
D1.ImLaps[153].plot_txv()

# fig, axs = plt.subplots(2,2, sharex='row', sharey='row')
# axs[0,0].scatter(D1.cell_rates[0], D1.cell_skaggs[0], c='C0', alpha=0.5)
# axs[0,1].scatter(D1.cell_rates[1], D1.cell_skaggs[1], c='C0', alpha=0.5)
# axs[1,0].scatter(D1.cell_reliability[0], D1.cell_skaggs[0], c='C0', alpha=0.5)
# axs[1,1].scatter(D1.cell_reliability[1], D1.cell_skaggs[1], c='C0', alpha=0.5)
# plt.show(block=False)


## 6. calculate shuffle controls
## 6.1. shuffle for candidate place cells
# cellids = np.nonzero(D1.candidate_PCs[0] + D1.candidate_PCs[1])[0]
## 6.1. shuffle for all active cells
cellids = np.nonzero((D1.cell_activelaps[0]>0.2) + (D1.cell_activelaps[1]>0.2))[0]
D1.calc_shuffle(cellids, n=100, mode='shift')




### this is how you save the P values into file:
np.savetxt("Pvals.csv", np.transpose(np.vstack((cellids, D1.shuffle_stats.P_all))), delimiter=",",fmt='%10.5f', header='cellid' + str(D1.shuffle_stats.P_all_names))



D1.shuffle_stats.plot_properties_shuffle()

D1.plot_ratemaps(cellids=D1.tuned_cells['cells_reli_0'], corridor=19, sorted=True)

for i_cell in D1.tuned_cells['cells_reli_0']:
	D1.plot_cell_laps(cellid=i_cell, signal='rate', save_data=False)

# > 200:  1076, 351, 269
# > 150:  785, 4
# > 100:  10, 14, 24, 28, 34, 36, 41
# < 100: 24

## corridor selective cells
selective_cells = cellids[np.where((D1.shuffle_stats.P_selectivity[0] < 0.025))[0]]
selective_cells_corr = cellids[np.where((D1.shuffle_stats.P_selectivity[1] < 0.025))[0]]
selective_cells_rew = cellids[np.where((D1.shuffle_stats.P_selectivity[2] < 0.025))[0]]

Nselective_cells = cellids[np.where((D1.shuffle_stats.P_selectivity[0] > 0.975))[0]]
Nselective_cells_corr = cellids[np.where((D1.shuffle_stats.P_selectivity[1] > 0.975))[0]]
Nselective_cells_rew = cellids[np.where((D1.shuffle_stats.P_selectivity[2] > 0.975))[0]]

D1.shuffle_stats.plot_properties_shuffle(cellids=Nselective_cells_corr)
D1.plot_ratemaps(cellids = Nselective_cells_corr)
D1.cell_corridor_selectivity.shape[:,Nselective_cells_corr]

D1.plot_ratemaps(cellids = selective_cells_rew)
D1.plot_ratemaps(cellids = Nselective_cells_rew)


## if you need to shuffle many calls and want to use a large number of shuffles, then you may want to speed up the computation using minibatchses
## the batchsize parameter allows you to select the number of cells used within each iteration
## always use the largest batchsize that is still fast
## the randseed parameter allows zou to replicate the results by selecting the same random seed for the shuffling
## it is ideally a random integer number, different for each analysis
D1.calc_shuffle(cellids, n=100, mode='shift', batchsize=5, randseed=124)
D1.shuffle_stats.plot_properties_shuffle(cellids=cellids)



D1.plot_properties(cellids=cellids, interactive=False)


## 6.1. shuffle for all cells with high specificity and activity rate
cellids = np.nonzero((D1.cell_tuning_specificity[0] > 5) & (D1.cell_rates[0][0,:] > 0.2)+ (D1.cell_tuning_specificity[1] > 5) & (D1.cell_rates[1][0,:] > 0.2))[0]
## 6.1. shuffle for the first 100 cells
cellids = np.arange(100)
cellids = np.nonzero((D1.cell_activelaps[0] > 0.2) + (D1.cell_activelaps[1] > 0.2))[0]

D1.plot_properties(cellids=cellids, interactive=False)
D1.calc_shuffle(cellids, n=100, mode='shift')
D1.shuffle_stats.plot_properties_shuffle(maxNcells=100)

## plot the ratemaps of all significantly specific cells
cells16 = cellids[np.where((D1.shuffle_stats.P_reliability[0] < 0.01) + (D1.shuffle_stats.P_tuning_specificity[0] < 0.01) + (D1.shuffle_stats.P_skaggs[0] < 0.01))[0]]
cells19 = cellids[np.where((D1.shuffle_stats.P_reliability[1] < 0.01) + (D1.shuffle_stats.P_tuning_specificity[1] < 0.01) + (D1.shuffle_stats.P_skaggs[1] < 0.01))[0]]
D1.plot_ratemaps(cellids = cells16, sorted=True, corridor=16)
D1.plot_ratemaps(cellids = cells19, sorted=True, corridor=19)
D1.plot_ratemaps(cellids = cells, sorted=True, corridor_sort=19)





## random cells
cellids = np.arange(100) + 100
D1.calc_shuffle(cellids, n=100, mode='shift')
D1.shuffle_stats.plot_properties_shuffle(maxNcells=100)

cells = cellids[np.where((D1.shuffle_stats.P_tuning_specificity[0] < 0.05) + (D1.shuffle_stats.P_tuning_specificity[1] < 0.05))[0]]
D1.plot_ratemaps(cellids = cells, sorted=True)
D1.plot_ratemaps(cellids = cells, sorted=True, corridor_sort=19)



D1.shuffle_stats.P_skaggs[0][D1.shuffle_stats.P_skaggs[0] < 1.0/1000] = 1.0/2000
D1.shuffle_stats.P_skaggs[1][D1.shuffle_stats.P_skaggs[1] < 1.0/1000] = 1.0/2000

D1.shuffle_stats.P_tuning_specificity[0][D1.shuffle_stats.P_tuning_specificity[0] < 1.0/1000] = 1.0/2000
D1.shuffle_stats.P_tuning_specificity[1][D1.shuffle_stats.P_tuning_specificity[1] < 1.0/1000] = 1.0/2000

D1.shuffle_stats.P_reliability[0][D1.shuffle_stats.P_reliability[0] < 1.0/1000] = 1.0/2000
D1.shuffle_stats.P_reliability[1][D1.shuffle_stats.P_reliability[1] < 1.0/1000] = 1.0/2000

fig, ax = plt.subplots(1, 3, figsize=(10,5), sharex='col', sharey='col')
plt.subplots_adjust(wspace=0.35, hspace=0.2)

ax[0].plot(D1.shuffle_stats.P_skaggs[0], D1.shuffle_stats.P_tuning_specificity[0], 'o', alpha=0.5, c='w', markeredgecolor='C1')
ax[0].plot(D1.shuffle_stats.P_skaggs[1], D1.shuffle_stats.P_tuning_specificity[1], 'o', alpha=0.5, c='w', markeredgecolor='C2')
ax[0].set_xlabel('Skaggs info P')
ax[0].set_ylabel('tuning specificity P')
ax[0].set_xscale('log')
ax[0].set_yscale('log')


ax[1].plot(D1.shuffle_stats.P_skaggs[0], D1.shuffle_stats.P_reliability[0], 'o', alpha=0.5, c='w', markeredgecolor='C1')
ax[1].plot(D1.shuffle_stats.P_skaggs[1], D1.shuffle_stats.P_reliability[1], 'o', alpha=0.5, c='w', markeredgecolor='C2')
ax[1].set_xlabel('Skaggs info P')
ax[1].set_ylabel('reliability P')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

ax[2].plot(D1.shuffle_stats.P_reliability[0], D1.shuffle_stats.P_tuning_specificity[0], 'o', alpha=0.5, c='w', markeredgecolor='C1')
ax[2].plot(D1.shuffle_stats.P_reliability[1], D1.shuffle_stats.P_tuning_specificity[1], 'o', alpha=0.5, c='w', markeredgecolor='C2')
ax[2].set_xlabel('reliability P')
ax[2].set_xlabel('tuning specificity P')
ax[2].set_xscale('log')
ax[2].set_yscale('log')

plt.show(block=False)
