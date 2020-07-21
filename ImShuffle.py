# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:14:19 2020

@author: bbujfalussy - ubalazs317@gmail.com
luko.balazs - lukobalazs@gmail.com
, 
"""


import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors as matcols
# from pathlib import Path # this only works with python > 3.4, not compatible with 2.7
from scipy.interpolate import interp1d
import scipy.stats
import csv

from utils import *
from Stages import *
from Corridors import *

def breakpoints(Nframes, Lmin=500, Nbreak=5):
    if (Nframes < (Lmin * Nbreak)):
        print('Nframes < (Lmin * Nbreak), we use smaller segments')
        Lmin = floor(Nframes / Nbreak)
    alpha = np.ones(Nbreak+1)
    seclengths = np.round(scipy.stats.dirichlet.rvs(alpha)[0] * (Nframes - Lmin * (Nbreak+1)) + Lmin) ## see Dirichlet distribution...
    seclengths[Nbreak] = Nframes - np.sum(seclengths[0:Nbreak]) # the original list may be a bit longer or shorter...
    sections = np.zeros((2, Nbreak+1))
    for i in range(Nbreak+1):
        sections[0,i] = np.sum(seclengths[0:i])# start index
        sections[1,i] = seclengths[i]# length
    return sections

class ImShuffle:
    'Base structure for shuffling analysis of imaging data'
    def __init__(self, datapath, date_time, name, task, stage, raw_spikes, frame_times, frame_pos, frame_laps, N_shuffle=1000, cellids=np.array([-1]), mode='random'):
        self.name = name
        self.date_time = date_time
        self.stage = 0
        self.stages = []
        self.stage = stage

        self.corridor_length_cm = 106.5 # cm
        self.corridor_length_roxel = 3500 # cm
        self.speed_factor = self.corridor_length_cm / self.corridor_length_roxel

        self.raw_spikes = raw_spikes
        self.frame_times = frame_times
        self.frame_pos = frame_pos
        self.frame_laps = frame_laps

        self.N_cells = self.raw_spikes.shape[0]
        if (cellids.size != self.N_cells):
            print('Cellids should be provided for shuffle analysis! We stop.')
            return
        if (cellids[0] == -1):
            print('Cellids should be provided for shuffle analysis! We stop.')
            return
        self.cellids = cellids

        self.N_frames = self.raw_spikes.shape[1]
        self.N_shuffle = N_shuffle
        self.mode = mode # random: totally randomize the spike times; shift: circularly shift spike times 

        stagefilename = datapath + task + '_stages.pkl'
        input_file = open(stagefilename, 'rb')
        if version_info.major == 2:
            self.stage_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.stage_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        corridorfilename = datapath + task + '_corridors.pkl'
        input_file = open(corridorfilename, 'rb')
        if version_info.major == 2:
            self.corridor_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.corridor_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        self.corridors = np.hstack([0, np.array(self.stage_list.stages[self.stage].corridors)])

          
        ##################################################
        ## shuffling the spikes data
        ##################################################

        self.shuffle_spikes = np.zeros((self.N_cells, self.N_frames, self.N_shuffle+1))
        self.shuffle_spikes[:,:,self.N_shuffle] = self.raw_spikes # the last is the real data ...

        if (self.mode == 'shift'):
            for i_shuffle in range(self.N_shuffle):
                spks = np.zeros_like(self.raw_spikes)
                ## we break up the array into 6 pieces of at least 500 frames, permuting them and circularly shifting by at least 500 frames
                Nbreak = 5
                sections = breakpoints(Nframes=self.N_frames, Lmin=500, Nbreak=Nbreak)
                order = np.random.permutation(Nbreak + 1)
                k = 0
                for j in range(Nbreak+1):
                    i_section = order[j]
                    spks[:,k:(k+int(sections[1,i_section]))] = self.raw_spikes[:,int(sections[0,i_section]):int(sections[0,i_section]+sections[1,i_section])]
                    k = k + int(sections[1,i_section])

                n_roll = np.random.randint((self.N_frames - 1000)) + 500
                spks_rolled = np.roll(spks, n_roll, axis=1)
                self.shuffle_spikes[:,:,i_shuffle] = spks_rolled
        else:
            if (self.mode != 'random'):
                print ('Warning: shuffling mode must be either random or shift. We will use radnom.')
            for i_shuffle in range(self.N_shuffle):
                spks = np.copy(self.raw_spikes)
                spks = np.moveaxis(spks, 1, 0)
                np.random.shuffle(spks)
                spks = np.moveaxis(spks, 1, 0)
                self.shuffle_spikes[:,:,i_shuffle] = spks


        ##################################################
        ## loading behavioral data
        ##################################################

        self.shuffle_ImLaps = [] # list containing a special class for storing the imaging and behavioral data for single laps
        self.n_laps = 0 # total number of laps
        self.i_Laps_ImData = np.zeros(1) # np array with the index of laps with imaging
        self.i_corridors = np.zeros(1) # np array with the index of corridors in each run

        self.get_lapdata_shuffle(datapath, date_time, name, task) # collects all behavioral and imaging data and sort it into laps, storing each in a Lap_ImData object
        self.N_bins = 50 # self.shuffle_ImLaps[self.i_Laps_ImData[2]].event_rate.shape[1]
        self.N_ImLaps = len(self.i_Laps_ImData)

        self.raw_activity_tensor = np.zeros((self.N_bins, self.N_cells, self.N_ImLaps, self.N_shuffle+1)) # a tensor with space x neurons x trials x shuffle containing the spikes
        self.raw_activity_tensor_time = np.zeros((self.N_bins, self.N_ImLaps)) # a tensor with space x trials containing the time spent at each location in each lap
        self.activity_tensor = np.zeros((self.N_bins, self.N_cells, self.N_ImLaps, self.N_shuffle+1)) # same as the activity tensor spatially smoothed
        self.activity_tensor_time = np.zeros((self.N_bins, self.N_ImLaps)) # same as the activity_tensor_time spatially smoothed
        self.combine_lapdata_shuffle() ## fills in the cell_activity tensor

        self.cell_rates = [] # a list, each element is a 3 x n_cells matrix with the average rate of the cells in the total corridor, pattern zone and reward zone
        self.cell_reliability = [] # a list, each element is a matrix with the reliability of the shuffles in a corridor
        self.cell_Fano_factor = [] # a list, each element is a matrix with the reliability of the shuffles in a corridor
        self.cell_skaggs=[] # a list, each element is a matrix with the skaggs93 spatial info of the shuffles in a corridor
        self.cell_activelaps=[] # a list, each element is a matrix with the % of significantly spiking laps of the shuffles in a corridor
        self.cell_tuning_specificity=[] # a list, each element is a matrix with the tuning specificity of the shuffles in a corridor
        self.cell_corridor_selectivity = np.zeros((3, self.N_cells, self.N_shuffle+1)) # a matrix with the selectivity of the shuffles in the maze, corridor, reward area

        self.P_reliability = [] # a list, each element is a vector with the P value estimated from shuffle control - P(reliability > measured)
        self.P_skaggs=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(Skaggs-info > measured)
        self.P_tuning_specificity=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(specificity > measured)
        self.P_selectivity = np.zeros((3, self.N_cells)) # a matrix with the selectivity of the the P value estimated from shuffle control - P(selectivity > measured)

        self.calculate_properties_shuffle()

        self.candidate_PCs = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping
        self.cell_corridor_similarity = np.zeros((2, self.N_cells, self.N_shuffle+1)) # a matrix with the pearson R and P value of the correlation between the ratemaps in the two mazes
        self.Hainmuller_PCs_shuffle()

    def get_lapdata_shuffle(self, datapath, date_time, name, task):

        time_array=[]
        lap_array=[]
        maze_array=[]
        position_array=[]
        mode_array=[]
        lick_array=[]
        action=[]

        data_log_file_string=datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_ExpStateMashineLog.txt'
        data_log_file=open(data_log_file_string)
        log_file_reader=csv.reader(data_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            time_array.append(float(line[0]))
            lap_array.append(int(line[1]))
            maze_array.append(int(line[2]))
            position_array.append(int(line[3]))
            mode_array.append(line[6] == 'Go')
            lick_array.append(line[9] == 'TRUE')
            action.append(str(line[14]))

        laptime = np.array(time_array)
        pos = np.array(position_array)
        lick = np.array(lick_array)
        lap = np.array(lap_array)
        maze = np.array(maze_array)
        mode = np.array(mode_array)
        N_0lap = 0 # Counting the non-valid laps
        self.n_laps = 0

        i_ImData = [] # index of laps with imaging data
        i_corrids = [] # ID of corridor for the current lap

        for i_lap in np.unique(lap):
            y = lap == i_lap # index for the current lap

            mode_lap = np.prod(mode[y]) # 1 if all elements are recorded in 'Go' mode

            maze_lap = np.unique(maze[y])
            if (len(maze_lap) == 1):
                corridor = self.corridors[int(maze_lap)] # the maze_lap is the index of the available corridors in the given stage
            else:
                corridor = -1

            if (corridor > 0):
                i_corrids.append(corridor) # list with the index of corridors in each run
                t_lap = laptime[y]
                pos_lap = pos[y]
    
                lick_lap = lick[y] ## vector of Trues and Falses
                t_licks = t_lap[lick_lap] # time of licks
    
                istart = np.where(y)[0][0]
                iend = np.where(y)[0][-1] + 1
                action_lap = action[istart:iend]
    
                reward_indices = [j for j, x in enumerate(action_lap) if x == "TrialReward"]
                t_reward = t_lap[reward_indices]
    
                actions = []
                for j in range(len(action_lap)):
                    if not((action_lap[j]) in ['No', 'TrialReward']):
                        actions.append([t_lap[j], action_lap[j]])

                ### imaging data    
                iframes = np.where(self.frame_laps == i_lap)[0]
                if (len(iframes) > 0): # there is imaging data belonging to this lap...
                    # print(i, min(iframes), max(iframes))
                    lap_frames_spikes = self.shuffle_spikes[:,iframes,:]
                    lap_frames_time = self.frame_times[iframes]
                    lap_frames_pos = self.frame_pos[iframes]
                    i_ImData.append(self.n_laps)
                else:
                    lap_frames_spikes = np.nan
                    lap_frames_time = np.nan
                    lap_frames_pos = np.nan 
                    
                # sessions.append(Lap_Data(name, i, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions))
                self.shuffle_ImLaps.append(Shuffle_ImData(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, lap_frames_spikes, lap_frames_pos, lap_frames_time, self.corridor_list, speed_factor=self.speed_factor))
                self.n_laps = self.n_laps + 1
            else:
                N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1) - we do not do anythin with this...

        self.i_Laps_ImData = np.array(i_ImData) # index of laps with imaging data
        self.i_corridors = np.array(i_corrids) # ID of corridor for the current lap
        # self.cell_activity = np.zeros((self.N_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials

    def combine_lapdata_shuffle(self): ## fills in the cell_activity tensor
        # self.raw_activity_tensor = np.zeros((self.N_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials containing the spikes
        # self.raw_activity_tensor_time = np.zeros((self.N_bins, self.N_ImLaps)) # a tensor with space x  trials containing the time spent at each location in each lap
        valid_lap = np.zeros(len(self.i_Laps_ImData))
        k_lap = 0
        for i_lap in self.i_Laps_ImData:
            if (self.shuffle_ImLaps[i_lap].n_cells > 1):
                valid_lap[k_lap] = 1
                # self.spks_pos = np.zeros((N_cells, 50, self.n_shuffle)) # sum of spike counts measured at a given position
                # self.raw_activity_tensor = np.zeros((50, N_cells, self.N_ImLaps, self.N_shuffle)) # a tensor with space x neurons x trials x shuffle containing the spikes
                self.raw_activity_tensor[:,:,k_lap,:] = np.moveaxis(self.shuffle_ImLaps[i_lap].spks_pos, 1, 0)
                self.raw_activity_tensor_time[:,k_lap] = self.shuffle_ImLaps[i_lap].T_pos
            k_lap = k_lap + 1

        ## smoothing - average of the 3 neighbouring bins
        self.activity_tensor[0,:,:,:] = (self.raw_activity_tensor[0,:,:,:] + self.raw_activity_tensor[1,:,:,:]) / 2
        self.activity_tensor[-1,:,:,:] = (self.raw_activity_tensor[-1,:,:,:] + self.raw_activity_tensor[-1,:,:,:]) / 2
        self.activity_tensor_time[0,:] = (self.raw_activity_tensor_time[0,:] + self.raw_activity_tensor_time[1,:]) / 2
        self.activity_tensor_time[-1,:] = (self.raw_activity_tensor_time[-1,:] + self.raw_activity_tensor_time[-1,:]) / 2
        for i_bin in np.arange(1, self.N_bins-1):
            self.activity_tensor[i_bin,:,:,:] = np.average(self.raw_activity_tensor[(i_bin-1):(i_bin+2),:,:,:], axis=0)
            self.activity_tensor_time[i_bin,:] = np.average(self.raw_activity_tensor_time[(i_bin-1):(i_bin+2),:], axis=0)

        i_valid_laps = np.nonzero(valid_lap)[0]
        self.i_Laps_ImData = self.i_Laps_ImData[i_valid_laps]
        self.raw_activity_tensor = self.raw_activity_tensor[:,:,i_valid_laps,:]
        self.raw_activity_tensor_time = self.raw_activity_tensor_time[:,i_valid_laps]
        self.activity_tensor = self.activity_tensor[:,:,i_valid_laps,:]
        self.activity_tensor_time = self.activity_tensor_time[:,i_valid_laps]



    def calculate_properties_shuffle(self):
        self.ratemaps = [] # a list, each element is an array space x neurons being the ratemaps of the cells in a given corridor
        self.cell_rates = [] # we do not append if it already exists...
        self.cell_reliability = []
        self.cell_Fano_factor = []
        self.cell_skaggs=[]
        self.cell_activelaps=[]
        self.cell_tuning_specificity=[]

        self.P_reliability = [] # a list, each element is a vector with the P value estimated from shuffle control - P(reliability > measured)
        self.P_skaggs=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(Skaggs-info > measured)
        self.P_tuning_specificity=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(specificity > measured)

        N_corridors = len(self.corridors)
        if (N_corridors > 1):
            for i_corridor in (np.arange(N_corridors-1)+1):
                corridor = self.corridors[i_corridor]
                # select the laps in the corridor 
                # only laps with imaging data are selected - this will index the activity_tensor
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor)[0] 
                N_laps_corr = len(i_laps)

                time_matrix_1 = self.activity_tensor_time[:,i_laps]
                total_time = np.sum(time_matrix_1, axis=1) # bins x cells -> bins; time spent in each location

                act_tensor_1 = self.activity_tensor[:,:,i_laps,:] ## bin x cells x laps x shuffle; all activity in all laps in corridor i
                total_spikes = np.sum(act_tensor_1, axis=2) ##  bin x cells x shuffle; total activity of the cells in corridor i

                rate_matrix = np.zeros_like(total_spikes) ## event rate 
                
                for i_cell in range(self.N_cells):
                    for i_shuffle in range(self.N_shuffle+1):
                        rate_matrix[:,i_cell,i_shuffle] = total_spikes[:,i_cell,i_shuffle] / total_time

                self.ratemaps.append(rate_matrix)

                print('calculating rate, reliability and Fano factor...')

                ## average firing rate
                rates = np.sum(total_spikes, axis=0) / np.sum(total_time) # cells x shuffle
                rates_pattern = np.sum(total_spikes[5:40,:,:], axis=0) / np.sum(total_time[5:40])
                rates_reward = np.sum(total_spikes[40:50], axis=0) / np.sum(total_time[40:50])
                self.cell_rates.append(np.stack((rates, rates_pattern, rates_reward), axis=0))

                ## reliability and Fano factor
                reliability = np.zeros((self.N_cells, self.N_shuffle+1))
                P_reliability = np.zeros(self.N_cells)
                Fano_factor = np.zeros((self.N_cells, self.N_shuffle+1))
                for i_cell in range(self.N_cells):
                    for i_shuffle in range(self.N_shuffle+1):
                        laps_rates = nan_divide(act_tensor_1[:,i_cell,:,i_shuffle], time_matrix_1, where=(time_matrix_1 > 0.025))
                        corrs_cell = vcorrcoef(np.transpose(laps_rates), rate_matrix[:,i_cell,i_shuffle])
                        reliability[i_cell, i_shuffle] = np.nanmean(corrs_cell)
                        Fano_factor[i_cell, i_shuffle] = np.nanmean(nan_divide(np.nanvar(laps_rates, axis=1), rate_matrix[:,i_cell,i_shuffle], rate_matrix[:,i_cell,i_shuffle] > 0))

                    shuffle_ecdf = reliability[i_cell, 0:self.N_shuffle]
                    data_point = reliability[i_cell, self.N_shuffle]
                    P_reliability[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

                self.cell_reliability.append(reliability)
                self.P_reliability.append(P_reliability)
                self.cell_Fano_factor.append(Fano_factor)



                print('calculating Skaggs spatial info...')
                ## Skaggs spatial info
                skaggs_matrix=np.zeros((self.N_cells, self.N_shuffle+1))
                P_skaggs = np.zeros(self.N_cells)
                P_x=total_time/np.sum(total_time)
                for i_cell in range(self.N_cells):
                    for i_shuffle in range(self.N_shuffle+1):
                        mean_firing = rates[i_cell,i_shuffle]
                        lambda_x = rate_matrix[:,i_cell,i_shuffle]
                        i_nonzero = np.nonzero(lambda_x > 0)
                        skaggs_matrix[i_cell,i_shuffle] = np.sum(lambda_x[i_nonzero]*np.log2(lambda_x[i_nonzero]/mean_firing)*P_x[i_nonzero]) / mean_firing

                    shuffle_ecdf = skaggs_matrix[i_cell, 0:self.N_shuffle]
                    data_point = skaggs_matrix[i_cell, self.N_shuffle]
                    P_skaggs[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

                self.cell_skaggs.append(skaggs_matrix)
                self.P_skaggs.append(P_skaggs)
                 
                ## active laps/ all laps spks
                #use raw spks instead activity tensor
                print('calculating proportion of active laps...')
                active_laps = np.zeros((self.N_cells, N_laps_corr, self.N_shuffle+1))

                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps_abs = self.i_Laps_ImData[np.nonzero(icorrids == corridor)[0]]
                k = 0
                for i_lap in i_laps_abs:#y=ROI
                    for i_shuffle in range(self.N_shuffle+1):
                        act_cells = np.nonzero(np.amax(self.shuffle_ImLaps[i_lap].frames_spikes[:,:,i_shuffle], 1) > 25)[0] # cells * frames * shuffle
                        active_laps[act_cells, k, i_shuffle] = 1 # cells * laps * shuffle
                    k = k + 1

                active_laps_ratio = np.sum(active_laps, 1) / N_laps_corr
                self.cell_activelaps.append(active_laps_ratio)
                
                ## tuning specificity
                # print('calculating circular tuning specificity ...')
                # tuning_spec=np.zeros((self.N_cells, self.N_shuffle+1))
                # r=1
                # deg_d=(2*np.pi)/50
                # x=np.zeros(50)
                # y=np.zeros(50)
                # x_ext=np.zeros((50,len(i_laps)))
                # y_ext=np.zeros((50,len(i_laps)))
                
                # for i in range(50):
                #    x_i,y_i=pol2cart(r,deg_d*i)
                #    x[i]=round(x_i,5)
                #    y[i]=round(y_i,5)
                   
                # for i in range(len(i_laps)):
                #     x_ext[:,i]=x
                #     y_ext[:,i]=y
                    
                # for i_cell in range(self.N_cells):
                #     for i_shuffle in range(self.N_shuffle+1):
                #         magn_x=nan_divide(self.activity_tensor[:,i_cell,i_laps,i_shuffle]*x_ext, self.activity_tensor_time[:,i_laps], self.activity_tensor_time[:,i_laps] > 0)
                #         magn_y=nan_divide(self.activity_tensor[:,i_cell,i_laps,i_shuffle]*y_ext, self.activity_tensor_time[:,i_laps], self.activity_tensor_time[:,i_laps] > 0)
                #         X = np.nansum(magn_x)
                #         Y = np.nansum(magn_y)
                #         Z = np.nansum(np.sqrt(magn_x**2 + magn_y**2))
                #         tuning_spec[i_cell,i_shuffle]=np.sqrt(X**2+Y**2) / Z

                ## linear tuning specificity
                print('calculating linear tuning specificity ...')
                tuning_spec=np.zeros((self.N_cells, self.N_shuffle+1))
                P_tuning_specificity = np.zeros(self.N_cells)
                xbins = (np.arange(50) + 0.5) * self.corridor_length_cm / 50
                
                for i_cell in range(self.N_cells):
                    for i_shuffle in range(self.N_shuffle+1):
                        rr = np.copy(rate_matrix[:,i_cell,i_shuffle])
                        rr[rr < np.mean(rr)] = 0
                        Px = rr / np.sum(rr)
                        mu = np.sum(Px * xbins)
                        sigma = np.sqrt(np.sum(Px * xbins**2) - mu**2)
                        tuning_spec[i_cell,i_shuffle] = self.corridor_length_cm / sigma

                    shuffle_ecdf = tuning_spec[i_cell, 0:self.N_shuffle]
                    data_point = tuning_spec[i_cell, self.N_shuffle]
                    P_tuning_specificity[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

                self.cell_tuning_specificity.append(tuning_spec)
                self.P_tuning_specificity.append(P_tuning_specificity)

            # self.cell_rates = [] # a list, each element is a 3 x n_cells matrix with the average rate of the cells in the total corridor, pattern zone and reward zone
            self.cell_corridor_selectivity[0,:,:] = (self.cell_rates[0][0,:,:] - self.cell_rates[1][0,:,:]) / (self.cell_rates[0][0,:,:] + self.cell_rates[1][0,:,:])
            self.cell_corridor_selectivity[1,:,:] = (self.cell_rates[0][1,:,:] - self.cell_rates[1][1,:,:]) / (self.cell_rates[0][1,:,:] + self.cell_rates[1][1,:,:])
            self.cell_corridor_selectivity[2,:,:] = (self.cell_rates[0][2,:,:] - self.cell_rates[1][2,:,:]) / (self.cell_rates[0][2,:,:] + self.cell_rates[1][2,:,:])
            
            for i_region in range(3):
                for i_cell in range(self.N_cells):
                    shuffle_ecdf = self.cell_corridor_selectivity[i_region,i_cell, 0:self.N_shuffle]
                    data_point = self.cell_corridor_selectivity[i_region,i_cell, self.N_shuffle]
                    self.P_selectivity[i_region,i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)



    def plot_properties_shuffle(self, cellids=np.array([-1]), maxNcells=10):
        ## plot the following properties: reliability, specificity, active laps, Skaggs info, Fano_factor, corridor selectivity, reward selectivity, maze selectivity
        ## accepted place cells are shown in a different color
        ## we prepare violin-plots for the selected cells - given in cellids
        ## the max number of cells selected is currently 10
        Nmax = np.min((maxNcells, self.N_cells))
        if (cellids[0] == -1):
            iplot_cells = np.arange(Nmax)
            cellids = self.cellids[0:Nmax]
        else:
            cellids, i_valid_cells, iplot_cells = np.intersect1d(cellids, self.cellids, return_indices=True)
            if (iplot_cells.size > Nmax):
                iplot_cells = iplot_cells[0:Nmax]
        Ncells_to_plot = iplot_cells.size

        colormap = np.array(['C1','C2'])
        n_corridors=self.corridors.size-1#we don't want to plot corridor 0
        fig, ax = plt.subplots(n_corridors, 3, figsize=(10,5), sharex='all', sharey='col')
        plt.subplots_adjust(wspace=0.35, hspace=0.2)
        title_string = 'shuffling mode: ' + self.mode
        plt.title(title_string)

        for i in range(n_corridors):
            corridor=self.corridors[i+1]#always plot the specified corridor
            i_corridor = int(np.nonzero(self.corridors == corridor)[0]) - 1
            cols = colormap[self.accepted_PCs[i_corridor][iplot_cells]]

            ## reliability    
            data = np.transpose(self.cell_reliability[i_corridor][iplot_cells,:])
            ax[i,0].violinplot(data[0:self.N_shuffle,:])
            ax[i,0].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c=cols)
            ylab_string = 'corridor ' + str(corridor)
            xlab_string = ''
            title_string = ''
            if (i==0) :
                title_string = 'reliability'
            if (i==(n_corridors-1)):
                xlab_string = 'cells'
                ax[i,0].set_xticks(np.arange(Ncells_to_plot)+1)
                ax[i,0].set_xticklabels(cellids, rotation=90)
            ax[i,0].set_title(title_string)
            ax[i,0].set_ylabel(ylab_string)
            ax[i,0].set_xlabel(xlab_string)

            ## specificity
            data = np.transpose(self.cell_tuning_specificity[i_corridor][iplot_cells,:])
            ax[i,1].violinplot(data[0:self.N_shuffle,:])
            ax[i,1].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c=cols)
            ylab_string = ''
            xlab_string = ''
            title_string = ''
            if (i==0) :
                title_string = 'specificity'
            if (i==(n_corridors-1)):
                xlab_string = 'cells'
                ax[i,1].set_xticks(np.arange(Ncells_to_plot)+1)
                ax[i,1].set_xticklabels(cellids, rotation=90)
            ax[i,1].set_title(title_string)
            ax[i,1].set_ylabel(ylab_string)
            ax[i,1].set_xlabel(xlab_string)

            # ## active laps
            # data = np.transpose(self.cell_activelaps[i_corridor][iplot_cells,:])
            # ax[i,2].violinplot(data[0:self.N_shuffle,:])
            # ax[i,2].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c=cols)
            # ylab_string = ''
            # xlab_string = ''
            # title_string = ''
            # if (i==0) :
            #     title_string = 'active laps (%)'
            # if (i==(n_corridors-1)):
            #     xlab_string = 'cells'
            #     ax[i,2].set_xticks(np.arange(Ncells_to_plot)+1)
            #     ax[i,2].set_xticklabels(cellids, rotation=90)
            # ax[i,2].set_title(title_string)
            # ax[i,2].set_ylabel(ylab_string)
            # ax[i,2].set_xlabel(xlab_string)
    
            ## Skagg's info
            data = np.transpose(self.cell_skaggs[i_corridor][iplot_cells,:])
            ax[i,2].violinplot(data[0:self.N_shuffle,:])
            ax[i,2].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c=cols)
            ylab_string = ''
            xlab_string = ''
            title_string = ''
            if (i==0) :
                title_string = 'Skaggs info (bit/event) %'
            if (i==(n_corridors-1)):
                xlab_string = 'cells'
                ax[i,2].set_xticks(np.arange(Ncells_to_plot)+1)
                ax[i,2].set_xticklabels(cellids, rotation=90)
            ax[i,2].set_title(title_string)
            ax[i,2].set_ylabel(ylab_string)
            ax[i,2].set_xlabel(xlab_string)

            # ## Fano_factor
            # data = np.transpose(self.cell_Fano_factor[i_corridor][iplot_cells,:])
            # ax[i,4].violinplot(data[0:self.N_shuffle,:])
            # ax[i,4].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c=cols)
            # ylab_string = ''
            # xlab_string = ''
            # title_string = ''
            # if (i==0) :
            #     title_string = 'Fano factor'
            # if (i==(n_corridors-1)):
            #     xlab_string = 'cells'
            #     ax[i,4].set_xticks(np.arange(Ncells_to_plot)+1)
            #     ax[i,4].set_xticklabels(cellids, rotation=90)
            # ax[i,4].set_title(title_string)
            # ax[i,4].set_ylabel(ylab_string)
            # ax[i,4].set_xlabel(xlab_string)


        plt.show(block=False)


        ###########################
        ## plots of corridor selectivity

        fig, ax = plt.subplots(1, 3, figsize=(5,3), sharex='all', sharey='all')
        plt.subplots_adjust(wspace=0.35, hspace=0.2)

        ## corridor selectivity    
        data = np.transpose(self.cell_corridor_selectivity[0,iplot_cells,:])
        ax[0].violinplot(data[0:self.N_shuffle,:])
        ax[0].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C1')
        ax[0].set_title('corridor selectivity')
        ax[0].set_ylabel('selectivity index')
        ax[0].set_xlabel('cells')
        ax[0].set_xticks(np.arange(Ncells_to_plot)+1)
        ax[0].set_xticklabels(cellids, rotation=90)

        ## corridor selectivity    
        data = np.transpose(self.cell_corridor_selectivity[1,iplot_cells,:])
        ax[1].violinplot(data[0:self.N_shuffle,:])
        ax[1].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C2')
        ax[1].set_title('pattern selectivity')
        ax[1].set_ylabel('selectivity index')
        ax[1].set_xlabel('cells')
        ax[1].set_xticks(np.arange(Ncells_to_plot)+1)
        ax[1].set_xticklabels(cellids, rotation=90)

        ## corridor selectivity    
        data = np.transpose(self.cell_corridor_selectivity[2,iplot_cells,:])
        ax[2].violinplot(data[0:self.N_shuffle,:])
        ax[2].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C3')
        ax[2].set_title('reward selectivity')
        ax[2].set_ylabel('selectivity index')
        ax[2].set_xlabel('cells')
        ax[2].set_xticks(np.arange(Ncells_to_plot)+1)
        ax[2].set_xticklabels(cellids, rotation=90)

        plt.show(block=False)

    def Hainmuller_PCs_shuffle(self):
        ## ratemaps: similar to the activity tensor, the laps are sorted by the corridors
        self.candidate_PCs = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping

        ## we calculate the rate matrix for all corridors - we need to use the same colors for the images
        for corrid in np.unique(self.i_corridors):
            # select the laps in the corridor 
            # only laps with imaging data are selected - this will index the activity_tensor
            i_corrid = int(np.nonzero(self.corridors == corrid)[0] - 1)
            rate_matrix = self.ratemaps[i_corrid]
            
            candidate_cells = np.zeros((self.N_cells, self.N_shuffle+1))
            accepted_cells = np.zeros(self.N_cells)
            
            i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corrid)[0] 
            N_laps_corr = len(i_laps)
            act_tensor_1 = self.activity_tensor[:,:,i_laps,:] ## bin x cells x laps x shuffle; all activity in all laps in corridor i

            for i_cell in np.arange(rate_matrix.shape[1]):
                for i_shuffle in np.arange(rate_matrix.shape[2]):
                    rate_i = rate_matrix[:,i_cell,i_shuffle]

                    ### calculate the baseline, peak and threshold for each cell
                    ## Hainmuller: average of the lowest 25%; 
                    baseline = np.mean(np.sort(rate_i)[:12])
                    peak_rate = np.max(rate_i)
                    threshold = baseline + 0.25 * (peak_rate - baseline)

                    ## 1) find the longest contiguous region of above threshold for at least 3 bins...
                    placefield_start = np.nan
                    placefield_length = 0
                    candidate_start = 0
                    candidate_length = 0
                    for k in range(50):
                        if (rate_i[k] > threshold):
                            candidate_length = candidate_length + 1
                            if (candidate_length == 1):
                                candidate_start = k
                            elif ((candidate_length > 2) & (candidate_length > placefield_length)):
                                placefield_length = candidate_length
                                placefield_start = candidate_start
                        else:
                            candidate_length = 0

                    if (not(np.isnan(placefield_start))):
                        ##  2) with average rate at least 7x the average rate outside
                        index_infield = np.arange(placefield_start,(placefield_start+placefield_length))
                        index_outfield = np.setdiff1d(np.arange(50), index_infield)
                        rate_inField = np.mean(rate_i[index_infield])
                        rate_outField = np.mean(rate_i[index_outfield])


                        ## significant (total spike is larger than 0.6) transient in the field in at least 20% of the runs
                        lapsums = act_tensor_1[index_infield,i_cell,:,i_shuffle].sum(0) # only laps in this corridor

                        if ( ( (sum(lapsums > 0.6) / float(N_laps_corr)) > 0.2)  & ( (rate_inField / rate_outField) > 7) ):
                            # accepted_cells[i_cell] = 1
                            candidate_cells[i_cell, i_shuffle] = 1

                place_cell_P = np.sum(candidate_cells[i_cell, 0:self.N_shuffle]) / self.N_shuffle
                if ((candidate_cells[i_cell,self.N_shuffle]==1) & (place_cell_P < 0.05)):
                    accepted_cells[i_cell] = 1

            self.candidate_PCs.append(candidate_cells)
            self.accepted_PCs.append(accepted_cells.astype(int))
        
        for i_cell in np.arange(rate_matrix.shape[1]):
            for i_shuffle in np.arange(rate_matrix.shape[2]):
                self.cell_corridor_similarity[:,i_cell, i_shuffle] = scipy.stats.pearsonr(self.ratemaps[0][:,i_cell,i_shuffle], self.ratemaps[1][:,i_cell,i_shuffle])


class Shuffle_ImData:
    'common base class for shuffled laps'

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, lap_frames_spikes, lap_frames_pos, lap_frames_time, corridor_list, dt=0.01, speed_factor=106.5/3500, printout=False):
        self.name = name
        self.lap = lap

        self.correct = False
        self.raw_time = laptime
        self.raw_position = position
        self.lick_times = lick_times
        self.reward_times = reward_times
        self.corridor = corridor # the ID of the corridor in the given stage; This indexes the corridors in the vector called self.corridors
        self.corridor_list = corridor_list # the ID of the corridor in the given stage; This indexes the corridors in the vector called self.corridors
        self.mode = mode # 1 if all elements are recorded in 'Go' mode
        self.actions = actions
        self.speed_threshold = 5 ## cm / s 106 cm - 3500 roxels; roxel/s * 106.5/3500 = cm/s
        self.speed_factor = speed_factor

        self.zones = np.vstack([np.array(self.corridor_list.corridors[self.corridor].reward_zone_starts), np.array(self.corridor_list.corridors[self.corridor].reward_zone_ends)])
        self.n_zones = np.shape(self.zones)[1]
        self.preZoneRate = [None, None] # only if 1 lick zone; Compare the 210 roxels just before the zone with the preceeding 210 

        self.dt = 0.01 # resampling frequency = 100 Hz
        # approximate frame period for imaging - 0.033602467
        # only use it to convert spikes to rates!
        self.dt_imaging = 0.033602467

        self.frames_spikes = lap_frames_spikes
        self.frames_pos = lap_frames_pos
        self.frames_time = lap_frames_time
        self.min_N_frames = 50 # we analyse imaging data if there are at least 50 frames in the lap
        self.n_cells = 1 # we still create the same np arrays even if there are no cells
        self.n_shuffle = 1 
        if (not(np.isnan(self.frames_time).any())): # we have real data
            if (len(self.frames_time) > self.min_N_frames): # we have more than 50 frames
                self.n_cells = self.frames_spikes.shape[0]
                self.n_shuffle = self.frames_spikes.shape[2]
            
        ####################################################################
        ## resample time and position with a uniform 100 Hz
        self.bincenters = np.arange(0, 3500, 70) + 70 / 2.0
        
        if (len(self.raw_time) > 2):
            F = interp1d(self.raw_time,self.raw_position)
            start_time = np.ceil(self.raw_time.min()/self.dt)*self.dt
            end_time = np.floor(self.raw_time.max()/self.dt)*self.dt
            Ntimes = int(round((end_time - start_time) / self.dt)) + 1
            self.laptime = np.linspace(start_time, end_time, Ntimes)
            ppos = F(self.laptime)
    
            self.lick_position = F(self.lick_times)
            self.reward_position = F(self.reward_times)

            # correct: if rewarded
            if (len(self.reward_times) > 0):
                self.correct = True
            # correct: if no licking in the zone
            lick_in_zone = np.nonzero((self.lick_position > self.zones[0] * 3500) & (self.lick_position <= self.zones[1] * 3500 + 1))[0]
            if (self.corridor_list.corridors[self.corridor].reward == 'Left'):
                if (len(lick_in_zone) == 0):
                    self.correct = True
            else :
                if ((len(lick_in_zone) == 0) & self.correct):
                    print ('Warning: rewarded lap with no lick in zone! lap number:' + str(self.lap))

            self.smooth_position = ppos
            
            ## calculate the smoothed speed 
            # self.speed = np.diff(np.hstack([self.smooth_position[0], self.smooth_position])) / self.dt # roxel [=rotational pixel] / s       
            speed = np.diff(self.smooth_position) * self.speed_factor / self.dt # cm / s       
            speed_first = 2 * speed[0] - speed[1] # linear extrapolation: x1 - (x2 - x1)
            self.speed = np.hstack([speed_first, speed])

            ## calculate the speed during the frames
            if (not(np.isnan(self.frames_time).any())): # we real data
                if (len(self.frames_time) > self.min_N_frames): # we have more than 50 frames
                    FF = interp1d(self.laptime,self.speed, fill_value="extrapolate") 
                    self.frames_speed = FF(self.frames_time)

            ####################################################################
            ## calculate the lick-rate and the average speed versus location    
            bin_counts = np.zeros(50)
            for pos in self.smooth_position:
                bin_number = int(pos // 70)
                bin_counts[bin_number] += 1
            self.T_pos = bin_counts * self.dt

            lbin_counts = np.zeros(50)
            for lpos in self.lick_position:
                lbin_number = int(lpos // 70)
                lbin_counts[lbin_number] += 1
            self.N_licks = lbin_counts
            self.lick_rate = nan_divide(self.N_licks, self.T_pos, where=(self.T_pos > 0.025))
    
            total_speed = np.zeros(50)
            for i_frame in range(len(self.smooth_position)):
                ind_bin = int(self.smooth_position[i_frame] // 70)
                total_speed[ind_bin] = total_speed[ind_bin] + self.speed[i_frame]
            total_speed = total_speed * self.dt
            self.ave_speed = nan_divide(total_speed, self.T_pos, where=(self.T_pos > 0.025))
        
            ####################################################################
            ## calculate the cell activations (spike rate) as a function of position
            self.spks_pos = np.zeros((self.n_cells, 50, self.n_shuffle)) # sum of spike counts measured at a given position
            self.event_rate = np.zeros((self.n_cells, 50, self.n_shuffle)) # spike rate 
            # self.dF_F_pos = np.zeros((self.n_cells, 50))
            if (not(np.isnan(self.frames_time).any())): # we real data
                if (len(self.frames_time) > self.min_N_frames): # we have more than 50 frames
                    for i_frame in range(len(self.frames_pos)):
                        ind_bin = int(self.frames_pos[i_frame] // 70)
                        if (self.frames_speed[ind_bin] > self.speed_threshold):
                            ### we need to multiply the values with dt_imaging as this converts probilities to expected counts
                            self.spks_pos[:,ind_bin,:] = self.spks_pos[:,ind_bin,:] + self.frames_spikes[:,i_frame,:] * self.dt_imaging
                    for ind_bin in range(50):
                        if (self.T_pos[ind_bin] > 0): # otherwise the rate will remain 0
                            self.event_rate[:,ind_bin,:] = self.spks_pos[:,ind_bin,:] / self.T_pos[ind_bin]


        else:
            self.lick_position = lick_times
            self.reward_position = reward_times
            self.smooth_position = position
            self.speed = np.zeros(len(position))
            self.T_pos = np.zeros(50)
            self.N_licks = np.zeros(50)
            self.ave_speed = np.zeros(50)
            self.lick_rate = np.zeros(50)
                

    # def plot_tx(self, fluo=False, th=25):
    #     colmap = plt.cm.get_cmap('jet')   
    #     colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
    #     fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

    #     ## first, plot position versus time
    #     ax_top.plot(self.laptime, self.smooth_position, c=colmap(50))
    #     ax_top.plot(self.raw_time, self.raw_position, c=colmap(90))

    #     ax_top.scatter(self.lick_times, np.repeat(self.smooth_position.min(), len(self.lick_times)), marker="|", s=100, c=colmap(180))
    #     ax_top.scatter(self.reward_times, np.repeat(self.smooth_position.min()+100, len(self.reward_times)), marker="|", s=100, c=colmap(230))
    #     ax_top.set_ylabel('position')
    #     ax_top.set_xlabel('time (s)')
    #     plot_title = 'Mouse: ' + self.name + ' position in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
    #     ax_top.set_title(plot_title)
    #     ax_top.set_ylim(0, 3500)

    #     ## next, plot dF/F versus time (or spikes)
    #     if (self.n_cells > 1):
    #         # act_cells = np.nonzero(np.amax(self.frames_dF_F, 1) > th)[0]
    #         act_cells = np.nonzero(np.amax(self.frames_spikes, 1) > th)[0]
    #         max_range = np.nanmax(self.event_rate)
    #         for i in range(self.n_cells):
    #         # for i in (252, 258, 275):
    #             if (fluo & (i in act_cells)):
    #                 ax_bottom.plot(self.frames_time, self.frames_dF_F[i,:] + i, alpha=0.5, c=colmap(np.remainder(i, 255)))
    #             events = self.frames_spikes[i,:]
    #             events = 50 * events / max_range
    #             ii_events = np.nonzero(events)[0]
    #             ax_bottom.scatter(self.frames_time[ii_events], np.ones(len(ii_events)) * i, s=events[ii_events], cmap=colmap, c=(np.ones(len(ii_events)) * np.remainder(i, 255)), norm=colnorm)

    #         ylab_string = 'dF_F, spikes (max: ' + str(np.round(max_range, 1)) +  ' )'
    #         ax_bottom.set_ylabel(ylab_string)
    #         ax_bottom.set_xlabel('time (s)')
    #         plot_title = 'dF/F of all neurons  in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
    #         ax_bottom.set_title(plot_title)
    #         ax_bottom.set_ylim(0, self.n_cells+5)
    #         # ax_bottom.set_ylim(250, 280)

    #     plt.show(block=False)
       

    # def plot_xv(self):
    #     colmap = plt.cm.get_cmap('jet')   
    #     colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)

    #     fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), sharex=True,  gridspec_kw={'height_ratios': [1, 3]})
    #     ax_top.plot(self.smooth_position, self.speed, c=colmap(80))
    #     ax_top.step(self.bincenters, self.ave_speed, where='mid', c=colmap(30))
    #     ax_top.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, c=colmap(180))
    #     ax_top.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, c=colmap(230))
    #     ax_top.set_ylabel('speed (cm/s)')
    #     ax_top.set_ylim([min(0, self.speed.min()), max(self.speed.max(), 30)])
    #     ax_top.set_xlabel('position')
    #     plot_title = 'Mouse: ' + self.name + ' speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
    #     ax_top.set_title(plot_title)


    #     bottom, top = ax_top.get_ylim()
    #     left = self.zones[0,0] * 3500
    #     right = self.zones[1,0] * 3500

    #     polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
    #     ax_top.add_patch(polygon)
    #     if (self.n_zones > 1):
    #         for i in range(1, np.shape(self.zones)[1]):
    #             left = self.zones[0,i] * 3500
    #             right = self.zones[1,i] * 3500
    #             polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
    #             ax_top.add_patch(polygon)

    #     ax2 = ax_top.twinx()
    #     ax2.step(self.bincenters, self.lick_rate, where='mid', c=colmap(200), linewidth=1)
    #     ax2.set_ylabel('lick rate (lick/s)', color=colmap(200))
    #     ax2.tick_params(axis='y', labelcolor=colmap(200))
    #     ax2.set_ylim([-1,max(2*np.nanmax(self.lick_rate), 20)])

    #     ## next, plot event rates versus space
    #     if (self.n_cells > 1):
    #         max_range = np.nanmax(self.event_rate)
    #         # for i in np.arange(250, 280):
    #         for i in range(self.n_cells):
    #             events = self.event_rate[i,:]
    #             events = 50 * events / max_range
    #             ii_events = np.nonzero(events)[0]
    #             ax_bottom.scatter(self.bincenters[ii_events], np.ones(len(ii_events)) * i, s=events[ii_events], cmap=colmap, c=(np.ones(len(ii_events)) * np.remainder(i, 255)), norm=colnorm)

    #         ax_bottom.set_ylabel('event rate')
    #         ax_bottom.set_xlabel('position')
    #         plot_title = 'event rate of all neurons  in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
    #         ax_bottom.set_title(plot_title)
    #         # ax_bottom.set_ylim(250, 280)
    #         ax_bottom.set_ylim(0, self.n_cells)

    #     plt.show(block=False)       


    #     colmap = plt.cm.get_cmap('jet')
    #     colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
    #     x = np.random.rand(4)
    #     y = np.random.rand(4)
    #     area = (np.abs(x/max(x))*30)**2
    #     colors = 232


