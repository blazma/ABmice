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

## a function to break a time series into Nbreak sections each at least Lmin long.
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
    def __init__(self, datapath, date_time, name, task, stage, raw_spikes, frame_times, frame_pos, frame_laps, N_shuffle=1000, cellids=np.array([-1]), mode='random', batchsize=None, selected_laps=None, speed_threshold=5, randseed=123, elfiz=False):
        ###########################################
        ## setting basic parameters for the session
        ###########################################

        self.datapath = datapath
        self.date_time = date_time
        self.name = name
        self.task = task

        self.stage = stage
        self.stages = []
        self.randseed = randseed
        self.selected_laps = selected_laps
        self.speed_threshold = speed_threshold
        self.elfiz = elfiz

        stagefilename = datapath + self.task + '_stages.pkl'
        input_file = open(stagefilename, 'rb')
        if version_info.major == 2:
            self.stage_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.stage_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        corridorfilename = datapath + self.task + '_corridors.pkl'
        input_file = open(corridorfilename, 'rb')
        if version_info.major == 2:
            self.corridor_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.corridor_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        self.all_corridors = np.hstack([0, np.array(self.stage_list.stages[self.stage].corridors)])# we always add corridor 0 - that is the grey zone

        ## in certain tasks, the same corridor may appear multiple times in different substages
        ## Labview uses different indexes for corridors in different substages, therefore 
        ## we need to keep this corridor in the list self.corridors for running self.get_lapdata()
        ## but we should remove the redundancy after the data is loaded

        self.last_zone_start = 0
        self.last_zone_end = 0
        for i_corridor in self.all_corridors:
            if (i_corridor > 0):
                if (max(self.corridor_list.corridors[i_corridor].reward_zone_starts) > self.last_zone_start):
                    self.last_zone_start = max(self.corridor_list.corridors[i_corridor].reward_zone_starts)
                if (max(self.corridor_list.corridors[i_corridor].reward_zone_ends) > self.last_zone_end):
                    self.last_zone_end = max(self.corridor_list.corridors[i_corridor].reward_zone_ends)

        self.speed_factor = 106.5 / 3500.0 ## constant to convert distance from pixel to cm
        self.corridor_length_roxel = (self.corridor_list.corridors[self.all_corridors[1]].length - 1024.0) / (7168.0 - 1024.0) * 3500
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor # cm
        self.N_pos_bins = int(np.round(self.corridor_length_roxel / 70))


        ###########################################
        ## imaging data - imaging data and time axis is given as an argument - we don't need to reload it
        ###########################################
        if (elfiz == True):
            ## we downsample it to 50 Hz for shuffling - this is not going to affect the spatial tuning too much, but use much less memory...
            N_frames = raw_spikes.shape[1]
            N_frames_new = int(N_frames / 100)
            Nmax = N_frames_new * 100
            self.dt_imaging = 0.02
            self.raw_spikes = np.sum(raw_spikes[0,0:Nmax].reshape(N_frames_new, 100), axis=1).reshape(1, N_frames_new)
            self.frame_times = np.mean(frame_times[0:Nmax].reshape(N_frames_new, 100), axis=1)
            self.frame_pos = np.mean(frame_pos[0:Nmax].reshape(N_frames_new, 100), axis=1)
            self.frame_laps = np.mean(frame_laps[0:Nmax].reshape(N_frames_new, 100), axis=1)
        else :
            self.raw_spikes = raw_spikes
            self.frame_times = frame_times
            self.frame_pos = frame_pos
            self.frame_laps = frame_laps
            self.dt_imaging = 0.033602467 # s - 33 Hz

        self.N_cells = self.raw_spikes.shape[0]
        if (cellids.size != self.N_cells):
            print('Length of cellids do not match the number of spike trains provided! We stop.')
            return
        if (cellids[0] == -1):
            print('Cellids should be provided for shuffle analysis! We stop.')
            return
        self.cellids = cellids

        self.N_frames = self.raw_spikes.shape[1]
        self.N_shuffle = N_shuffle
        self.mode = mode # random: totally randomize the spike times; shift: circularly shift spike times 

          
        ###########################################
        ## setting up minibatches 
        ###########################################
        ## shuffling in minibatches - to save memory when running high number of shuffles with a lots of cells
        ## defining containers for the P-values
        ## still not super efficient as we load the behavior data separately for each minibatch...
        if batchsize is None:
            self.batchsize = self.N_cells
        else:
            self.batchsize = batchsize

        self.ratemaps = [] # a list, each element is an array space x neurons being the ratemap of the cells in a given corridor

        self.cell_reliability = [] # a list, each element is a matrix with the reliability of the shuffles in a corridor
        self.P_reliability = [] # a list, each element is a vector with the P value estimated from shuffle control - P(reliability > measured)

        self.cell_skaggs=[] # a list, each element is a matrix with the skaggs93 spatial info of the shuffles in a corridor
        self.P_skaggs=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(Skaggs-info > measured)

        self.cell_tuning_specificity=[] # a list, each element is a matrix with the tuning specificity of the shuffles in a corridor
        self.P_tuning_specificity=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(specificity > measured)

        self.accepted_PCs = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018

        self.cell_corridor_selectivity = np.zeros([2, self.N_cells, self.N_shuffle]) # a matrix with the selectivity index of the cells. Second row indicates the corridor with the highers rate.
        self.P_corridor_selectivity = np.zeros([self.N_cells])

        self.cell_corridor_similarity = np.zeros([self.N_cells, self.N_shuffle]) # a vector with the similarity index of the cells.
        self.P_corridor_similarity = np.zeros([self.N_cells])

        ##################################################
        ## shuffling the spikes data
        ##################################################

        i_start = 0
        i_end = min(i_start + self.batchsize, self.N_cells)
        if (i_end == (self.N_cells - 1)): # the last minibatch would contain only 1 cell
            i_end = i_end + 1
        i_minibatch = 0

        while (i_start < self.N_cells):
            batchsize = i_end - i_start # the last cycle in the loop will have a different size...
            batch_ids = np.arange(i_start, i_end)
            print('calculating minibatch ' + str(i_minibatch) + ', batch length: ', str(batchsize))

            self.shuffle_spikes = np.zeros((batchsize, self.N_frames, self.N_shuffle+1))
            self.shuffle_spikes[:,:,self.N_shuffle] = self.raw_spikes[batch_ids,:] # the last element is the real data ...

            if (self.mode == 'shift'):
                for i_shuffle in range(self.N_shuffle):
                    spks = np.zeros_like(self.raw_spikes[batch_ids,:])
                    ## we break up the array into 6 pieces of at least 500 frames, permuting them and circularly shifting by at least 500 frames
                    Nbreak = 5
                    sections = breakpoints(Nframes=self.N_frames, Lmin=500, Nbreak=Nbreak)
                    order = np.random.permutation(Nbreak + 1)
                    k = 0
                    for j in range(Nbreak+1):
                        i_section = order[j]
                        spks[:,k:(k+int(sections[1,i_section]))] = self.raw_spikes[batch_ids,int(sections[0,i_section]):int(sections[0,i_section]+sections[1,i_section])]
                        k = k + int(sections[1,i_section])

                    n_roll = np.random.randint((self.N_frames - 1000)) + 500
                    spks_rolled = np.roll(spks, n_roll, axis=1)
                    self.shuffle_spikes[:,:,i_shuffle] = spks_rolled
            else:
                if (self.mode != 'random'):
                    print ('Warning: shuffling mode must be either random or shift. We will use random.')
                for i_shuffle in range(self.N_shuffle):
                    spks = np.copy(self.raw_spikes[batch_ids,:])
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

            self.get_lapdata_shuffle(self.datapath, self.date_time, self.name, self.task, selected_laps=self.selected_laps) # collects all behavioral and imaging data and sort it into laps, storing each in a Lap_ImData object

            ## in certain tasks, the same corridor may appear multiple times in different substages
            ## we need to keep this corridor in the list self.all_corridors for running self.get_lapdata()
            ## but we should remove the redundancy after the data is loaded
            self.all_corridors = np.unique(self.all_corridors)
            self.N_all_corridors = len(self.all_corridors)

            ## only analyse corridors with at least 3 laps 
            # - the data still remains in the ImLaps list and will appear in the activity tensor!
            #   but the corridor will not 
            #   we also do NOT include corridor 0 here
            if (self.N_all_corridors > 1):
                N_laps_corr = np.zeros(self.N_all_corridors) 
                for i_corridor in np.arange(self.N_all_corridors):
                    corridor = self.all_corridors[i_corridor]
                    N_laps_corr[i_corridor] = sum(self.i_corridors == corridor)

                self.corridors = self.all_corridors[np.where(N_laps_corr > 3)[0]]
                self.N_corridors = len(self.corridors)

            else :
                self.corridors = np.setdiff1d(self.all_corridors, 0)
                self.N_corridors = len(self.corridors)

            self.N_ImLaps = len(self.i_Laps_ImData)

            self.raw_activity_tensor = np.zeros((self.N_pos_bins, batchsize, self.N_ImLaps, self.N_shuffle+1)) # a tensor with space x neurons x trials x shuffle containing the spikes
            self.raw_activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # a tensor with space x trials containing the time spent at each location in each lap
            self.activity_tensor = np.zeros((self.N_pos_bins, batchsize, self.N_ImLaps, self.N_shuffle+1)) # same as the activity tensor spatially smoothed
            self.activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # same as the activity_tensor_time spatially smoothed
            self.combine_lapdata_shuffle() ## fills in the cell_activity tensor


            self.cell_rates_batch = [] # a list, each element is a 1 x n_cells matrix with the average rate of the cells in the total corridor
            if (self.task == 'contingency_learning'):
                self.cell_pattern_rates_batch = []
            self.cell_activelaps_batch=[] # a list, each element is a matrix with the % of significantly spiking laps of the shuffles in a corridor
            self.cell_Fano_factor_batch = [] # a list, each element is a matrix with the reliability of the shuffles in a corridor

            self.cell_reliability_batch = [] # a list, each element is a matrix with the reliability of the shuffles in a corridor
            self.cell_skaggs_batch=[] # a list, each element is a matrix with the skaggs93 spatial info of the shuffles in a corridor
            self.cell_tuning_specificity_batch=[] # a list, each element is a matrix with the tuning specificity of the shuffles in a corridor

            self.P_reliability_batch = [] # a list, each element is a vector with the P value estimated from shuffle control - P(reliability > measured)
            self.P_skaggs_batch=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(Skaggs-info > measured)
            self.P_tuning_specificity_batch=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(specificity > measured)

            self.cell_corridor_selectivity_batch = np.zeros([batchsize, self.N_shuffle]) # a matrix with the selectivity index of the cells
            self.P_corridor_selectivity_batch = np.zeros([batchsize])
            if (self.task == 'contingency_learning'):            
                self.P_pattern_selectivity_batch = np.zeros([4, batchsize])

            self.ratemaps_batch = [] # a list, each element is an array space x neurons being the ratemap of the cells in a given corridor
            self.cell_corridor_similarity_batch = np.zeros([batchsize, self.N_shuffle]) # a vector with the similarity index of the cells.
            self.P_corridor_similarity_batch = np.zeros([batchsize])

            self.calculate_properties_shuffle()

            self.candidate_PCs_batch = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
            self.accepted_PCs_batch = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping
            self.Hainmuller_PCs_shuffle()

            if (i_minibatch == 0):
                self.cell_reliability = self.cell_reliability_batch
                self.P_reliability = self.P_reliability_batch
                self.cell_skaggs = self.cell_skaggs_batch
                self.P_skaggs = self.P_skaggs_batch
                self.cell_tuning_specificity = self.cell_tuning_specificity_batch
                self.P_tuning_specificity = self.P_tuning_specificity_batch
                self.accepted_PCs = self.accepted_PCs_batch

                self.cell_corridor_selectivity = self.cell_corridor_selectivity_batch  
                self.P_corridor_selectivity = self.P_corridor_selectivity_batch 
                if (self.task == 'contingency_learning'):            
                    self.cell_pattern_selectivity = self.cell_pattern_selectivity_batch
                    self.P_pattern_selectivity = self.P_pattern_selectivity_batch

                self.ratemaps = self.ratemaps_batch
                self.cell_corridor_similarity = self.cell_corridor_similarity_batch 
                self.P_corridor_similarity = self.P_corridor_similarity_batch 

            else:
                for i_cor in range(self.N_corridors):
                    print(np.shape(self.cell_reliability[i_cor]), np.shape(self.cell_reliability_batch[i_cor]))
                    self.cell_reliability[i_cor] = np.vstack((self.cell_reliability[i_cor], self.cell_reliability_batch[i_cor]))
                    self.P_reliability[i_cor] = np.hstack((self.P_reliability[i_cor], self.P_reliability_batch[i_cor]))
                    self.cell_skaggs[i_cor] = np.vstack((self.cell_skaggs[i_cor], self.cell_skaggs_batch[i_cor]))
                    self.P_skaggs[i_cor] = np.hstack((self.P_skaggs[i_cor], self.P_skaggs_batch[i_cor]))
                    self.cell_tuning_specificity[i_cor] = np.vstack((self.cell_tuning_specificity[i_cor], self.cell_tuning_specificity_batch[i_cor]))
                    self.P_tuning_specificity[i_cor] = np.hstack((self.P_tuning_specificity[i_cor], self.P_tuning_specificity_batch[i_cor]))
                    self.accepted_PCs[i_cor] = np.hstack((self.accepted_PCs[i_cor], self.accepted_PCs_batch[i_cor]))
                    self.ratemaps[i_cor] = np.concatenate((self.ratemaps[i_cor], self.ratemaps_batch[i_cor]), axis=1)

                # matrix, N x M
                self.cell_corridor_selectivity = np.concatenate((self.cell_corridor_selectivity, self.cell_corridor_selectivity_batch))
                self.P_corridor_selectivity = np.concatenate((self.P_corridor_selectivity, self.P_corridor_selectivity_batch))
                if (self.task == 'contingency_learning'):
                    # 4 x N x M = N_corridor x 4 x N_cell x N_shuffle            
                    self.cell_pattern_selectivity = np.concatenate((self.cell_pattern_selectivity, self.cell_pattern_selectivity_batch), axis=1)
                    # np.zeros([4, batchsize])
                    self.P_pattern_selectivity = np.concatenate((self.P_pattern_selectivity, self.P_pattern_selectivity_batch), axis=1)

                self.cell_corridor_similarity = np.concatenate((self.cell_corridor_similarity, self.cell_corridor_similarity_batch)) 
                self.P_corridor_similarity = np.concatenate((self.P_corridor_similarity, self.P_corridor_similarity_batch))

            i_start = i_end
            i_end = min(i_start + batchsize, self.N_cells)
            if (i_end == (self.N_cells - 1)): # the last minibatch would contain only 1 cell
                i_end = i_end + 1
            i_minibatch = i_minibatch + 1


    def get_lapdata_shuffle(self, datapath, date_time, name, task, selected_laps=None):

        time_array=[]
        lap_array=[]
        maze_array=[]
        position_array=[]
        mode_array=[]
        lick_array=[]
        action=[]
        substage=[]

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
            substage.append(str(line[17]))

        laptime = np.array(time_array)
        lap = np.array(lap_array)
        sstage = np.array(substage)

        pos = np.array(position_array)
        lick = np.array(lick_array)
        maze = np.array(maze_array)
        mode = np.array(mode_array)
        N_0lap = 0 # Counting the non-valid laps

        #################################################
        ## position, and lap info has been already added to the imaging frames
        #################################################
        i_ImData = [] # index of laps with imaging data
        i_corrids = [] # ID of corridor for the current lap

        ### include only a subset of laps
        if selected_laps is None:
            all_laps_set = np.unique(lap)
        else:
            all_laps_set = np.intersect1d(np.unique(lap), selected_laps)

        self.n_laps = 0


        for i_lap in all_laps_set:
            y = lap == i_lap # index for the current lap

            mode_lap = np.prod(mode[y]) # 1 if all elements are recorded in 'Go' mode

            maze_lap = np.unique(maze[y])
            if (len(maze_lap) == 1):
                corridor = self.all_corridors[int(maze_lap)] # the maze_lap is the index of the available corridors in the given stage
            else:
                corridor = -1

            sstage_lap = np.unique(sstage[y])
            if (len(sstage_lap) > 1):
                print('More than one substage in lap ', self.n_laps)
                corridor = -2

            if (sum(y) < self.N_pos_bins):
                print('Very short lap found, we have total ', sum(y), 'datapoints recorded by the ExpStateMachine in lap ', self.n_laps)
                corridor = -3

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
    
                ## detecting invalid laps - terminated before the animal could receive reward
                valid_lap = False
                if (len(t_reward) > 0): # lap is valid if the animal got reward
                    valid_lap = True
                if (max(pos_lap) > (self.corridor_length_roxel * self.last_zone_end)): # # lap is valid if the animal left the last reward zone
                    valid_lap = True
                if (valid_lap == False):
                    mode_lap = 0

                actions = []
                for j in range(len(action_lap)):
                    if not((action_lap[j]) in ['No', 'TrialReward']):
                        actions.append([t_lap[j], action_lap[j]])

                ### include only a subset of laps
                add_lap = True
                if (mode_lap == 0):
                    add_lap = False

                ### imaging data    
                iframes = np.where(self.frame_laps == i_lap)[0]
                # print(self.n_laps, len(iframes), i_lap)

                if ((len(iframes) > self.N_pos_bins) & (add_lap == True)): # there is imaging data belonging to this lap...
                    # print('imaging data found', min(iframes), max(iframes))
                    lap_frames_spikes = self.shuffle_spikes[:,iframes,:]
                    lap_frames_time = self.frame_times[iframes]
                    lap_frames_pos = self.frame_pos[iframes]
                    i_ImData.append(self.n_laps)
                else:
                    # print('no imaging data found in lap', self.n_laps)
                    lap_frames_spikes = np.nan
                    lap_frames_time = np.nan
                    lap_frames_pos = np.nan 
                    
                # sessions.append(Lap_Data(name, i, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions))
                self.shuffle_ImLaps.append(Shuffle_ImData(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, lap_frames_spikes, lap_frames_pos, lap_frames_time, self.corridor_list, speed_threshold=self.speed_threshold, elfiz=self.elfiz, dt_imaging=self.dt_imaging))
                self.n_laps = self.n_laps + 1
            else:
                N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1) - we do not do anythin with this...

        self.i_Laps_ImData = np.array(i_ImData) # index of laps with imaging data
        self.i_corridors = np.array(i_corrids) # ID of corridor for the current lap
        # self.cell_activity = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials

    def combine_lapdata_shuffle(self): ## fills in the cell_activity tensor
        # self.raw_activity_tensor = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials containing the spikes
        # self.raw_activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # a tensor with space x  trials containing the time spent at each location in each lap
        valid_lap = np.zeros(len(self.i_Laps_ImData))
        k_lap = 0
        for i_lap in self.i_Laps_ImData:
            if (self.shuffle_ImLaps[i_lap].n_cells > 0):
                valid_lap[k_lap] = 1
                self.raw_activity_tensor[:,:,k_lap,:] = np.moveaxis(self.shuffle_ImLaps[i_lap].spks_pos, 1, 0)
                self.raw_activity_tensor_time[:,k_lap] = self.shuffle_ImLaps[i_lap].T_pos_fast
            k_lap = k_lap + 1

        ## smoothing - average of the 3 neighbouring bins
        self.activity_tensor[0,:,:,:] = (self.raw_activity_tensor[0,:,:,:] + self.raw_activity_tensor[1,:,:,:]) / 2
        self.activity_tensor[-1,:,:,:] = (self.raw_activity_tensor[-1,:,:,:] + self.raw_activity_tensor[-1,:,:,:]) / 2
        self.activity_tensor_time[0,:] = (self.raw_activity_tensor_time[0,:] + self.raw_activity_tensor_time[1,:]) / 2
        self.activity_tensor_time[-1,:] = (self.raw_activity_tensor_time[-1,:] + self.raw_activity_tensor_time[-1,:]) / 2
        for i_bin in np.arange(1, self.N_pos_bins-1):
            self.activity_tensor[i_bin,:,:,:] = np.average(self.raw_activity_tensor[(i_bin-1):(i_bin+2),:,:,:], axis=0)
            self.activity_tensor_time[i_bin,:] = np.average(self.raw_activity_tensor_time[(i_bin-1):(i_bin+2),:], axis=0)

        i_valid_laps = np.nonzero(valid_lap)[0]
        self.i_Laps_ImData = self.i_Laps_ImData[i_valid_laps]
        self.raw_activity_tensor = self.raw_activity_tensor[:,:,i_valid_laps,:]
        self.raw_activity_tensor_time = self.raw_activity_tensor_time[:,i_valid_laps]
        self.activity_tensor = self.activity_tensor[:,:,i_valid_laps,:]
        self.activity_tensor_time = self.activity_tensor_time[:,i_valid_laps]



    def calculate_properties_shuffle(self):
        self.cell_reliability_batch = []
        self.cell_Fano_factor_batch = []
        self.cell_skaggs_batch=[]

        self.cell_activelaps_batch=[]
        self.cell_tuning_specificity_batch=[]

        self.cell_rates_batch = [] # we do not append if it already exists...
        if (self.task == 'contingency_learning'):
            self.cell_pattern_rates_batch = []
        self.ratemaps_batch = [] # a list, each element is an array space x neurons being the ratemaps of the cells in a given corridor

        self.P_reliability_batch = [] # a list, each element is a vector with the P value estimated from shuffle control - P(reliability > measured)
        self.P_skaggs_batch=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(Skaggs-info > measured)
        self.P_tuning_specificity_batch=[] # a list, each element is a vector with the the P value estimated from shuffle control - P(specificity > measured)

        minibatchsize = self.activity_tensor.shape[1]
        self.cell_corridor_selectivity_batch = np.zeros([minibatchsize, self.N_shuffle]) # a matrix with the selectivity index of the cells. Second row indicates the corridor with the highers rate.
        self.P_corridor_selectivity_batch = np.zeros([minibatchsize])

        self.cell_corridor_similarity_batch = np.zeros([minibatchsize, self.N_shuffle]) # a vector with the similarity index of the cells.
        self.P_corridor_similarity_batch = np.zeros([minibatchsize])

        if (self.N_corridors > 0):
            for i_corridor in np.arange(self.N_corridors):
                corridor = self.corridors[i_corridor]
                if (sum(self.i_corridors[self.i_Laps_ImData] == corridor) > 5):
                    # select the laps in the corridor 
                    # only laps with imaging data are selected - this will index the activity_tensor
                    i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor)[0] 
                    N_laps_corr = len(i_laps)

                    time_matrix_1 = self.activity_tensor_time[:,i_laps]
                    total_time = np.sum(time_matrix_1, axis=1) # bins x cells -> bins; time spent in each location

                    act_tensor_1 = self.activity_tensor[:,:,i_laps,:] ## bin x cells x laps x shuffle; all activity in all laps in corridor i
                    total_spikes = np.sum(act_tensor_1, axis=2) ##  bin x cells x shuffle; total activity of the cells in corridor i

                    rate_matrix = np.zeros_like(total_spikes) ## event rate 
                    
                    for i_cell in range(minibatchsize):
                        for i_shuffle in range(self.N_shuffle+1):
                            rate_matrix[:,i_cell,i_shuffle] = total_spikes[:,i_cell,i_shuffle] / total_time

                    self.ratemaps_batch.append(rate_matrix)

                    print('calculating rate, reliability and Fano factor...')

                    ## average firing rate
                    rates = np.sum(total_spikes, axis=0) / np.sum(total_time) # cells x shuffle
                    self.cell_rates_batch.append(rates)

                    if (self.task == 'contingency_learning'):
                        rates_pattern1 = np.sum(total_spikes[0:14,:,:], axis=0) / np.sum(total_time[0:14]) # N x M
                        rates_pattern2 = np.sum(total_spikes[14:28,:,:], axis=0) / np.sum(total_time[14:28])
                        rates_pattern3 = np.sum(total_spikes[28:42,:,:], axis=0) / np.sum(total_time[28:42])
                        rates_reward = np.sum(total_spikes[42:47,:,:], axis=0) / np.sum(total_time[42:47])
                        self.cell_pattern_rates_batch.append(np.stack([rates_pattern1, rates_pattern2, rates_pattern3, rates_reward])) # 4 x N x M


                    ## reliability and Fano factor
                    reliability = np.zeros((minibatchsize, self.N_shuffle+1))
                    P_reliability_batch = np.zeros(minibatchsize)
                    Fano_factor = np.zeros((minibatchsize, self.N_shuffle+1))
                    for i_cell in range(minibatchsize):
                        for i_shuffle in range(self.N_shuffle+1):
                            laps_rates = nan_divide(act_tensor_1[:,i_cell,:,i_shuffle], time_matrix_1, where=(time_matrix_1 > 0.025))
                            corrs_cell = vcorrcoef(np.transpose(laps_rates), rate_matrix[:,i_cell,i_shuffle])
                            reliability[i_cell, i_shuffle] = np.nanmean(corrs_cell)
                            Fano_factor[i_cell, i_shuffle] = np.nanmean(nan_divide(np.nanvar(laps_rates, axis=1), rate_matrix[:,i_cell,i_shuffle], rate_matrix[:,i_cell,i_shuffle] > 0))

                        shuffle_ecdf = reliability[i_cell, 0:self.N_shuffle]
                        data_point = reliability[i_cell, self.N_shuffle]
                        P_reliability_batch[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

                    self.cell_reliability_batch.append(reliability)
                    self.P_reliability_batch.append(P_reliability_batch)
                    self.cell_Fano_factor_batch.append(Fano_factor)


                    print('calculating Skaggs spatial info...')
                    ## Skaggs spatial info
                    skaggs_matrix=np.zeros((minibatchsize, self.N_shuffle+1))
                    P_skaggs_batch = np.zeros(minibatchsize)
                    P_x=total_time/np.sum(total_time)
                    for i_cell in range(minibatchsize):
                        for i_shuffle in range(self.N_shuffle+1):
                            mean_firing = rates[i_cell,i_shuffle]
                            lambda_x = rate_matrix[:,i_cell,i_shuffle]
                            i_nonzero = np.nonzero(lambda_x > 0)
                            skaggs_matrix[i_cell,i_shuffle] = np.sum(lambda_x[i_nonzero]*np.log2(lambda_x[i_nonzero]/mean_firing)*P_x[i_nonzero]) / mean_firing

                        shuffle_ecdf = skaggs_matrix[i_cell, 0:self.N_shuffle]
                        data_point = skaggs_matrix[i_cell, self.N_shuffle]
                        P_skaggs_batch[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

                    self.cell_skaggs_batch.append(skaggs_matrix)
                    self.P_skaggs_batch.append(P_skaggs_batch)
                     
                    ## active laps/ all laps spks
                    #use raw spks instead activity tensor
                    print('calculating proportion of active laps...')
                    active_laps = np.zeros((minibatchsize, N_laps_corr, self.N_shuffle+1))

                    icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                    i_laps_abs = self.i_Laps_ImData[np.nonzero(icorrids == corridor)[0]]
                    k = 0
                    for i_lap in i_laps_abs:#y=ROI
                        for i_shuffle in range(self.N_shuffle+1):
                            act_cells = np.nonzero(np.amax(self.shuffle_ImLaps[i_lap].frames_spikes[:,:,i_shuffle], 1) > 25)[0] # cells * frames * shuffle
                            active_laps[act_cells, k, i_shuffle] = 1 # cells * laps * shuffle
                        k = k + 1

                    active_laps_ratio = np.sum(active_laps, 1) / N_laps_corr
                    self.cell_activelaps_batch.append(active_laps_ratio)
                    
                    ## linear tuning specificity
                    print('calculating linear tuning specificity ...')
                    tuning_spec=np.zeros((minibatchsize, self.N_shuffle+1))
                    P_tuning_specificity_batch = np.zeros(minibatchsize)
                    xbins = (np.arange(self.N_pos_bins) + 0.5) * self.corridor_length_cm / self.N_pos_bins
                    
                    for i_cell in range(minibatchsize):
                        for i_shuffle in range(self.N_shuffle+1):
                            rr = np.copy(rate_matrix[:,i_cell,i_shuffle])
                            rr[rr < np.mean(rr)] = 0
                            Px = rr / np.sum(rr)
                            mu = np.sum(Px * xbins)
                            sigma = np.sqrt(np.sum(Px * xbins**2) - mu**2)
                            tuning_spec[i_cell,i_shuffle] = self.corridor_length_cm / sigma

                        shuffle_ecdf = tuning_spec[i_cell, 0:self.N_shuffle]
                        data_point = tuning_spec[i_cell, self.N_shuffle]
                        P_tuning_specificity_batch[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

                    self.cell_tuning_specificity_batch.append(tuning_spec)
                    self.P_tuning_specificity_batch.append(P_tuning_specificity_batch)

            # # self.cell_rates_batch = [] # a list, each element is a 5 x n_cells matrix with the average rate of the cells in the total corridor, pattern zones 1-3 and reward zone
            # if ((self.task == 'contingency_learning') & (self.N_corridors == 3)):
            #     self.cell_corridor_selectivity_batch[0,:,:] = (self.cell_rates_batch[0][0,:,:] - self.cell_rates_batch[1][0,:,:]) / (self.cell_rates_batch[0][0,:,:] + self.cell_rates_batch[1][0,:,:])
            #     self.cell_corridor_selectivity_batch[1,:,:] = (self.cell_rates_batch[0][1,:,:] - self.cell_rates_batch[1][1,:,:]) / (self.cell_rates_batch[0][1,:,:] + self.cell_rates_batch[1][1,:,:])
            #     self.cell_corridor_selectivity_batch[2,:,:] = (self.cell_rates_batch[0][2,:,:] - self.cell_rates_batch[1][2,:,:]) / (self.cell_rates_batch[0][2,:,:] + self.cell_rates_batch[1][2,:,:])
            #     self.cell_corridor_selectivity_batch[3,:,:] = (self.cell_rates_batch[0][3,:,:] - self.cell_rates_batch[1][3,:,:]) / (self.cell_rates_batch[0][3,:,:] + self.cell_rates_batch[1][3,:,:])
            #     self.cell_corridor_selectivity_batch[4,:,:] = (self.cell_rates_batch[0][4,:,:] - self.cell_rates_batch[1][4,:,:]) / (self.cell_rates_batch[0][4,:,:] + self.cell_rates_batch[1][4,:,:])
            
            #     for i_region in range(5):
            #         for i_cell in range(minibatchsize):
            #             shuffle_ecdf = self.cell_corridor_selectivity_batch[i_region,i_cell, 0:self.N_shuffle]
            #             data_point = self.cell_corridor_selectivity_batch[i_region,i_cell, self.N_shuffle]
            #             self.P_selectivity_batch[i_region,i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

        if (self.N_corridors > 1):
            self.calc_selectivity_similarity()

    def calc_selectivity_similarity(self, zone=None):
        ## corridor selectivity calculated for M corridors for all neurons
        ## selectivity is defined as (max(r) - min(r)) / sum(r) 
        ##           is always positive
        ##           is near 0 for non-selective cells
        ##           is 1 for cells that are active for a single corridor
        ## the selectivity is stored in a vector of 2 elements for each cell:
        ##      1. selectivity
        ##      2. id of the corridor with the maximal firing rate
        ##
        ## corridor similarity is calculated for M corridors for all neurons
        ## similarity is defined as the average correlation between the ratemaps
        ##           is always positive
        ##           is near 0 for cells with uncorrelated ratemaps
        ##           is 1 if cells have identical ratemaps for all the corridors
        ## the similarity is stored in a vector of 1 element for each cell
        ## for indexing: 
        ## K    corridors
        ## N    cells 
        ## M    shuffles
        ## L    corridor length in bins

        print('calculating corridor selectivity ...')
        rate_matrix = np.array(self.cell_rates_batch) # K x N x M

        max_rate = np.max(rate_matrix, axis=0)
        i_corr_max = np.argmax(rate_matrix, axis=0)
        min_rate = np.min(rate_matrix, axis=0)
        sumrate = np.sum(rate_matrix, axis=0)

        self.cell_corridor_selectivity_batch = (max_rate - min_rate) / sumrate # matrix, N x M


        # in Rita's task, we also calculate corridor selectivity in the pattern and reward zones:
        if (self.task == 'contingency_learning'):
            rate_matrix = np.array(self.cell_pattern_rates_batch) # K x 4 x N x M = N_corridor x 4 x N_cell x N_shuffle
            max_rate = np.max(rate_matrix, axis=0) # 4 x N x M
            min_rate = np.min(rate_matrix, axis=0)
            sumrate = np.sum(rate_matrix, axis=0)
            self.cell_pattern_selectivity_batch = (max_rate - min_rate) / sumrate
        
        print('calculating corridor similarity ...  Number of corridors:', self.N_corridors)
        minibatchsize = self.activity_tensor.shape[1]

        map_mat = np.array(self.ratemaps_batch) # K x L x N x M
        M = int(self.N_corridors * (self.N_corridors - 1) / 2)
        similarity_matrix = np.zeros((M, minibatchsize, self.N_shuffle+1))
        m = 0
        for i_cor in np.arange(self.N_corridors-1):
            for j_cor in np.arange(i_cor+1, self.N_corridors):
                for i_cell in np.arange(minibatchsize):
                    # print(i_cor, j_cor)
                    X = np.transpose(map_mat[i_cor,:,i_cell,:])
                    Y = np.transpose(map_mat[j_cor,:,i_cell,:])
                    similarity_matrix[m,i_cell,] = Mcorrcoef(X,Y, zero_var_out=0)

        self.cell_corridor_similarity_batch = np.mean(similarity_matrix, axis=0) # matrix of N x M

        for i_cell in np.arange(minibatchsize):
            shuffle_ecdf = self.cell_corridor_selectivity_batch[i_cell, 0:self.N_shuffle]
            data_point = self.cell_corridor_selectivity_batch[i_cell, self.N_shuffle]
            self.P_corridor_selectivity_batch[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

            shuffle_ecdf = self.cell_corridor_similarity_batch[i_cell, 0:self.N_shuffle]
            data_point = self.cell_corridor_similarity_batch[i_cell, self.N_shuffle]
            self.P_corridor_similarity_batch[i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)

            if (self.task == 'contingency_learning'):
                for kk in np.arange(4):
                    shuffle_ecdf = self.cell_pattern_selectivity_batch[kk, i_cell, 0:self.N_shuffle]
                    data_point = self.cell_pattern_selectivity_batch[kk, i_cell, self.N_shuffle]
                    self.P_pattern_selectivity_batch[kk, i_cell] = sum(shuffle_ecdf > data_point) / float(self.N_shuffle)


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
        # n_corridors=self.corridors.size-1#we don't want to plot corridor 0
        n_corridors=len(self.cell_rates_batch)
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
            # data = np.transpose(self.cell_activelaps_batch[i_corridor][iplot_cells,:])
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
            # data = np.transpose(self.cell_Fano_factor_batch[i_corridor][iplot_cells,:])
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


        # ###########################
        # ## plots of corridor selectivity
        # if ((self.task == 'contingency_learning') & (self.N_corridors == 3)):

        #     fig, ax = plt.subplots(1, 5, figsize=(7,3), sharex='all', sharey='all')
        #     plt.subplots_adjust(wspace=0.35, hspace=0.2)

        #     ## corridor selectivity    
        #     data = np.transpose(self.cell_corridor_selectivity[0,iplot_cells,:])
        #     ax[0].violinplot(data[0:self.N_shuffle,:])
        #     ax[0].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C1')
        #     ax[0].set_title('total corridor selectivity')
        #     ax[0].set_ylabel('selectivity index')
        #     ax[0].set_xlabel('cells')
        #     ax[0].set_xticks(np.arange(Ncells_to_plot)+1)
        #     ax[0].set_xticklabels(cellids, rotation=90)

        #     ## corridor selectivity    
        #     data = np.transpose(self.cell_corridor_selectivity[1,iplot_cells,:])
        #     ax[1].violinplot(data[0:self.N_shuffle,:])
        #     ax[1].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C2')
        #     ax[1].set_title('pattern 1 selectivity')
        #     ax[1].set_ylabel('selectivity index')
        #     ax[1].set_xlabel('cells')
        #     ax[1].set_xticks(np.arange(Ncells_to_plot)+1)
        #     ax[1].set_xticklabels(cellids, rotation=90)

        #     ## corridor selectivity    
        #     data = np.transpose(self.cell_corridor_selectivity[2,iplot_cells,:])
        #     ax[2].violinplot(data[0:self.N_shuffle,:])
        #     ax[2].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C3')
        #     ax[2].set_title('pattern 2 selectivity')
        #     ax[2].set_ylabel('selectivity index')
        #     ax[2].set_xlabel('cells')
        #     ax[2].set_xticks(np.arange(Ncells_to_plot)+1)
        #     ax[2].set_xticklabels(cellids, rotation=90)

        #     ## corridor selectivity    
        #     data = np.transpose(self.cell_corridor_selectivity[3,iplot_cells,:])
        #     ax[3].violinplot(data[0:self.N_shuffle,:])
        #     ax[3].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C4')
        #     ax[3].set_title('pattern 3 selectivity')
        #     ax[3].set_ylabel('selectivity index')
        #     ax[3].set_xlabel('cells')
        #     ax[3].set_xticks(np.arange(Ncells_to_plot)+1)
        #     ax[3].set_xticklabels(cellids, rotation=90)

        #     ## corridor selectivity    
        #     data = np.transpose(self.cell_corridor_selectivity[4,iplot_cells,:])
        #     ax[4].violinplot(data[0:self.N_shuffle,:])
        #     ax[4].scatter(np.arange(Ncells_to_plot)+1, data[self.N_shuffle,:], c='C5')
        #     ax[4].set_title('reward selectivity')
        #     ax[4].set_ylabel('selectivity index')
        #     ax[4].set_xlabel('cells')
        #     ax[4].set_xticks(np.arange(Ncells_to_plot)+1)
        #     ax[4].set_xticklabels(cellids, rotation=90)

        #     plt.show(block=False)

    def Hainmuller_PCs_shuffle(self):
        ## ratemaps: similar to the activity tensor, the laps are sorted by the corridors
        self.candidate_PCs_batch = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs_batch = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping

        ## we calculate the rate matrix for all corridors - we need to use the same colors for the images
        for corrid in np.unique(self.i_corridors):
            # select the laps in the corridor 
            # only laps with imaging data are selected - this will index the activity_tensor
            if (sum(self.i_corridors[self.i_Laps_ImData] == corrid) > 5):
                i_corrid = int(np.nonzero(self.corridors == corrid)[0] - 1)
                rate_matrix = self.ratemaps_batch[i_corrid]
                cell_number = rate_matrix.shape[1]

                candidate_cells = np.zeros((cell_number, self.N_shuffle+1))
                accepted_cells = np.zeros(cell_number)
                
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
                        for k in range(self.N_pos_bins):
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
                            index_outfield = np.setdiff1d(np.arange(self.N_pos_bins), index_infield)
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

                self.candidate_PCs_batch.append(candidate_cells)
                self.accepted_PCs_batch.append(accepted_cells.astype(int))
        
        # if (self.N_corridors == 2):
        #     for i_cell in np.arange(rate_matrix.shape[1]):
        #         for i_shuffle in np.arange(rate_matrix.shape[2]):
        #             self.cell_corridor_similarity[:,i_cell, i_shuffle] = scipy.stats.pearsonr(self.ratemaps[0][:,i_cell,i_shuffle], self.ratemaps[1][:,i_cell,i_shuffle])


class Shuffle_ImData:
    'common base class for shuffled laps'

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, lap_frames_spikes, lap_frames_pos, lap_frames_time, corridor_list, dt=0.01, printout=False, speed_threshold=5, elfiz=False, dt_imaging=0.033602467):
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
        self.elfiz=elfiz

        self.speed_threshold = speed_threshold ## cm / s 106 cm - 3500 roxels; roxel/s * 106.5/3500 = cm/s
        self.corridor_length_roxel = (self.corridor_list.corridors[self.corridor].length - 1024.0) / (7168.0 - 1024.0) * 3500
        self.N_pos_bins = int(np.round(self.corridor_length_roxel / 70))
        self.speed_factor = 106.5 / 3500 ## constant to convert distance from pixel to cm
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor # cm

        self.dt_imaging = dt_imaging

        self.frames_spikes = lap_frames_spikes
        self.frames_pos = lap_frames_pos
        self.frames_time = lap_frames_time
        
        self.bincenters = np.arange(0, self.corridor_length_roxel, 70) + 70 / 2.0
        self.n_cells = 1 # we still create the same np arrays even if there are no cells
        self.n_shuffle = 1 

        ##################################################################################
        ## lick position and reward position
        ##################################################################################
        F = interp1d(self.raw_time,self.raw_position)    
        self.lick_position = F(self.lick_times)
        self.reward_position = F(self.reward_times)

        ##################################################################################
        ## speed vs. time
        ##################################################################################
        self.imaging_data = True

        if (np.isnan(self.frames_time).any()): # we have real data
            self.imaging_data = False
            ## resample time uniformly for calculating speed
            start_time = np.ceil(self.raw_time.min()/self.dt_imaging)*self.dt_imaging
            end_time = np.floor(self.raw_time.max()/self.dt_imaging)*self.dt_imaging
            Ntimes = int(round((end_time - start_time) / self.dt_imaging)) + 1
            self.frames_time = np.linspace(start_time, end_time, Ntimes)
            self.frames_pos = F(self.frames_time)

        ## calculate the speed during the frames
        speed = np.diff(self.frames_pos) * self.speed_factor / self.dt_imaging # cm / s       
        speed_first = 2 * speed[0] - speed[1] # linear extrapolation: x1 - (x2 - x1)
        self.frames_speed = np.hstack([speed_first, speed])

        ##################################################################################
        ## speed, lick and spiking vs. position
        ##################################################################################

        ####################################################################
        ## calculate the lick-rate and the average speed versus location    
        bin_counts = np.zeros(self.N_pos_bins)
        fast_bin_counts = np.zeros(self.N_pos_bins)

        for i_frame in range(len(self.frames_pos)):
            bin_number = int(self.frames_pos[i_frame] // 70)
            bin_counts[bin_number] += 1
            if (self.frames_speed[i_frame] > self.speed_threshold):
                fast_bin_counts[bin_number] += 1

        self.T_pos = bin_counts * self.dt_imaging           # used for lick rate and average speed
        self.T_pos_fast = fast_bin_counts * self.dt_imaging # used for spike rate calculations


        ####################################################################
        ## calculate the cell activations (spike rate) as a function of position
        if (self.imaging_data == True):        
            self.n_cells = self.frames_spikes.shape[0]
            self.n_shuffle = self.frames_spikes.shape[2]
    
            self.spks_pos = np.zeros((self.n_cells, self.N_pos_bins, self.n_shuffle)) # sum of spike counts measured at a given position
            self.event_rate = np.zeros((self.n_cells, self.N_pos_bins, self.n_shuffle)) # spike rate 
            
            for i_frame in range(len(self.frames_pos)):
                bin_number = int(self.frames_pos[i_frame] // 70)
                if (self.frames_speed[i_frame] > self.speed_threshold):
                    if (self.elfiz):
                        self.spks_pos[:,bin_number,:] = self.spks_pos[:,bin_number,:] + self.frames_spikes[:,i_frame,:]
                    else:
                        ### we need to multiply the values with dt_imaging as this converts probilities to expected counts
                        self.spks_pos[:,bin_number,:] = self.spks_pos[:,bin_number,:] + self.frames_spikes[:,i_frame,:] * self.dt_imaging
            for bin_number in range(self.N_pos_bins):
                if (self.T_pos_fast[bin_number] > 0): # otherwise the rate will remain 0
                    self.event_rate[:,bin_number,:] = self.spks_pos[:,bin_number,:] / self.T_pos_fast[bin_number]                
