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
from matplotlib.patches import Polygon
import sys
from sys import version_info
import copy
import time
import os
import pickle
from xml.dom import minidom
from matplotlib.backends.backend_pdf import PdfPages

from Stages import *
from Corridors import *


class ImagingSessionData:
    'Base structure for both imaging and behaviour data'
    def __init__(self, datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=np.nan):
        self.name = name
        self.stage = 0
        self.stages = []
        self.sessionID = sessionID

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

        self.get_stage(datapath, date_time, name, task)
        self.corridors = np.hstack([0, np.array(self.stage_list.stages[self.stage].corridors)])

        ##################################################
        ## loading the trigger signals to match LabView and Imaging data axis
        ## CAPITAL LETTERS: variables defined with IMAGING time axis
        ## normal LETTERS: variables defined with LabView time axis
        ##################################################
        ### trigger log starts and durations

        ###############################
        ### this all will need to move to the LocateImaging routine to save memory
        ###############################

        beh_folder = datapath + 'data/' + name + '_' + task + '/' + date_time + '/'
        trigger_log_file_string = beh_folder + date_time + '_rn013_contingency_learning_TriggerLog.txt'
                
        ## matching imaging time with labview time
        self.imstart_time = 0 # the labview time of the first imaging frame
        self.LocateImaging(trigger_log_file_string, TRIGGER_VOLTAGE_FILENAME)

        ##################################################
        ## loading imaging data
        ##################################################
        F_string = suite2p_folder + 'F.npy'
        Fneu_string=suite2p_folder + 'Fneu.npy'
        spks_string=suite2p_folder + 'spks.npy'
        # stat_string=suite2p_folder+'/stat.npy'
        # ops_string=suite2p_folder+'/ops.npy'
        iscell_string=suite2p_folder + 'iscell.npy'
        
        self.F_all = np.load(F_string) # npy array, N_ROI x N_frames, fluorescence traces of ROIs from suite2p
        self.Fneu = np.load(Fneu_string) # npy array, N_ROI x N_frames, fluorescence traces of neuropil from suite2p
        self.spks_all = np.load(spks_string) # npy array, N_ROI x N_frames, spike events detected from suite2p
        self.iscell = np.load(iscell_string) # np array, N_ROI x 2, 1st col: binary classified as cell. 2nd P_cell?
        print('suite2p data loaded')               

        ## arrays containing only valid cells
        self.neuron_index = np.nonzero(self.iscell[:,0])[0]
        self.F = self.F_all[self.neuron_index,:]
        self.spks = self.spks_all[self.neuron_index,:]
        self.dF_F = np.copy(self.F)
        self.N_cells = self.F.shape[0]
        self.calc_dF_F()
        # self.cell_SDs = np.sqrt(np.var(self.dF_F, 1)) - we calculate this later

        self.frame_times = np.nan # labview coordinates
        self.LoadImaging_times(imaging_logfile_name, self.imstart_time)
        self.frame_pos = np.zeros(len(self.frame_times)) # position and 
        self.frame_laps = np.zeros(len(self.frame_times)) # lap number for the imaging frames, to be filled later
        print('suite2p time axis loaded')       
          
        ##################################################
        ## loading behavioral data
        ##################################################

        self.ImLaps = [] # list containing a special class for storing the imaging and behavioral data for single laps
        self.n_laps = 0 # total number of laps
        self.i_Laps_ImData = np.zeros(1) # np array with the index of laps with imaging
        self.i_corridors = np.zeros(1) # np array with the index of corridors in each run

        self.get_lapdata(datapath, date_time, name, task) # collects all behavioral and imaging data and sort it into laps, storing each in a Lap_ImData object
        self.N_bins = 50 # self.ImLaps[self.i_Laps_ImData[2]].event_rate.shape[1]
        self.N_ImLaps = len(self.i_Laps_ImData)

        self.activity_tensor = np.zeros((self.N_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials containing the spikes
        self.activity_tensor_time = np.zeros((self.N_bins, self.N_ImLaps)) # a tensor with space x  trials containing the time spent at each location in each lap
        self.combine_lapdata() ## fills in the cell_activity tensor

        self.cell_SDs = np.zeros(self.N_cells) # a vector with the SD of the cells
        self.cell_SNR = np.zeros(self.N_cells) # a vector with the signal to noise ratio of the cells (max F / SD)
        self.cell_rates = [] # a list, each element is a vector with the average rate of the cells in a corridor
        self.cell_reliability = [] # a list, each element is a vector with the reliability of the cells in a corridor
        self.cell_Fano_factor = [] # a list, each element is a vector with the reliability of the cells in a corridor
        self.cell_skaggs=[] # a list, each element is a vector with the skaggs93 spatial info of the cells in a corridor
        self.cell_activelaps=[] # a list, each element is a vector with the % of significantly spiking laps of the cells in a corridor
        self.cell_activelaps_df=[] # a list, each element is a vector with the % of significantly active (dF/F) laps of the cells in a corridor
        self.cell_tuning_specificity=[] # a list, each element is a vector with the tuning specificity of the cells in a corridor
        self.calculate_properties()

        self.test_anticipatory()

        
    def get_stage(self, datapath, date_time, name, task):
        # function that reads the action_log_file and finds the current stage
        action_log_file_string=datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_UserActionLog.txt'
        action_log_file=open(action_log_file_string)
        log_file_reader=csv.reader(action_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            if (line[1] == 'Stage'):
                self.stage = int(round(float(line[2])))

    def LoadImaging_times(self, imaging_logfile_name, offset):
        # function that reads the action_log_file and finds the current stage
        # minidom is an xml file interpreter for python
        # hope it works for python 3.7...
        imaging_logfile = minidom.parse(imaging_logfile_name)
        voltage_rec = imaging_logfile.getElementsByTagName('VoltageRecording')
        voltage_delay = float(voltage_rec[0].attributes['absoluteTime'].value)
        ## the offset is the time of the first voltage signal in Labview time
        ## the signal's 0 has a slight delay compared to the time 0 of the imaging recording 
        ## we substract this delay from the offset to get the LabView time of the time 0 of the imaging recording
        corrected_offset = offset - voltage_delay

        frames = imaging_logfile.getElementsByTagName('Frame')
        self.frame_times = np.zeros(len(frames)) # this is already in labview time
        for i in range(len(frames)):
            self.frame_times[i] = float(frames[i].attributes['absoluteTime'].value) + corrected_offset
        if (len(self.frame_times) != self.F.shape[1]):
            print('Warning: imaging frame number does not match suite2p frame number! Something is wrong!')
            N_frames = min([len(self.frame_times), self.F.shape[1]])
            if (len(self.frame_times) < self.F.shape[1]):
                self.F_all = self.F_all[:, 1:N_frames]
                self.spks_all = self.spks_all[:, 1:N_frames]
                self.Fneu = self.Fneu[:, 1:N_frames]
                self.F = self.F[:, 1:N_frames]
                self.spks = self.spks[:, 1:N_frames]
                self.dF_F = self.dF_F[:, 1:N_frames]
            else:
                self.frame_times = self.frame_times[0:N_frames]   

    def LoadExpLog(self, exp_log_file_string): # BBU: just reads the raw data, no separation into laps
        position=[]
        exp_loop_timestamps=[]
        mazeID=[]
        exp_log_file=open(exp_log_file_string,'r')
        log_file_reader=csv.reader(exp_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            position.append(int(line[3]))
            exp_loop_timestamps.append(float(line[0]))
            mazeID.append(int(line[2]))
        data=[position,exp_loop_timestamps, mazeID]
        print('exp file loaded')
        return data

    def test_anticipatory(self):
        corridor_ids = np.zeros(self.n_laps)
        for i in range(self.n_laps):
            corridor_ids[i] = self.ImLaps[i].corridor # the true corridor ID
        corridor_types = np.unique(corridor_ids)
        nrow = len(corridor_types)
        self.anticipatory = []

        for row in range(nrow):
            ids = np.where(corridor_ids == corridor_types[row])
            n_laps = np.shape(ids)[1]
            n_zones = np.shape(self.ImLaps[ids[0][0]].zones)[1]
            if (n_zones == 1):
                lick_rates = np.zeros([2,n_laps])
                k = 0
                for lap in np.nditer(ids):
                    lick_rates[:,k] = self.ImLaps[lap].preZoneRate
                    k = k + 1
                self.anticipatory.append(anticipatory_Licks(lick_rates[0,:], lick_rates[1,:], corridor_types[row]))


    ##########################################################
    def LocateImaging(self, trigger_log_file_string, TRIGGER_VOLTAGE_FILENAME):
        # 1. trigger_data_voltage  = np.array, 4 x N_triggers, each row is the 1. start time, 2, end time, duration, ITT
        # 2. select only trigger data with ITT > 10 ms
        # 3. find the shortest trigger
        # 4. find candidate trigger times in the log_trigger by matching trigger duration
        # 5. check the next ITTS in voltage recording and log trigger
        # intputs: 
        #   self.trigger_log_starts, 
        #   self.trigger_log_lengths, 
        #   self.TRIGGER_VOLTAGE_VALUE, 
        #   self.TRIGGER_VOLTAGE_TIMES
        # output:
        #   self.imstart_time: singe scalar [s]: Labview time of the start of voltage recordings
        #
        # only works for 1 imaging session...
        # self.imstart_time = 537.133055 # Bazsi's best guess
        # print('Imaging time axis guessed by Bazsi...')
        
        #0)load recorded trigger 
        trigger_log_starts = []
        trigger_log_lengths = []      
        trigger_log_file=open(trigger_log_file_string)
        log_file_reader=csv.reader(trigger_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:             
            trigger_log_starts.append(line[0])
            trigger_log_lengths.append(line[1])
        print('trigger logfile loaded')


        TRIGGER_VOLTAGE_VALUE = [] 
        TRIGGER_VOLTAGE_TIMES = []
        trigger_signal_file=open(TRIGGER_VOLTAGE_FILENAME, 'r')
        trigger_reader=csv.reader(trigger_signal_file, delimiter=',')
        next(trigger_reader, None)
        for line in trigger_reader:
            TRIGGER_VOLTAGE_VALUE.append(float(line[1]))
            TRIGGER_VOLTAGE_TIMES.append(float(line[0]))        
        v=np.array(TRIGGER_VOLTAGE_VALUE)
        x=np.array(TRIGGER_VOLTAGE_TIMES)
        print('trigger voltage signal loaded')
        
        ## find trigger start and end times
        rise_index=np.nonzero((v[0:-1] < 1)&(v[1:]>= 1))[0]+1#+1 needed otherwise we are pointing to the index just before the trigger
        RISE_X=x[rise_index]
        
        fall_index=np.nonzero((v[0:-1] > 1)&(v[1:]<= 1))[0]+1
        FALL_X=x[fall_index]
        
        # pairing rises with falls
        if (RISE_X[0]>FALL_X[0]):
            print('deleting first fall')
            FALL_X = np.delete(FALL_X,0)
        if (RISE_X[-1] > FALL_X[-1]):
            print('deleting last rise')
            RISE_X=np.delete(RISE_X,-1)

        if np.size(RISE_X)!=np.size(FALL_X):
            print(np.size(RISE_X),np.size(FALL_X))
            print('trigger ascending and desending edges do not match! unable to locate imaging part')
            self.imstart_time = np.nan
            return

        #1) filling up trigger_data_voltage array:
        #trigger_data_voltage: 1. start time, 2. end time, 3. duration, 4.ITT, 5. index
        trigger_data_voltage=np.zeros((np.size(RISE_X),5))
        trigger_data_voltage[:,0]=RISE_X
        trigger_data_voltage[:,1]=FALL_X
        trigger_data_voltage[:,2]=FALL_X-RISE_X
        TEMP_FALL=np.concatenate([[0],FALL_X])
        TEMP_FALL=np.delete(TEMP_FALL,-1)
        trigger_data_voltage[:,3]=RISE_X-TEMP_FALL
        trigger_data_voltage[:,4]=np.arange(0,np.size(RISE_X))
            
        #2) keeping only triggers with ITT>10    
        valid_indexes=np.nonzero(trigger_data_voltage[:,3]>10)[0]
        trigger_data_voltage_sub=trigger_data_voltage[valid_indexes,:]
        
        #3) find the valid shortest trigger
        minindex=np.argmin(trigger_data_voltage_sub[:,2])
        used_index=int(trigger_data_voltage_sub[minindex][4])
        n_extra_indexes=min(5,trigger_data_voltage.shape[0]-used_index)
        print(n_extra_indexes)
    
        #4)find the candidate trigger times
        candidate_log_indexes=[]
        for i in range(len(trigger_log_lengths)):
            if abs(float(trigger_log_lengths[i])-trigger_data_voltage[used_index][2])<5:
                candidate_log_indexes.append(i)
    
        #5)check the next ITT-s, locate relevant behavior
        print('min trigger length:',trigger_data_voltage[used_index,2])
        if trigger_data_voltage[used_index,2]>450:
            print('Warning! No short enough trigger in this recording! Unable to locate imaging')
            self.imstart_time = np.nan
            return
        else:
            match_found=False
            for i in range(len(candidate_log_indexes)):    
                log_reference_index=candidate_log_indexes[i]
                difs=[]
                for j in range(n_extra_indexes):
                    dif_log=(float(trigger_log_starts[log_reference_index+j])-float(trigger_log_starts[log_reference_index]))*1000
                    dif_mes=trigger_data_voltage[used_index+j,0]-trigger_data_voltage[used_index,0]
                    delta=abs(dif_log-dif_mes)
                    difs.append(delta)
#                    print(self.trigger_log_lengths[candidate_log_indexes[i]],'log',dif_log,'mes', dif_mes,'dif', delta)
                if max(difs) < 2:
                    if match_found==False:                      
                        lap_time_of_first_frame=float(trigger_log_starts[log_reference_index])-trigger_data_voltage[used_index,0]/1000
                        print('relevant behavior located, lap time of the first frame:',lap_time_of_first_frame)
                        match_found=True
                    else:
                        print('Warning! More than one trigger matches found!')
            if match_found==True:
                self.imstart_time = lap_time_of_first_frame
            else:
                print('no precise trigger mach found: need to refine code or check device')
                self.imstart_time = np.nan


    def calc_dF_F(self):
        n_cells = self.F.shape[0]
        #calculate baseline
        index=0
        for i in range(n_cells):
            trace=self.F[i,]
            hist=np.histogram(trace, bins=100)
            max_index = np.where(hist[0] == max(hist[0]))[0][0]
            baseline = hist[1][max_index]
            self.dF_F[i,] = (self.F[i,] - baseline) / baseline
        print('dF/F calculated for cell ROI-s')

    ##############################################################
    ## loading the LabView data
    ## separating data into laps
    ##############################################################

    def get_lapdata(self, datapath, date_time, name, task):

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

        #################################################
        ## add position, and lap info into the imaging frames
        #################################################
        F = interp1d(laptime, pos) 
        self.frame_pos = np.round(F(self.frame_times))
        F = interp1d(laptime, lap) 
        self.frame_laps = F(self.frame_times) 
        ## frame_laps is NOT integer for frames between laps
        ## however, it MAY not be integer even even for frames within a lap...
        #################################################
        ## end of - add position, and lap info into the imaging frames
        #################################################
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
                    lap_frames_dF_F = self.dF_F[:,iframes]
                    lap_frames_spikes = self.spks[:,iframes]
                    lap_frames_time = self.frame_times[iframes]
                    lap_frames_pos = self.frame_pos[iframes]
                    i_ImData.append(self.n_laps)
                else:
                    lap_frames_dF_F = np.nan
                    lap_frames_spikes = np.nan
                    lap_frames_time = np.nan
                    lap_frames_pos = np.nan 
                    
                # sessions.append(Lap_Data(name, i, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions))
                self.ImLaps.append(Lap_ImData(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time, self.corridor_list))
                self.n_laps = self.n_laps + 1
            else:
                N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1) - we do not do anythin with this...

        self.i_Laps_ImData = np.array(i_ImData) # index of laps with imaging data
        self.i_corridors = np.array(i_corrids) # ID of corridor for the current lap
        # self.cell_activity = np.zeros((self.N_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials
    def combine_lapdata(self): ## fills in the cell_activity tensor
        valid_lap = np.zeros(len(self.i_Laps_ImData))
        k_lap = 0
        for i_lap in self.i_Laps_ImData:
            if (self.ImLaps[i_lap].n_cells > 1):
                valid_lap[k_lap] = 1
                self.activity_tensor[:,:,k_lap] = np.transpose(self.ImLaps[i_lap].spks_pos)
                self.activity_tensor_time[:,k_lap] = self.ImLaps[i_lap].T_pos
            k_lap = k_lap + 1

        i_valid_laps = np.nonzero(valid_lap)[0]
        self.i_Laps_ImData = self.i_Laps_ImData[i_valid_laps]
        self.activity_tensor = self.activity_tensor[:,:,i_valid_laps]
        self.activity_tensor_time = self.activity_tensor_time[:,i_valid_laps]

    def calculate_properties(self, nSD=4):
        self.cell_SDs = np.zeros(self.N_cells) # a vector with the SD of the cells
        self.cell_SNR = np.zeros(self.N_cells) # a vector with the signal to noise ratio of the cells (max F / SD)
        self.cell_rates = [] # we do not append it is already exists...
        self.cell_reliability = []
        self.cell_Fano_factor = []
        self.cell_skaggs=[]
        self.cell_activelaps=[]
        self.cell_activelaps_df=[]
        self.cell_tuning_specificity=[]


        print('calculating SNR...')
        sp_threshold = 20
        frame_rate = int(np.ceil(1/np.median(np.diff(self.frame_times))))
        T_after_spike = 3 #s 
        T_before_spike = 0.5 #s 
        Tmin_no_spike = 1 #s 
        L_after_spike = int(round(T_after_spike  * frame_rate))
        L_before_spike = int(round(T_before_spike  * frame_rate))
        Lmin_no_spike = int(round(Tmin_no_spike * frame_rate ))

        N_frames = len(self.frame_times) 
        filt = np.ones(frame_rate)

        ### 1. find places where there are no spikes for a long interval 
        ### 1.1. we add all spikes in a 1s window by convolving it with a 1s box car function
        # fig, ax = plt.subplots(figsize=(12,8))
        # i_plot = 0  
        # cells = np.sort(np.random.randint(0, self.N_cells, 6))
        # cells[0] = 4
        # cells[5] = 1064

        for i_cell in range(self.N_cells):
            # print(i_cell)
            allspikes_1s = np.hstack([np.repeat(sp_threshold, frame_rate-1), np.convolve(self.spks[i_cell,:], filt, mode='valid')])

            ### 1.2 no spikes if the sum remains smaller than sp_threshold
            sp_1s = np.copy(allspikes_1s)
            sp_1s[np.nonzero(allspikes_1s < sp_threshold)[0]] = 0


            ### 1.3. find silent sections
            rise_index=np.nonzero((sp_1s[0:-1] < 1)&(sp_1s[1:]>= 1))[0]+1
            fall_index=np.nonzero((sp_1s[0:-1] > 1)&(sp_1s[1:]<= 1))[0]+1
            if (len(rise_index) == 0):
                rise_index = np.array([int(len(sp_1s))])
            if (max(rise_index) < N_frames-1000):
                rise_index = np.hstack([rise_index, int(len(sp_1s))])


            # pairing rises with falls
            if (fall_index[0]>rise_index[0]):
                # print('deleting first rise')
                rise_index = np.delete(rise_index,0)
            if (fall_index[-1] > rise_index[-1]):
                # print('deleting last fall')
                fall_index=np.delete(fall_index,-1)
            if (len(rise_index) != len(fall_index)):
                print('rise and fall could not be matched for cell ' +  str(i_cell))

            long_index = np.nonzero((rise_index - fall_index) > L_after_spike + L_before_spike + Lmin_no_spike)[0]
            rise_ind = rise_index[long_index]
            fall_ind = fall_index[long_index]

            sds = np.zeros(len(rise_ind))
            sdsD = np.zeros(len(rise_ind))
            for k in range(len(rise_ind)):
                i_start = fall_ind[k] + L_after_spike
                i_end = rise_ind[k] - L_before_spike
                sds[k] = np.sqrt(np.var(self.dF_F[i_cell,i_start:i_end]))

            self.cell_SDs[i_cell] = np.mean(sds)
            self.cell_SNR[i_cell] = max(self.dF_F[i_cell,:]) / np.mean(sds)

        #     if (i_cell in cells):
        #         title_string = 'cell: ' + str(i_cell) + ', SD: ' + str(round(self.cell_SDs[i_cell], 2)) + ', SNR:' + str(round(self.cell_SNR[i_cell], 2))
        #         ax.plot(self.frame_times, self.dF_F[i_cell,:] + 10*i_plot, 'C0')
        #         ax.plot(self.frame_times, self.spks[i_cell,:]/100 + 10*i_plot, 'C2')
        #         ax.plot(self.frame_times, allspikes_1s/100 + 10*i_plot, 'C3')
        #         ax.plot(self.frame_times, sp_1s/100 + 10*i_plot, 'C1')
        #         ax.text(min(self.frame_times), 8 + 10*i_plot, title_string)
        #         i_plot = i_plot + 1

        # ax.set(xlabel='time (s)', ylabel='Fluorescence and spikes', ylim=[-1, 10*i_plot])
        # plt.show(block=False)

        # fig, ax = plt.subplots()
        # ax.scatter(self.cell_SDs, self.cell_SNR, c='C0', alpha=0.5)
        # ax.scatter(self.cell_SDs[cells], self.cell_SNR[cells], c='C1', alpha=0.5)
        # ax.set(xlabel='SD', ylabel='SNR')
        # plt.show(block=False)


        # fig, ax = plt.subplots()
        # ax.plot(D1.frame_times, D1.F[660,:], c='C0')
        # ax.plot(D1.frame_times, D1.Fneu[660,:], c='C2')
        # ax.plot(D1.frame_times, D1.Fneu[66,:], c='C1')
        # ax.set(xlabel='time', ylabel='F')
        # plt.show(block=False)

        # cneu = np.zeros(D1.N_cells)
        # cneu183 = np.zeros(D1.N_cells)
        # for i_cell in range(D1.N_cells):
        #     cneu[i_cell] = np.corrcoef(D1.F[i_cell,:], D1.Fneu[i_cell,:])[1,0]
        #     cneu183[i_cell] = np.corrcoef(D1.F[183,:], D1.Fneu[i_cell,:])[1,0]
        
        # fig, ax = plt.subplots()
        # ax.scatter(range(D1.N_cells), cneu183, c='C0')
        # ax.set(xlabel='cell', ylabel='corr with 183')
        # plt.show(block=False)


        print('SNR done')

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

                act_tensor_1 = self.activity_tensor[:,:,i_laps] ## bin x cells x laps; all activity in all laps in corridor i
                total_spikes = np.sum(act_tensor_1, axis=2) ##  bin x cells; total activity of the selected cells in corridor i

                rate_matrix = np.zeros_like(total_spikes) ## event rate 
                
                for i_cell in range(self.N_cells):
                # for i_cell in range(total_spikes.shape[1]):
                    rate_matrix[:,i_cell] = total_spikes[:,i_cell] / total_time

                print('calculating rate, reliability and Fano factor...')

                ## average firing rate
                rates = np.sum(total_spikes, axis=0) / np.sum(total_time)
                self.cell_rates.append(rates)

                ## reliability and Fano factor
                reliability = np.zeros(self.N_cells)
                Fano_factor = np.zeros(self.N_cells)
                for i_cell in range(self.N_cells):
                    laps_rates = nan_divide(act_tensor_1[:,i_cell,:], time_matrix_1, where=(time_matrix_1 > 0.025))
                    corrs_cell = vcorrcoef(np.transpose(laps_rates), rate_matrix[:,i_cell])
                    reliability[i_cell] = np.nanmean(corrs_cell)
                    Fano_factor[i_cell] = np.nanmean(nan_divide(np.nanvar(laps_rates, axis=1), rate_matrix[:,i_cell], rate_matrix[:,i_cell] > 0))
                self.cell_reliability.append(reliability)
                self.cell_Fano_factor.append(Fano_factor)


                print('calculating Skaggs spatial info...')
                ## Skaggs spatial info
                skaggs_vector=np.zeros(self.N_cells)
                P_x=total_time/np.sum(total_time)
                for i_cell in range(self.N_cells):
                    mean_firing = rates[i_cell]
                    lambda_x = rate_matrix[:,i_cell]
                    i_nonzero = np.nonzero(lambda_x > 0)
                    skaggs_vector[i_cell] = np.sum(lambda_x[i_nonzero]*np.log2(lambda_x[i_nonzero]/mean_firing)*P_x[i_nonzero]) / mean_firing
                self.cell_skaggs.append(skaggs_vector)
                 
                ## active laps/ all laps spks
                #use raw spks instead activity tensor
                print('calculating proportion of active laps...')
                active_laps = np.zeros((self.N_cells, N_laps_corr))

                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with iage data
                i_laps_abs = self.i_Laps_ImData[np.nonzero(icorrids == corridor)[0]]
                k = 0
                for i_lap in i_laps_abs:#y=ROI
                    act_cells = np.nonzero(np.amax(self.ImLaps[i_lap].frames_spikes, 1) > 25)[0]
                    active_laps[act_cells, k] = 1
                    k = k + 1

                active_laps_ratio = np.sum(active_laps, 1) / N_laps_corr
                self.cell_activelaps.append(active_laps_ratio)
                
                ## dF/F active laps/all laps
                print('calculating proportion of active laps based on dF/F ...')
                active_laps_df = np.zeros((self.N_cells, N_laps_corr))
                k = 0
                for i_lap in i_laps_abs:
                    act_cells = np.nonzero(np.amax(self.ImLaps[i_lap].frames_dF_F, 1) > (self.cell_SDs*nSD))[0]
                    active_laps_df[act_cells, k] = 1
                    k = k + 1
                active_laps_ratio_df = np.sum(active_laps_df, 1) / N_laps_corr
                self.cell_activelaps_df.append(active_laps_ratio_df)

                ## tuning specificity
                print('calculating tuning specificity ...')
                tuning_spec=np.zeros(self.N_cells)
                r=1
                deg_d=(2*np.pi)/50
                x=np.zeros(50)
                y=np.zeros(50)
                x_ext=np.zeros((50,len(i_laps)))
                y_ext=np.zeros((50,len(i_laps)))
                
                for i in range( 50):
                   x_i,y_i=pol2cart(r,deg_d*i)
                   x[i]=round(x_i,5)
                   y[i]=round(y_i,5)
                   
                for i in range(len(i_laps)):
                    x_ext[:,i]=x
                    y_ext[:,i]=y
                    
                for i_cell in range(self.N_cells):
                    magn_x=nan_divide(self.activity_tensor[:,i_cell,i_laps]*x_ext, self.activity_tensor_time[:,i_laps], self.activity_tensor_time[:,i_laps] > 0)
                    magn_y=nan_divide(self.activity_tensor[:,i_cell,i_laps]*y_ext, self.activity_tensor_time[:,i_laps], self.activity_tensor_time[:,i_laps] > 0)
                    X = np.nansum(magn_x)
                    Y = np.nansum(magn_y)
                    Z = np.nansum(np.sqrt(magn_x**2 + magn_y**2))
                    tuning_spec[i_cell]=np.sqrt(X**2+Y**2) / Z
                self.cell_tuning_specificity.append(tuning_spec)


    def plot_properties(self, cellids=np.array([-1])):
        
        n_corridors=self.corridors.size-1#we don't want to plot corridor 0
        fig, ax = plt.subplots(n_corridors, 4, figsize=(10,5), sharex='col', sharey='col')
        plt.subplots_adjust(wspace=0.35, hspace=0.2)
        # matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

        ax[1,0].scatter(self.cell_SDs, self.cell_SNR, alpha=0.5, c='w', edgecolors='C3')
        if (max(cellids) > 0):
            ax[1,0].scatter(self.cell_SDs[cellids], self.cell_SNR[cellids], alpha=0.75, c='C3')
        # sc = ax_left.scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
        title_string = 'SNR vs. SD'
        ax[1,0].set_title(title_string)
        ax[1,0].set_ylabel('SNR')
        ax[1,0].set_xlabel('SD')
        
        ax[0, 0].axis('off')

        for i in range(n_corridors):
            corridor=self.corridors[i+1]#always plot the specified corridor
            i_corridor = int(np.nonzero(self.corridors == corridor)[0]) - 1
            
            rates = self.cell_rates[i_corridor]
            reliability = self.cell_reliability[i_corridor]
            skaggs_info=self.cell_skaggs[i_corridor]
            Fano_factor = self.cell_Fano_factor[i_corridor]
            specificity = self.cell_tuning_specificity[i_corridor]   
            act_laps = self.cell_activelaps[i_corridor]
#            act_laps_dF = self.cell_activelaps_df[i_corridor]
    
            ax[i,1].scatter(rates, reliability, alpha=0.5, s=skaggs_info*50, c='w', edgecolors='C0')
            if (max(cellids) > 0):
                ax[i,1].scatter(rates[cellids], reliability[cellids], alpha=0.75, s=skaggs_info[cellids]*50, c='C0')
            # sc = ax_left.scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
            title_string = 'corr.' + str(corridor)
            ax[i,1].set_title(title_string)
            ax[i,1].set_ylabel('reliability')
            if (i == n_corridors - 1): ax[i,1].set_xlabel('average event rate')
    
    
            ax[i,2].scatter(act_laps, specificity, alpha=0.5, s=skaggs_info*50, c='w', edgecolors='C1')
            if (max(cellids) > 0):
                ax[i,2].scatter(act_laps[cellids], specificity[cellids], alpha=0.75, s=skaggs_info[cellids]*50, c='C1')
    
            title_string = 'corr.' + str(corridor)
            ax[i,2].set_title(title_string)
            ax[i,2].set_ylabel('tuning specificity')
            if (i == n_corridors - 1): ax[i,2].set_xlabel('percent active laps spikes')
    
            ax[i,3].scatter(skaggs_info, Fano_factor, alpha=0.5, s=skaggs_info*50, c='w', edgecolors='C2')
            if (max(cellids) > 0):
                ax[i,3].scatter(skaggs_info[cellids], Fano_factor[cellids], alpha=0.75, s=skaggs_info[cellids]*50, c='C2')
            # sc = ax_right.scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
            title_string = 'corr.' + str(corridor)
            ax[i,3].set_title(title_string)
            ax[i,3].set_ylabel('Fano factor')
            if (i == n_corridors - 1): ax[i,3].set_xlabel('Skaggs info (bit/event)')

        plt.show(block=False)
           
#            ax[i,0].scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
#            ax[i,0].axis(xmin=xmin0,xmax=xmax0)
#            ax[i,0].axis(ymin=ymin0,ymax=ymax0)
#            title_string = 'reliability vs. rate in corr.' + str(corridor)
#            ax[i,0].set_title(title_string)
#            ax[i,0].set_ylabel('reliability')
#            ax[i,0].set_xlabel('average event rate')
#    
#    
#            ax[i,1].scatter(Fano_factor, specificity, alpha=0.5, s=skaggs_info*50, c='g')
#            ax[i,1].axis(xmin=xmin1,xmax=xmax1)
#            ax[i,1].axis(ymin=ymin1,ymax=ymax1)
#            title_string = 'specificity vs. Fano factor in corr.' + str(corridor)
#            ax[i,1].set_title(title_string)
#            ax[i,1].set_ylabel('tuning specificity')
#            ax[i,1].set_xlabel('Fano factor')
#    
#            ax[i,2].scatter(act_laps, act_laps_dF, alpha=0.5, s=skaggs_info*50, c='r')
#            ax[i,2].axis(xmin=xmin2,xmax=xmax2)
#            ax[i,2].axis(ymin=ymin2,ymax=ymax2)
#            title_string = 'perc. active laps spikes vs. dF in corr.' + str(corridor)
#            ax[i,2].set_title(title_string)
#            ax[i,2].set_ylabel('percent active laps dF/F')
#            ax[i,2].set_xlabel('percent active laps spikes')

        
        
        #####################
#         annot = ax_left.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                     bbox=dict(boxstyle="round", fc="w"),
#                     arrowprops=dict(arrowstyle="->"))
#         annot.set_visible(False)
        
#         def update_annot(ind):
#             pos = sc.get_offsets()[ind["ind"][0]]
#             annot.xy = pos
#             index = "{}".format(" ".join(list(map(str,ind["ind"]))))
#             skaggs_index=int(index)
#             skaggs_text=str(round(skaggs_info[skaggs_index],2))
#             text=index+' sk: '+skaggs_text
#             annot.set_text(text)
# #            annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
#             annot.get_bbox_patch().set_alpha(0.4)
        
#         def hover(event):
#             vis = annot.get_visible()
#             if event.inaxes == ax_left:
#                 cont, ind = sc.contains(event)
#                 if cont:
#                     update_annot(ind)
#                     annot.set_visible(True)
#                     fig.canvas.draw_idle()
#                 else:
#                     if vis:
#                         annot.set_visible(False)
#                         fig.canvas.draw_idle()
        
#         fig.canvas.mpl_connect("motion_notify_event", hover)

    def get_lap_indexes(self, corridor=-1, i_lap=-1):
        ## print the indexes of i_lap (or each lap) in a given corridor
        ## if corridor == -1 then the first corridor is used
        if (corridor == -1):
            corridor = np.unique(self.i_corridors)[0]
        # select the laps in the corridor 
        # only laps with imaging data are selected - this will index the activity_tensor
        i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor)[0] 
        N_laps_corr = len(i_laps)
        print('lap # in corridor ' + str(corridor) + ' with imaging data', 'lap number within session')
        if (i_lap == -1):
            for i_lap in range(N_laps_corr):
                print (i_lap, self.i_Laps_ImData[i_laps[i_lap]])
        else :
            print (i_lap, self.i_Laps_ImData[i_laps[i_lap]])



    def plot_ratemaps(self, corridor=-1, normalized=False, sorted=False, cellids=np.array([-1])):
        ## plot the average event rate of all cells in a given corridor
        ## if corridor == -1 then the first corridor is used
        ## normalisec: each cell ratemap is normalised to have a max = 1
        ## sorted: sorting the ramemaps by their peaks - not yet implemented
        ## cellids: np array with the indexes of the cells to be plotted. when -1: all cells are plotted
        
        rates = []
        vmax = 0

        ## we calculate the rate matrix for all corridors - we need to use the same colors for the images
        for corrid in np.unique(self.i_corridors):
            # select the laps in the corridor 
            # only laps with imaging data are selected - this will index the activity_tensor
            i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corrid)[0] 
            N_laps_corr = len(i_laps)

            if (cellids[0] == -1):
                cellids = np.array(range(self.activity_tensor.shape[1]))
            N_cells_plotted = len(cellids)

            time_matrix_1 = self.activity_tensor_time[:,i_laps]
            total_time = np.sum(time_matrix_1, axis=1) # bins x cells -> bins; time spent in each location

            act_tensor_1 = self.activity_tensor[:,:,i_laps] ## bin x cells x laps; all activity in all laps in corridor i
            act_tensor_2 = act_tensor_1[:,cellids,:] ##  bin x cells x laps; activity of the selected cells in all laps in corridor i
            total_spikes = np.sum(act_tensor_2, axis=2) ##  bin x cells; total activity of the selected cells in corridor i
            rate_matrix = np.zeros_like(total_spikes) ## event rate 
            
            for i_cell in range(N_cells_plotted):
            # for i_cell in range(total_spikes.shape[1]):
                if (normalized == True):
                    rate_i = total_spikes[:,i_cell] / total_time
                    rate_matrix[:,i_cell] = rate_i / np.max(rate_i)
                else :
                    rate_matrix[:,i_cell] = total_spikes[:,i_cell] / total_time

            if (np.max(rate_matrix) > vmax):
                vmax = np.max(rate_matrix)

            rates.append(rate_matrix)

        if (corridor == -1):
            corrids = np.unique(self.i_corridors)
            n_corrid = len(corrids)

            fig, axs = plt.subplots(1, n_corrid, figsize=(6+n_corrid*2,8))
            ims = []
            for i_corrid in range(n_corrid):
                if (N_cells_plotted == self.N_cells) :
                    title_string = 'ratemap of all cells in corridor ' + str(corrids[i_corrid])
                else :
                    title_string = 'ratemap of some cells in corridor ' + str(corrids[i_corrid])
                axs[i_corrid].set_title(title_string)
                ims.append(axs[i_corrid].matshow(np.transpose(rates[i_corrid]), aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='viridis'))
            
            plt.colorbar(ims[i_corrid])
            plt.show(block=False)       

        else:
            if (N_cells_plotted == self.N_cells) :
                title_string = 'ratemap of all cells in corridor ' + str(corridor)
            else :
                title_string = 'ratemap of some cells in corridor ' + str(corridor)

            fig, ax_bottom = plt.subplots(figsize=(6,8))
            i_corrid = int(np.nonzero(np.unique(self.i_corridors)==corridor)[0])
            ax_bottom.set_title(title_string)
            im1 = ax_bottom.matshow(np.transpose(rates[i_corrid]), aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='binary')
            plt.colorbar(im1)
            plt.show(block=False)       




        # imgplot = plt.imshow(lum_img)


    def plot_cell_laps(self, cellid, multipdf_object=-1, signal='dF', corridor=-1, reward=True, write_pdf=False):
        ## plot the activity of a single cell in all trials in a given corridor
        ## signal can be 
        ##          'dF' when dF/F and spikes are plotted as a function of time
        ##          'rate' when rate vs. position is plotted
        
        #process input corridors
        if (corridor == -1):
            corridor = np.unique(self.i_corridors)
        else:
            if corridor in np.unique(self.i_corridors):
                corridor=np.array(corridor)
            else:
                print('Warning: specified corridor does not exist in this session!')
                return
        
        #create title string for the given corridor, cell with added info 
        def CreateTitle(corridor):
            ##select the appropriate numpy array index to contain properties for userspecified corridor
            CI=-2#the specified corridor's index among nonzero corridors
            for corr in range(len(self.corridors)):
                if self.corridors[corr]==corridor:
                    CI=corr-1#always corridor 0 starts
                    cell_info='\n'+ 'skgs: '+str(round(self.cell_skaggs[CI][cellid],2))+' %actF: '+str(round(self.cell_activelaps_df[CI][cellid],2))+' %actS: '+str(round(self.cell_activelaps[CI][cellid],2))+'\n'+'TunSp: '+str(round(self.cell_tuning_specificity[CI][cellid],2))+' FF: '+str(round(self.cell_Fano_factor[CI][cellid],2))+' rate: '+str(round(self.cell_rates[CI][cellid],2))+' rel: '+str(round(self.cell_reliability[CI][cellid],2))    
                    break
            if CI==-2:
                print('Warning: specified corridor does not exist in this session!')
            return(cell_info)
        
        #plotting
        if (signal == 'dF'):
            fig, ax = plt.subplots(1,corridor.size,squeeze=False, figsize=(6*corridor.size,8), sharex=True)
            for cor_index in range(corridor.size):
                if corridor.size==1:
                    corridor_to_plot=corridor
                else:
                    corridor_to_plot=corridor[cor_index]
                cell_info=CreateTitle(corridor_to_plot)

                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps = self.i_Laps_ImData[np.nonzero(icorrids == corridor_to_plot)[0]]
                
                reward_times = []     
                dFs = []
                spikes = []
                times = []
                for i_lap in i_laps:
                    dFs.append(self.ImLaps[i_lap].frames_dF_F[cellid,:])
                    spikes.append(self.ImLaps[i_lap].frames_spikes[cellid,:])
                    tt = self.ImLaps[i_lap].frames_time
                    times.append(tt - np.min(tt))
                    reward_times.append(self.ImLaps[i_lap].reward_times - np.min(tt))
    
                colmap = plt.cm.get_cmap('jet')   
                colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
    #            fig, ax = plt.subplots(figsize=(6,8))
    
                n_laps = len(times)
                max_range = max(spikes[0])
                for i in range(n_laps):
                    if (max(spikes[i]) > max_range):
                        max_range = max(spikes[i])
    
                for i in range(n_laps):
                    ax[0,cor_index].plot(times[i], dFs[i] + i, alpha=0.5, c=colmap(np.remainder(10*i, 255)))
                    events = spikes[i]
                    events = 50 * events / max_range
                    ii_events = np.nonzero(events)[0]
                    ax[0,cor_index].scatter(times[i][ii_events], np.ones(len(ii_events)) * i, s=events[ii_events], cmap=colmap, c=(np.ones(len(ii_events)) * np.remainder(10*i, 255)), norm=colnorm)
                    if (reward == True): 
                        ax[0,cor_index].scatter(reward_times[i], np.repeat(i, len(reward_times[i])), marker="s", s=50, edgecolors=colmap(np.remainder(10*i, 255)), facecolors='none')
    
                ylab_string = 'dF_F, spikes (max: ' + str(np.round(max_range, 1)) +  ' )'
                ax[0,cor_index].set_ylabel(ylab_string)
                ax[0,cor_index].set_xlabel('time (s)')
                plot_title = 'dF/F of neuron ' + str(cellid) + ' in all laps in corridor ' + str(corridor_to_plot)+cell_info
                ax[0,cor_index].set_title(plot_title)
                ax[0,cor_index].set_ylim(0, n_laps+5)
            
            #write pdf if needed
            if write_pdf==True and multipdf_object!=-1:
                plt.savefig(multipdf_object, format='pdf')
            
            plt.show(block=False)
            
            if write_pdf==True:
                plt.close()

        if (signal == 'rate'):
            if corridor.size >1:
                #finding colormap ranges
                min_intensity=1000
                max_intensity=-1
                for cor_index in range(corridor.size):
                    corridor_to_plot=corridor[cor_index]
                    
                    #calculate rate matrix ...again :(
                    i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]
                    
                    total_spikes = self.activity_tensor[:,cellid,i_laps]
                    total_time = self.activity_tensor_time[:,i_laps]
                    rate_matrix = total_spikes / total_time
                    
                    loc_max=np.nanmax(rate_matrix[rate_matrix != np.inf])
                    loc_min=np.nanmin(rate_matrix)
                    if  loc_max > max_intensity :
                        max_intensity = loc_max
                    if  loc_min < min_intensity:
                        min_intensity = loc_min
                        
            #main part of ratemap plotting
            fig, ax = plt.subplots(2,corridor.size, squeeze=False, sharey='row', figsize=(6*corridor.size,8),sharex=True)
            for cor_index in range(corridor.size):
                if corridor.size==1:
                    corridor_to_plot=corridor
                else:
                    corridor_to_plot=corridor[cor_index]
                cell_info=CreateTitle(corridor_to_plot)   
                
                # getting rewarded corridors
                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps_beh = self.i_Laps_ImData[np.nonzero(icorrids == corridor_to_plot)[0]]
                reward_times = []
                for i_lap in i_laps_beh:
                    tt = self.ImLaps[i_lap].frames_time
                    reward_times.append(self.ImLaps[i_lap].reward_times - np.min(tt))
                
                # select the laps in the corridor (these are different indexes from upper ones!)
                # only laps with imaging data are selected - this will index the activity_tensor
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]               
                
                #calculate rate matrix
                total_spikes = self.activity_tensor[:,cellid,i_laps]
                total_time = self.activity_tensor_time[:,i_laps]
                rate_matrix = total_spikes / total_time

                #calculate for plotting average rates
                average_firing_rate=np.sum(rate_matrix, axis=1)/i_laps.size
                std=np.std(rate_matrix, axis=1)/np.sqrt(i_laps.size)
                errorbar_x=np.arange(50)
                
                #plotting
                title_string = 'ratemap of cell ' + str(cellid) + ' in corridor ' + str(corridor_to_plot)+cell_info
                ax[0,cor_index].set_title(title_string)
                ax[1,cor_index].fill_between(errorbar_x,average_firing_rate+std, average_firing_rate-std, alpha=0.3)
                ax[1,cor_index].plot(average_firing_rate,zorder=0)
                n_laps = rate_matrix.shape[1]

                if corridor.size>1:
                    im1 = ax[0,cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower',vmin=min_intensity, vmax=max_intensity)
                else:
                    im1 = ax[0,cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower')
                if (reward == True):
                    for i in range(n_laps):
                        if (len(reward_times[i]) > 0):
                            ax[0, cor_index].scatter(50, i, marker="s", edgecolors='r', facecolors='none', s=16)
                plt.colorbar(im1, orientation='horizontal',ax=ax[1,cor_index])
                ax[0,cor_index].set_xlim(0, 50)
                ax[1,cor_index].set_xlim(0, 50)
            
            #write pdf if asked
            if write_pdf==True and multipdf_object!=-1:
                plt.savefig(multipdf_object, format='pdf')
            
            plt.show(block=False)
            
            if write_pdf==True:
                plt.close()
            
    

    def plot_session(self):
        ## find the number of different corridors
        if (self.n_laps > 0):
            corridor_ids = np.zeros(self.n_laps)
            for i in range(self.n_laps):
                corridor_ids[i] = self.ImLaps[i].corridor
            corridor_types = np.unique(corridor_ids)
            nrow = len(corridor_types)
            nbins = len(self.ImLaps[0].bincenters)
            cmap = plt.cm.get_cmap('jet')   

            rowHeight = 2
            if (nrow > 4):
                rowHeight = 1.5
            fig, axs = plt.subplots(nrows=nrow, ncols=1, figsize=(8,rowHeight*nrow), squeeze=False)
            # plt.figure(figsize=(5,2*nrow))
            speed_color = cmap(30)
            speed_color_trial = (speed_color[0], speed_color[1], speed_color[2], (0.05))

            lick_color = cmap(200)
            lick_color_trial = (lick_color[0], lick_color[1], lick_color[2], (0.05))

            for row in range(nrow):
                # ax = plt.subplot(nrow, 1, row+1)
                ids = np.where(corridor_ids == corridor_types[row])
                avespeed = np.zeros(nbins)
                n_lap_bins = np.zeros(nbins) # number of laps in a given bin (data might be NAN for some laps)
                n_laps = np.shape(ids)[1]
                maxspeed = 200
                for lap in np.nditer(ids):
                    axs[row,0].step(self.ImLaps[lap].bincenters, self.ImLaps[lap].ave_speed, where='mid', c=speed_color_trial)
                    nans_lap = np.isnan(self.ImLaps[lap].ave_speed)
                    avespeed = nan_add(avespeed, self.ImLaps[lap].ave_speed)
                    n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                    if (max(self.ImLaps[lap].ave_speed) > maxspeed): maxspeed = max(self.ImLaps[lap].ave_speed)
                maxspeed = min(maxspeed, 2000)
                
                avespeed = nan_divide(avespeed, n_lap_bins, n_lap_bins > 0)
                axs[row,0].step(self.ImLaps[lap].bincenters, avespeed, where='mid', c=speed_color)
                axs[row,0].set_ylim([-1,1.2*maxspeed])

                if (row == 0):
                    if (self.sessionID >= 0):
                        plot_title = 'session:' + str(self.sessionID) + ': ' + str(int(n_laps)) + ' laps in corridor ' + str(int(corridor_types[row]))
                    else:
                        plot_title = str(int(n_laps)) + ' laps in corridor ' + str(int(corridor_types[row]))                    
                else:
                    plot_title = str(int(n_laps)) + ' laps in corridor ' + str(int(corridor_types[row]))

                if (self.ImLaps[lap].zones.shape[1] > 0):
                    bottom, top = axs[row,0].get_ylim()
                    left = self.ImLaps[lap].zones[0,0] * 3500
                    right = self.ImLaps[lap].zones[1,0] * 3500

                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                    axs[row,0].add_patch(polygon)
                    n_zones = np.shape(self.ImLaps[lap].zones)[1]
                    if (n_zones > 1):
                        for i in range(1, np.shape(self.ImLaps[lap].zones)[1]):
                            left = self.ImLaps[lap].zones[0,i] * 3500
                            right = self.ImLaps[lap].zones[1,i] * 3500
                            polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                            axs[row,0].add_patch(polygon)
                    # else: ## test for lick rate changes before the zone
                    #     self.anticipatory = np.zeros([2,n_laps])
                    #     k = 0
                    #     for lap in np.nditer(ids):
                    #         self.anticipatory[:,k] = self.ImLaps[lap].preZoneRate
                    #         k = k + 1
                    else: # we look for anticipatory licking tests
                        P_statement = ', anticipatory P value not tested'
                        for k in range(len(self.anticipatory)):
                            if (self.anticipatory[k].corridor == corridor_types[row]):
                                P_statement = ', anticipatory P = ' + str(round(self.anticipatory[k].test[1],5))
                        plot_title = plot_title + P_statement

                axs[row,0].set_title(plot_title)

                ax2 = axs[row,0].twinx()
                n_lap_bins = np.zeros(nbins) # number of laps in a given bin (data might be NAN for some laps)
                maxrate = 10
                avelick = np.zeros(nbins)
                for lap in np.nditer(ids):
                    ax2.step(self.ImLaps[lap].bincenters, self.ImLaps[lap].lick_rate, where='mid', c=lick_color_trial, linewidth=1)
                    nans_lap = np.isnan(self.ImLaps[lap].lick_rate)
                    avelick = nan_add(avelick, self.ImLaps[lap].lick_rate)
                    n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                    if (np.nanmax(self.ImLaps[lap].lick_rate) > maxrate): maxrate = np.nanmax(self.ImLaps[lap].lick_rate)
                maxrate = min(maxrate, 20)

                avelick = nan_divide(avelick, n_lap_bins, n_lap_bins > 0)
                ax2.step(self.ImLaps[lap].bincenters, avelick, where='mid', c=lick_color)
                ax2.set_ylim([-1,1.2*maxrate])


                if (row==(nrow-1)):
                    axs[row,0].set_ylabel('speed (roxels/s)', color=speed_color)
                    axs[row,0].tick_params(axis='y', labelcolor=speed_color)
                    ax2.set_ylabel('lick rate (lick/s)', color=lick_color)
                    ax2.tick_params(axis='y', labelcolor=lick_color)
                    axs[row,0].set_xlabel('position (roxel)')
                else:
                    axs[row,0].set_xticklabels([])
                    axs[row,0].tick_params(axis='y', labelcolor=speed_color)
                    ax2.tick_params(axis='y', labelcolor=lick_color)

            plt.show(block=False)
        else:
            fig = plt.figure(figsize=(8,3))
            # plt.scatter([-4, -3, -2], [2,3,4])
            plt.title('No data to show')
            plt.show(block=False)


def vcorrcoef(X,y):
    # correlation between the rows of the matrix X with dimensions (N x k) and a vector y of size (1 x k)
    # about 200 times faster than calculating correlations row by row
    Xm = np.reshape(np.nanmean(X,axis=1),(X.shape[0],1))
    i_nonzero = np.nonzero(Xm[:,0] != 0)[0]
    X_nz = X[i_nonzero,:]
    Xm_nz = Xm[i_nonzero,:]

    ym = np.nanmean(y)
    r_num = np.nansum((X_nz-Xm_nz)*(y-ym),axis=1)
    r_den = np.sqrt(np.nansum((X_nz-Xm_nz)**2,axis=1)*np.nansum((y-ym)**2))
    r = r_num/r_den
    return r

def nan_divide(a, b, where=True):
    'division function that returns np.nan where the division is not defined'
    x = np.zeros_like(a)
    x.fill(np.nan)
    x = np.divide(a, b, out=x, where=where)
    return x

def nan_add(a, b):
    'addition function that handles NANs by replacing them with zero - USE with CAUTION!'
    a[np.isnan(a)] = 0
    b[np.isnan(b)] = 0
    x = np.array(a + b)
    return x

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class Lap_ImData:
    'common base class for individual laps'

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time, corridor_list, dt=0.01, printout=False):
        self.name = name
        self.lap = lap

        self.raw_time = laptime
        self.raw_position = position
        self.lick_times = lick_times
        self.reward_times = reward_times
        self.corridor = corridor # the ID of the corridor in the given stage; This indexes the corridors in the vector called self.corridors
        self.corridor_list = corridor_list # the ID of the corridor in the given stage; This indexes the corridors in the vector called self.corridors
        self.mode = mode # 1 if all elements are recorded in 'Go' mode
        self.actions = actions

        self.zones = np.vstack([np.array(self.corridor_list.corridors[self.corridor].reward_zone_starts), np.array(self.corridor_list.corridors[self.corridor].reward_zone_ends)])
        self.n_zones = np.shape(self.zones)[1]
        self.preZoneRate = [None, None] # only if 1 lick zone; Compare the 210 roxels just before the zone with the preceeding 210 

        self.dt = 0.01 # resampling frequency = 100 Hz
        # approximate frame period for imaging - 0.033602467
        # only use it to convert spikes to rates!
        self.dt_imaging = 0.033602467

        self.frames_dF_F = lap_frames_dF_F
        self.frames_spikes = lap_frames_spikes
        self.frames_pos = lap_frames_pos
        self.frames_time = lap_frames_time
        self.min_N_frames = 50 # we analyse imaging data if there are at least 50 frames in the lap
        self.n_cells = 1 # we still create the same np arrays even if there are no cells
        if (not(np.isnan(self.frames_time).any())): # we real data
            if (len(self.frames_time) > self.min_N_frames): # we have more than 50 frames
                self.n_cells = self.frames_dF_F.shape[0]
            
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
    
            ## smooth the position data with a 50 ms Gaussian kernel
            # sdfilt = 0.05
            # xfilt = np.arange(-4*sdfilt, 4*sdfilt+self.dt, self.dt)
            # filt = np.exp(-(xfilt ** 2) / (2 * (sdfilt**2)))
            # filt = filt  / sum(filt)

            # dx1 = ppos[1] - ppos[0]
            # dxx1 = ppos[0] - np.arange(20, 0, -1) * dx1

            # dx2 = ppos[-1] - ppos[-2]
            # dxx2 = ppos[-1] + np.arange(20, 0, -1) * dx2

            # pppos = np.hstack([dxx1, ppos, dxx2])
            # pppos = np.hstack([np.repeat(ppos[0], 20), ppos, np.repeat(ppos[-1], 20)])
            # smooth_position = np.convolve(pppos, filt, mode='valid')
            self.smooth_position = ppos
            
            ## calculate the smoothed speed 
            # self.speed = np.diff(np.hstack([self.smooth_position[0], self.smooth_position])) / self.dt # roxel [=rotational pixel] / s       
            speed = np.diff(self.smooth_position) / self.dt # roxel [=rotational pixel] / s       
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
            self.spks_pos = np.zeros((self.n_cells, 50)) # sum of spike counts measured at a given position
            self.event_rate = np.zeros((self.n_cells, 50)) # spike rate 
            # self.dF_F_pos = np.zeros((self.n_cells, 50))
            if (not(np.isnan(self.frames_time).any())): # we real data
                if (len(self.frames_time) > self.min_N_frames): # we have more than 50 frames
                    for i_frame in range(len(self.frames_pos)):
                        ind_bin = int(self.frames_pos[i_frame] // 70)
                        ### we need to multiply the values with dt_imaging as this converts probilities to expected counts
                        self.spks_pos[:,ind_bin] = self.spks_pos[:,ind_bin] + self.frames_spikes[:,i_frame] * self.dt_imaging
                    for ind_bin in range(50):
                        if (self.T_pos[ind_bin] > 0): # otherwise the rate will remain 0
                            self.event_rate[:,ind_bin] = self.spks_pos[:,ind_bin] / self.T_pos[ind_bin]


            ####################################################################
            ## Calculate the lick rate befor the reward zone - anticipatory licks 210 roxels before zone start
            ## only when the number of zones is 1!
    
            if (self.n_zones == 1):
    
                zone_start = int(self.zones[0][0] * 3500)
                lz_posbins = [0, zone_start-420, zone_start-210, zone_start, 3500]
    
                lz_bin_counts = np.zeros(4)
                for pos in self.smooth_position:
                    bin_number = [ n for n,i in enumerate(lz_posbins) if i>=pos ][0] - 1
                    lz_bin_counts[bin_number] += 1
                T_lz_pos = lz_bin_counts * self.dt
    
                lz_lbin_counts = np.zeros(4)
                for lpos in self.lick_position:
                    lbin_number = [ n for n,i in enumerate(lz_posbins) if i>=lpos ][0] - 1
                    lz_lbin_counts[lbin_number] += 1
                lz_lick_rate = nan_divide(lz_lbin_counts, T_lz_pos, where=(T_lz_pos>0.025))
                self.preZoneRate = [lz_lick_rate[1], lz_lick_rate[2]]
        else:
            self.lick_position = lick_times
            self.reward_position = reward_times
            self.smooth_position = position
            self.speed = np.zeros(len(position))
            self.T_pos = np.zeros(50)
            self.N_licks = np.zeros(50)
            self.ave_speed = np.zeros(50)
            self.lick_rate = np.zeros(50)
            self.preZoneRate = np.zeros(2)
                

    def plot_tx(self, fluo=False, th=25):
        colmap = plt.cm.get_cmap('jet')   
        colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios': [1, 3]})

        ## first, plot position versus time
        ax_top.plot(self.laptime, self.smooth_position, c=colmap(50))
        ax_top.plot(self.raw_time, self.raw_position, c=colmap(90))

        ax_top.scatter(self.lick_times, np.repeat(self.smooth_position.min(), len(self.lick_times)), marker="|", s=100, c=colmap(180))
        ax_top.scatter(self.reward_times, np.repeat(self.smooth_position.min()+100, len(self.reward_times)), marker="|", s=100, c=colmap(230))
        ax_top.set_ylabel('position')
        ax_top.set_xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)
        ax_top.set_ylim(0, 3500)

        ## next, plot dF/F versus time (or spikes)
        if (self.n_cells > 1):
            # act_cells = np.nonzero(np.amax(self.frames_dF_F, 1) > th)[0]
            act_cells = np.nonzero(np.amax(self.frames_spikes, 1) > th)[0]
            max_range = np.nanmax(self.event_rate)
            for i in range(self.n_cells):
            # for i in (252, 258, 275):
                if (fluo & (i in act_cells)):
                    ax_bottom.plot(self.frames_time, self.frames_dF_F[i,:] + i, alpha=0.5, c=colmap(np.remainder(i, 255)))
                events = self.frames_spikes[i,:]
                events = 50 * events / max_range
                ii_events = np.nonzero(events)[0]
                ax_bottom.scatter(self.frames_time[ii_events], np.ones(len(ii_events)) * i, s=events[ii_events], cmap=colmap, c=(np.ones(len(ii_events)) * np.remainder(i, 255)), norm=colnorm)

            ylab_string = 'dF_F, spikes (max: ' + str(np.round(max_range, 1)) +  ' )'
            ax_bottom.set_ylabel(ylab_string)
            ax_bottom.set_xlabel('time (s)')
            plot_title = 'dF/F of all neurons  in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
            ax_bottom.set_title(plot_title)
            ax_bottom.set_ylim(0, self.n_cells+5)
            # ax_bottom.set_ylim(250, 280)

        plt.show(block=False)
       
        # laptime = D1.ImLaps[200].laptime
        # smooth_position = D1.ImLaps[200].smooth_position
        # lick_times = D1.ImLaps[200].lick_times
        # reward_times = D1.ImLaps[200].reward_times
        # lap = D1.ImLaps[200].lap
        # corridor = D1.ImLaps[200].corridor
        # lick_rate = D1.ImLaps[200].lick_rate
        # n_cells = D1.ImLaps[200].n_cells
        # frames_time = D1.ImLaps[200].frames_time
        # frames_dF_F = D1.ImLaps[200].frames_dF_F
        # frames_spikes = D1.ImLaps[200].frames_spikes
        # frames_pos = D1.ImLaps[200].frames_pos

        # fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios': [1, 3]})
        # ax_top.plot(laptime, smooth_position, c='g')

        # ax_top.scatter(lick_times, np.repeat(smooth_position.min(), len(lick_times)), marker="|", s=100)
        # ax_top.scatter(reward_times, np.repeat(smooth_position.min()+100, len(reward_times)), marker="|", s=100, c='r')
        # ax_top.set_ylabel('position')
        # ax_top.set_xlabel('time (s)')
        # plot_title = 'Mouse: ' + name + ' position in lap ' + str(lap) + ' in corridor ' + str(corridor)
        # ax_top.set_title(plot_title)

        # if (n_cells > 1):
        #     for i in (252, 258, 275):
        #         ax_bottom.plot(frames_time, frames_dF_F[i,:] + i, c=cmap(np.remainder(i, 255)))
        #         events = frames_spikes[i,:]
        #         events = 10 * events / max(events)
        #         ii_events = np.nonzero(events)[0]
        #         ax_bottom.scatter(frames_time[ii_events], np.ones(len(ii_events)) * i, s=events[ii_events], c=cmap(np.remainder(i, 255)))

        #     ax_bottom.set_ylabel('dF_F')
        #     ax_bottom.set_xlabel('time (s)')
        #     plot_title = 'dF/F of all neurons  in lap ' + str(lap) + ' in corridor ' + str(corridor)
        #     ax_bottom.set_title(plot_title)
        #     ax_bottom.set_ylim(250, 280)

        # plt.show(block=False)

    def plot_xv(self):
        colmap = plt.cm.get_cmap('jet')   
        colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios': [1, 3]})
        ax_top.plot(self.smooth_position, self.speed, c=colmap(80))
        ax_top.step(self.bincenters, self.ave_speed, where='mid', c=colmap(30))
        ax_top.scatter(self.lick_position, np.repeat(100, len(self.lick_position)), marker="|", s=100, c=colmap(180))
        ax_top.scatter(self.reward_position, np.repeat(200, len(self.reward_position)), marker="|", s=100, c=colmap(230))
        ax_top.set_ylabel('speed (roxel/s)')
        ax_top.set_ylim([min(0, self.speed.min()), max(self.speed.max(), 1000)])
        ax_top.set_xlabel('position')
        plot_title = 'Mouse: ' + self.name + ' speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)


        bottom, top = ax_top.get_ylim()
        left = self.zones[0,0] * 3500
        right = self.zones[1,0] * 3500

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        ax_top.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * 3500
                right = self.zones[1,i] * 3500
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                ax_top.add_patch(polygon)

        ax2 = ax_top.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', c=colmap(200), linewidth=1)
        ax2.set_ylabel('lick rate (lick/s)', color=colmap(200))
        ax2.tick_params(axis='y', labelcolor=colmap(200))
        ax2.set_ylim([-1,2*np.nanmax(self.lick_rate)])

        ## next, plot event rates versus space
        if (self.n_cells > 1):
            max_range = np.nanmax(self.event_rate)
            # for i in np.arange(250, 280):
            for i in range(self.n_cells):
                events = self.event_rate[i,:]
                events = 50 * events / max_range
                ii_events = np.nonzero(events)[0]
                ax_bottom.scatter(self.bincenters[ii_events], np.ones(len(ii_events)) * i, s=events[ii_events], cmap=colmap, c=(np.ones(len(ii_events)) * np.remainder(i, 255)), norm=colnorm)

            ax_bottom.set_ylabel('event rate')
            ax_bottom.set_xlabel('position')
            plot_title = 'event rate of all neurons  in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
            ax_bottom.set_title(plot_title)
            # ax_bottom.set_ylim(250, 280)
            ax_bottom.set_ylim(0, self.n_cells)

        plt.show(block=False)       


        colmap = plt.cm.get_cmap('jet')
        colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
        x = np.random.rand(4)
        y = np.random.rand(4)
        area = (np.abs(x/max(x))*30)**2
        colors = 232

        # plt.scatter(x, y, s=area, c=np.ones(4)*colors, cmap=colmap)
        # plt.scatter(x, y, s=area, c=np.ones(4)*110, cmap=colmap, norm=colnorm)
        # plt.figure(figsize=(4, 4))
        # # plt.scatter(x, y, s=area, c=[3,45,112,213], alpha=0.7, cmap=colmap, norm=colnorm)
        # plt.plot(x, y, alpha=0.7, c=colmap(23))
        # plt.show(block=False)


        # plt.figure(figsize=(4, 4))
        # plt.plot(range(len(sk_old)), sk_old, 'rs', range(len(sk_D1)), sk_D1, 'g^')
        # plt.show(block=False)

        # np.nonzero(np.amax(D1.F_all, 1) > 700)[0]
        # colmap = plt.cm.get_cmap('jet')   
        # fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10,8), gridspec_kw={'height_ratios': [1, 3]})
        # ax_top.plot(range(17858), D1.F_all[734,])
        
        # act_cells = np.nonzero(np.amax(F_all[:,6000:9000], 1) > 700)[0]
        # k = 0
        # for i in range(1394):
        # # for i in (252, 258, 275):
        #   if (i in act_cells):
        #       ii = np.random.randint(255)
        #       ax_bottom.plot(range(17070), F_all[i,] + k, alpha=0.75, c=colmap(ii))
        #       k = k + 500
        # plt.show(block=False)

        # # smooth_position = D1.ImLaps[200].smooth_position
        # speed = D1.ImLaps[200].speed
        # lick_position = D1.ImLaps[200].lick_position
        # lick_times = D1.ImLaps[200].lick_times
        # reward_position = D1.ImLaps[200].reward_position
        # reward_times = D1.ImLaps[200].reward_times
        # lap = D1.ImLaps[200].lap
        # corridor = D1.ImLaps[200].corridor
        # lick_rate = D1.ImLaps[200].lick_rate
        # ave_speed = D1.ImLaps[200].ave_speed
        # zones = D1.ImLaps[0].zones
        # bincenters = np.arange(0, 3500, 70) + 70 / 2.0
        # event_rate = D1.ImLaps[200].event_rate
        # n_cells = D1.ImLaps[200].n_cells

        # fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios': [1, 3]})
        # ax_top.plot(smooth_position, speed, c=cmap(80))
        # ax_top.plot(bincenters, ave_speed, c=cmap(30))
        # ax_top.scatter(lick_position, np.repeat(speed.min(), len(lick_position)), marker="|", s=100, c=cmap(180))
        # ax_top.scatter(reward_position, np.repeat(speed.min(), len(reward_position)), marker="|", s=100, c=cmap(230))
        # ax_top.set_ylabel('speed (roxel/s)')
        # ax_top.set_xlabel('position')
        # plot_title = 'Mouse: ' + name + ' speed in lap ' + str(lap) + ' in corridor ' + str(corridor)
        # ax_top.set_title(plot_title)

        # bottom, top = ax_top.get_ylim()
        # left = zones[0,0] * 3500
        # right = zones[1,0] * 3500

        # polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        # ax_top.add_patch(polygon)
        # if (np.shape(zones)[1] > 1):
        #     for i in range(1, np.shape(zones)[1]):
        #         left = zones[0,i] * 3500
        #         right = zones[1,i] * 3500
        #         polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        #         ax_top.add_patch(polygon)


        # ax2 = ax_top.twinx()
        # ax2.step(bincenters, lick_rate, c=cmap(200), linewidth=1)
        # ax2.set_ylabel('lick rate', color=cmap(200))
        # ax2.tick_params(axis='y', labelcolor=cmap(200))
        # ax2.set_ylim([-1,2*max(lick_rate)])

        # next, plot dF/F versus time (or spikes)
        # if (n_cells > 1):
        #     for i in np.arange(250, 280):
        #         ax_bottom.scatter(bincenters, np.ones(50) * i, s=event_rate[i,], c=cmap(np.remainder(i, 255)))
        #     ax_bottom.set_ylabel('event rate')
        #     ax_bottom.set_xlabel('position')
        #     plot_title = 'event rate of all neurons  in lap ' + str(lap) + ' in corridor ' + str(corridor)
        #     ax_bottom.set_title(plot_title)
        #     ax_bottom.set_ylim(250, 280)


        # plt.show(block=False)       


    def plot_txv(self):
        cmap = plt.cm.get_cmap('jet')   
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,6))

        ## first, plot position versus time
        ax_top.plot(self.laptime, self.smooth_position, c=cmap(50))
        ax_top.plot(self.raw_time, self.raw_position, c=cmap(90))

        ax_top.scatter(self.lick_times, np.repeat(self.smooth_position.min(), len(self.lick_times)), marker="|", s=100, c=cmap(180))
        ax_top.scatter(self.reward_times, np.repeat(self.smooth_position.min()+200, len(self.reward_times)), marker="|", s=100, c=cmap(230))
        ax_top.set_ylabel('position')
        ax_top.set_xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position and speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)
        ax_top.set_ylim(0, 3600)


        ## next, plot speed versus position
        ax_bottom.plot(self.smooth_position, self.speed, c=cmap(80))
        ax_bottom.step(self.bincenters, self.ave_speed, where='mid', c=cmap(30))
        ax_bottom.scatter(self.lick_position, np.repeat(100, len(self.lick_position)), marker="|", s=100, c=cmap(180))
        ax_bottom.scatter(self.reward_position, np.repeat(200, len(self.reward_position)), marker="|", s=100, c=cmap(230))
        ax_bottom.set_ylabel('speed (roxel/s)')
        ax_bottom.set_xlabel('position')
        ax_bottom.set_ylim([min(0, self.speed.min()), max(self.speed.max(), 1000)])

        bottom, top = plt.ylim()
        left = self.zones[0,0] * 3500
        right = self.zones[1,0] * 3500

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        ax_bottom.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * 3500
                right = self.zones[1,i] * 3500
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                ax_bottom.add_patch(polygon)

        ax2 = ax_bottom.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', c=cmap(230), linewidth=1)
        ax2.set_ylabel('lick rate (lick/s)', color=cmap(230))
        ax2.tick_params(axis='y', labelcolor=cmap(230))
        ax2.set_ylim([-1,2*np.nanmax(self.lick_rate)])

        plt.show(block=False)       




class anticipatory_Licks:
    'simple class for containing anticipatory licking data'
    def __init__(self, baseline_rate, anti_rate, corridor):
        nan_rates = np.isnan(baseline_rate) + np.isnan(anti_rate)
        baseline_rate = baseline_rate[np.logical_not(nan_rates)]
        anti_rate = anti_rate[np.logical_not(nan_rates)]
        self.baseline = baseline_rate
        self.anti_rate = anti_rate

        self.m_base = np.mean(self.baseline)
        self.m_anti = np.mean(self.anti_rate)
        if (self.m_base < self.m_anti):
            greater = True
        else:
            greater = False
        self.corridor = int(corridor)
        self.test = scipy.stats.wilcoxon(self.baseline, self.anti_rate)
        self.anti = False
        if ((self.test[1] < 0.01 ) & (greater == True)):
            self.anti = True

