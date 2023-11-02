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
from sys import platform
import copy
import time
import os
import pickle
#pickle.HIGHEST_PROTOCOL = 3
import functools
from xml.dom import minidom
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import pandas as pd

from utils import *
from Stages import *
from Corridors import *
from ImShuffle import *

import PlaceField as PF
from PlaceField import PlaceField
from BtspAnalysis import BtspAnalysis

if (platform == 'darwin'):
    csv_kwargs = {'delimiter':' '}
else:
    # csv_kwargs = {'delimiter':' ', 'newline':''}
    csv_kwargs = {'delimiter':' '}


class ImagingSessionData:
    'Base structure for both imaging and behaviour data'
    def __init__(self, datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=np.nan, selected_laps=None, speed_threshold=5, randseed=123, elfiz=False, reward_zones=None, spikes_tag='', data_folder='analysed_data'):
        self.datapath = datapath
        self.date_time = date_time
        self.name = name
        self.task = task
        self.suite2p_folder = suite2p_folder
        self.imaging_logfile_name = imaging_logfile_name
        self.multiplane = False

        self.stage = 0
        self.stages = []
        self.sessionID = sessionID
        self.randseed = randseed
        self.selected_laps = selected_laps
        self.speed_threshold = speed_threshold
        self.elfiz = elfiz
        self.minimum_Nlaps = 3
        self.substage_change_laps = [0]
        self.substage_change_time = [0]
        self.data_folder = data_folder

        stagefilename = self.datapath + self.task + '_stages.pkl'
        input_file = open(stagefilename, 'rb')
        if version_info.major == 2:
            self.stage_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.stage_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        corridorfilename = self.datapath + self.task + '_corridors.pkl'
        input_file = open(corridorfilename, 'rb')
        if version_info.major == 2:
            self.corridor_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.corridor_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        self.get_stage(self.datapath, self.date_time, self.name, self.task) # reads the stage of the experiment from the log file
        self.all_corridors = np.hstack([0, np.array(self.stage_list.stages[self.stage].corridors)])# we always add corridor 0 - that is the grey zone

        if (reward_zones is not None):
            corridors_changed = np.unique(reward_zones[:,0])
            for i_corrid in corridors_changed:
                zones_i = np.flatnonzero(reward_zones[:,0] == i_corrid)
                zones_start_corrid = reward_zones[zones_i,1]
                zones_end_corrid = reward_zones[zones_i,2]
                self.corridor_list.corridors[int(i_corrid)].reward_zone_starts = zones_start_corrid
                self.corridor_list.corridors[int(i_corrid)].reward_zone_ends = zones_end_corrid
                print ('reward zones added manually')

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
        ##################################################
        ## loading the trigger signals to match LabView and Imaging data axis
        ## CAPITAL LETTERS: variables defined with IMAGING time axis
        ## normal LETTERS: variables defined with LabView time axis
        ##################################################
        ### trigger log starts and durations

        ###############################
        ### this all will need to move to the LocateImaging routine to save memory
        ###############################

        beh_folder = self.datapath + 'data/' + name + '_' + self.task + '/' + date_time + '/'
        trigger_log_file_string = beh_folder + date_time + '_' + name + '_' + self.task + '_TriggerLog.txt'
                
        ## matching imaging time with labview time
        self.imstart_time = 0 # the labview time of the first imaging frame
        self.imstart_time = LocateImaging(trigger_log_file_string, TRIGGER_VOLTAGE_FILENAME)

        ##################################################
        ## loading imaging data
        ##################################################
        
        self.even_odd_rate_calculated = False
        self.start_end_rate_calculated = False
        
        if (self.elfiz):
            F_string = self.suite2p_folder + 'Vm_' + self.imaging_logfile_name + '.npy'
            spks_string = self.suite2p_folder + 'spikes_' + self.imaging_logfile_name + '.npy'
            time_string = self.suite2p_folder + 'frame_times_' + self.imaging_logfile_name + '.npy'

            self.F = np.load(F_string) # npy array, N_ROI x N_frames, fluorescence traces of ROIs from suite2p
            self.raw_spks = np.load(spks_string) # npy array, N_ROI x N_frames, spike events detected from suite2p
            print('elfiz data loaded')       

            self.frame_times = np.load(time_string)[0] + self.imstart_time
            print('elfiz time axis loaded')       
            self.frame_period = np.median(np.diff(self.frame_times))
            self.frame_pos = np.zeros(len(self.frame_times)) # position and 
            self.frame_laps = np.zeros(len(self.frame_times)) # lap number for the imaging frames, to be filled later


            ## arrays containing only valid cells
            self.dF_F = np.copy(self.F)
            self.spks = np.copy(self.raw_spks) # could be normalized in calc_dF_F: spks / F / SD(F)
            self.N_cells = self.F.shape[0]
            # cell_SDs = np.sqrt(np.var(dF_F, 1)) - we calculate this later
            self.cell_SDs = np.zeros(self.N_cells) # a vector with the SD of the cells
            self.cell_SNR = np.zeros(self.N_cells) # a vector with the signal to noise ratio of the cells (max F / SD)
            self.calc_SD_SNR_elfiz()
            self.calc_active_elfiz()
        else :
            F_string = self.suite2p_folder + 'F.npy'
            # Fneu_string = self.suite2p_folder + 'Fneu.npy'
            spks_string = self.suite2p_folder + 'spks' + spikes_tag + '.npy'
            iscell_string = self.suite2p_folder + 'iscell.npy'
            
            self.F_all = np.load(F_string) # npy array, N_ROI x N_frames, fluorescence traces of ROIs from suite2p
            # self.Fneu = np.load(Fneu_string) # npy array, N_ROI x N_frames, fluorescence traces of neuropil from suite2p
            self.spks_all = np.load(spks_string) # npy array, N_ROI x N_frames, spike events detected from suite2p
            self.iscell = np.load(iscell_string) # np array, N_ROI x 2, 1st col: binary classified as cell. 2nd P_cell?
            self.stat_string = self.suite2p_folder + 'stat.npy' # we may load these later if needed
            self.ops_string = self.suite2p_folder + 'ops.npy'
            print('suite2p data loaded')               

            self.frame_times = np.nan # labview coordinates
            imtimes_success = self.LoadImaging_times(self.imstart_time)
            if (imtimes_success == False):
                return 
            self.frame_period = np.median(np.diff(self.frame_times))
            self.frame_pos = np.zeros(len(self.frame_times)) # position and 
            self.frame_laps = np.zeros(len(self.frame_times)) # lap number for the imaging frames, to be filled later
            print('suite2p time axis loaded')       

            ## arrays containing only valid cells
            self.neuron_index = np.nonzero(self.iscell[:,0])[0]
            self.F = self.F_all[self.neuron_index,:]
            self.raw_spks = self.spks_all[self.neuron_index,:]
            self.dF_F = np.copy(self.F)
            self.spks = np.copy(self.raw_spks) # could be normalized in calc_dF_F: spks / F / SD(F)
            self.N_cells = self.F.shape[0]
            # self.cell_SDs = np.sqrt(np.var(self.dF_F, 1)) - we calculate this later
            self.cell_SDs = np.zeros(self.N_cells) # a vector with the SD of the cells
            self.cell_SNR = np.zeros(self.N_cells) # a vector with the signal to noise ratio of the cells (max F / SD)
            self.calc_dF_F()
            self.detect_events()
          
        ##################################################
        ## loading behavioral data
        ##################################################

        self.ImLaps = [] # list containing a special class for storing the imaging and behavioral data for single laps
        self.n_laps = 0 # total number of laps
        self.i_Laps_ImData = np.zeros(1) # np array with the index of laps with imaging
        self.i_corridors = np.zeros(1) # np array with the index of corridors in each run

        self.get_lapdata(self.datapath, self.date_time, self.name, self.task, selected_laps=self.selected_laps) # collects all behavioral and imaging data and sort it into laps, storing each in a Lap_ImData object
        self.calc_active()
        if (self.n_laps == -1):
            print('Error: missing laps are found in the ExpStateMachineLog file! No analysis was performed, check the logfiles!')
            return
        print('laps with im data: ', self.i_Laps_ImData)

        if not elfiz:
            frame_time_lap_maze_pos = np.array((self.frame_times, self.frame_laps, self.frame_maze, self.frame_pos))
            time_lap_maze_pos_FILE = self.suite2p_folder + 'frame_time_lap_maze_pos.npy'
            np.save(time_lap_maze_pos_FILE, frame_time_lap_maze_pos)

        ## in certain tasks, the same corridor may appear multiple times in different substages
        ## we need to keep this corridor in the list self.corridors for running self.get_lapdata()
        ## but we should remove the redundancy after the data is loaded
        self.all_corridors = np.unique(self.all_corridors)
        self.N_all_corridors = len(self.all_corridors)

        ## only analyse corridors with at least 3 laps 
        # - the data still remains in the ImLaps list and will appear in the activity tensor!
        #   but the corridor will not 
        #   we also do NOT include corridor 0 here
        if (self.N_all_corridors > 1):
            # print('i laps with imaging data:', self.i_Laps_ImData)
            # print('i of corridors: ', self.i_corridors)
            corridors, N_laps_corr = np.unique(self.i_corridors[self.i_Laps_ImData], return_counts=True)
            self.corridors = corridors[np.flatnonzero(N_laps_corr >= self.minimum_Nlaps)]
            self.N_corridors = len(self.corridors)
        else :
            self.corridors = np.setdiff1d(self.all_corridors, 0)
            self.N_corridors = len(self.corridors)

        self.N_ImLaps = len(self.i_Laps_ImData)

        # calculate P-correct for each corridor
        self.Ps_correct = {}  # corridor_id : P_correct
        self.calc_correct_lap_proportions()

        # print('number of laps with imaging data:', self.N_ImLaps)
        self.raw_activity_tensor = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials containing the spikes
        self.raw_activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # a tensor with space x trials containing the time spent at each location in each lap
        self.activity_tensor = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # same as the activity tensor spatially smoothed
        self.activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # same as the activity tensor time spatially smoothed
        # print('laps with image data:')
        # print(self.i_Laps_ImData)
        self.combine_lapdata() ## fills in the cell_activity tensor

        self.cell_activelaps=[] # a list, each element is a vector with the % of significantly spiking laps of the cells in a corridor
        self.cell_Fano_factor = [] # a list, each element is a vector with the reliability of the cells in a corridor

        self.cell_reliability = [] # a list, each element is a vector with the reliability of the cells in a corridor
        self.cell_skaggs=[] # a list, each element is a vector with the skaggs93 spatial info of the cells in a corridor
        self.cell_tuning_specificity=[] # a list, each element is a vector with the tuning specificity of the cells in a corridor

        self.cell_rates = [] # a list, each element is a 1 x n_cells matrix with the average rate of the cells in the whole corridor
        self.cell_corridor_selectivity = np.zeros([2,self.N_cells]) # a matrix with the selectivity index of the cells. Second row indicates the corridor with the highers rate.

        self.ratemaps = [] # a list, each element is an array space x neurons x trials being the ratemaps of the cells in a given corridor
        self.cell_corridor_similarity = np.zeros(self.N_cells) # a vector with the similarity index of the cells.

        self.calculate_properties()

        self.candidate_PCs = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping
        self.Hainmuller_PCs()

        self.test_anticipatory()

        # BTSP analysis
        self.p95 = []
        self.btsp_analysis = None  # this will be a BtspAnalysis object after calling run_btsp_analysis()

        # lick and speed selectivity
        self.speed_selectivity_laps = {}
        self.lick_selectivity_laps = {}
        self.speed_selectivity_cross_corridor = np.nan
        self.lick_selectivity_cross_corridor = np.nan
        self.calc_speed_and_lick_selectivity()

    def get_analysis_ID(self, s2p_ids):
        # map suite2p ids to analysis ids - writes results to console and returns them
        s2p_ids = np.array(s2p_ids) 
        if np.nonzero(s2p_ids>self.iscell.shape[0])[0].size !=0:
            print("Too large input ID found!")
            return 
        helper_array = np.zeros(self.iscell.shape[0]) 
        new_index = 0 
        for i in range(self.iscell.shape[0]): 
            if self.iscell[i,0]==1: 
                helper_array[i] = new_index 
                new_index += 1 
            else: 
                # helper_array[i] = np.nan
                helper_array[i] = -1
        helper_array = np.array([int(x) for x in helper_array])
        print(helper_array)
        print(helper_array[s2p_ids])
        return helper_array[s2p_ids]
    
    def get_suite2p_ID(self, cellids):
        # map analysis ids to suite2p ids - writes results to console and returns them
        cellids = np.array(cellids)
        print(self.neuron_index[cellids])
        return self.neuron_index[cellids]
    
    def write_params(self, filename):
        # write the parameters of the current ImagingSessionData into the given file

        data_folder = self.suite2p_folder + self.data_folder
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        param_filename = data_folder + '/' + filename
        with open(param_filename, mode='w', newline='') as paramfile:
            file_writer = csv.writer(paramfile, **csv_kwargs)
            file_writer.writerow(('#name:', self.name))
            file_writer.writerow(('#task:', self.task))
            file_writer.writerow(('#date_time:', self.date_time))
            file_writer.writerow(('#datapath:', self.datapath))
            file_writer.writerow(('#suite2p_folder:', self.suite2p_folder))
            file_writer.writerow(('#stage:', self.stage))
            file_writer.writerow(('#randseed:', self.randseed))
            file_writer.writerow(('#selected_laps:', str(self.selected_laps)))
            file_writer.writerow(('#speed_threshold:', self.speed_threshold))

        print('Session parameters written into file: ', filename)

    def check_params(self, filename):
        # read the parameters from the file and compare it to the current ImagingSessionData 
        data_folder = self.suite2p_folder + self.data_folder
        if not os.path.exists(data_folder):
            print ('Error: data directory', data_folder, 'does NOT exist!')    
            return False

        param_filename = data_folder + '/' + filename
        param_file=open(param_filename, newline='')
        param_file_reader=csv.reader(param_file, delimiter=' ')

        name = next(param_file_reader)
        if (name[1] != self.name):
            print('Error: name read from file not equals the session name!')
            return False

        task = next(param_file_reader)
        if (task[1] != self.task):
            print('Error: task read from file not equals the session task!')
            return False

        date_time = next(param_file_reader)
        if (date_time[1] != self.date_time):
            print('Error: date_time read from file not equals the date_time in the loaded session!')
            return False

        datapath = next(param_file_reader)
        # if (datapath[1] != self.datapath):
        #     print('Error: datapath read from file not equals the datapath in the loaded session!')
        #     return False

        suite2p_folder = next(param_file_reader)
        # if (suite2p_folder[1] != self.suite2p_folder):
        #     print('Error: suite2p_folder read from file not equals the suite2p_folder in the loaded session!')
        #     return False

        stage = next(param_file_reader)
        if (int(stage[1]) != self.stage):
            print('Error: stage read from file not equals the stage in the loaded session!')
            return False

        randseed = next(param_file_reader)
        if (int(randseed[1]) != self.randseed):
            print('Error: randseed read from file not equals the randseed in the loaded session!')
            return False

        selected_laps = next(param_file_reader)
        if self.selected_laps is None:
            if (selected_laps[1] != 'None'):
                print('Error: selected_laps read from file not equals the selected_laps in the loaded session!')
                return False
        elif (selected_laps[1] == 'None'):
            if self.selected_laps is not None:            
                print('Error: selected_laps read from file not equals the selected_laps in the loaded session!')
                return False
        else:
            L = len(selected_laps[1])-1
            sel_laps = np.fromstring(selected_laps[1][1:L], dtype=int, sep=' ')
            diff_laps = np.setdiff1d(sel_laps, self.selected_laps)
            if (len(diff_laps) > 0):
                print('Error: selected_laps read from file not equals the selected_laps in the loaded session!')
                return False

        speed_threshold = next(param_file_reader)
        if (int(speed_threshold[1]) != self.speed_threshold):
            print('Error: speed_threshold read from file not equals the speed_threshold in the loaded session!')
            return False 

        return True

    def get_stage(self, datapath, date_time, name, task):
        # function that reads the action_log_file and finds the current stage
        action_log_file_string=datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_UserActionLog.txt'
        action_log_file=open(action_log_file_string, newline='')
        log_file_reader=csv.reader(action_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            if (line[1] == 'Stage'):
                self.stage = int(round(float(line[2])))

    def LoadImaging_times(self, offset):
        # function that reads the action_log_file and finds the current stage
        # minidom is an xml file interpreter for python
        # hope it works for python 3.7...
        imaging_logfile = minidom.parse(self.imaging_logfile_name)
        voltage_rec = imaging_logfile.getElementsByTagName('VoltageRecording')
        voltage_delay = float(voltage_rec[0].attributes['absoluteTime'].value)
        ## the offset is the time of the first voltage signal in Labview time
        ## the signal's 0 has a slight delay compared to the time 0 of the imaging recording 
        ## we substract this delay from the offset to get the LabView time of the time 0 of the imaging recording
        corrected_offset = offset - voltage_delay
        print('corrected offset:', corrected_offset, 'voltage_delay:', voltage_delay)  
        
        #find out whether it's a multiplane recording
        sequence = imaging_logfile.getElementsByTagName('Sequence')
        frames = imaging_logfile.getElementsByTagName('Frame')
        if sequence[0].attributes['type'].value == 'TSeries ZSeries Element':
            print('multi-plane')
            self.multiplane = True
            #for multiplane recordings we drop last frame as it is sometimes 'missing' for one of the planes
            self.F_all = self.F_all[:, 0:-1]
            self.spks_all = self.spks_all[:, 0:-1]
            # self.Fneu = self.Fneu[:, 0:-1]
            if len(frames) %2 == 0:    
                len_frames_used = int(len(frames)/2-1)
            if len(frames) %2 == 1:
                len_frames_used = int((len(frames)-1)/2)
            # for frame time we use the average of the two planes time
            self.frame_times = np.zeros(len_frames_used)
            self.im_reftime = float(frames[1].attributes['relativeTime'].value) - float(frames[1].attributes['absoluteTime'].value)
            for i in range(len_frames_used): ## checkit: why i and i+1 and not 2i and 2i - 1?
                self.frame_times[i] = (float(frames[2*i].attributes['relativeTime'].value) + float(frames[2*i+1].attributes['relativeTime'].value))/2 + corrected_offset
            
        else:
            print('single-plane')

            self.frame_times = np.zeros(len(frames)) # this is already in labview time
            self.im_reftime = float(frames[1].attributes['relativeTime'].value) - float(frames[1].attributes['absoluteTime'].value)
            for i in range(len(frames)):
                self.frame_times[i] = float(frames[i].attributes['relativeTime'].value) + corrected_offset       
        
        if (len(self.frame_times) != self.F_all.shape[1]):
            print('ERROR: imaging frame number does not match suite2p frame number! Something is wrong!')
            print('shape of the dF array:', self.F_all.shape)
            print('shape of spikes array:', self.spks_all.shape[1])
            print('suite2p frame number:', self.F_all.shape[1])
            print('imaging frame number:', len(self.frame_times))
            raise ValueError('ERROR: imaging frame number does not match suite2p frame number! Something is wrong!')

            # N_frames = min([len(self.frame_times), self.F_all.shape[1]])
            # if (len(self.frame_times) < self.F_all.shape[1]):
            #     self.F_all = self.F_all[:, 0:N_frames]
            #     self.spks_all = self.spks_all[:, 0:N_frames]
            #     # self.Fneu = self.Fneu[:, 0:N_frames]
            # else:
            #     self.frame_times = self.frame_times[0:N_frames]   


    def LoadExpLog(self, exp_log_file_string): # BBU: just reads the raw data, no separation into laps
        position=[]
        exp_loop_timestamps=[]
        mazeID=[]
        exp_log_file=open(exp_log_file_string,'r', newline='')
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
        self.anticipatory = []
        print('testing anticipatory licks...')
        print('N_corridors: ', self.N_corridors)
        print('corridors: ', self.corridors)

        for row in range(self.N_corridors):
            ids = np.flatnonzero(self.i_corridors == self.corridors[row])
            n_laps = len(ids)
            n_zones = np.shape(self.ImLaps[ids[0]].zones)[1]
            if (n_zones == 1):
                lick_rates = np.zeros([2,n_laps])
                k = 0
                for lap in np.nditer(ids):
                    if (self.ImLaps[lap].mode == 1):
                        lick_rates[:,k] = self.ImLaps[lap].preZoneRate
                    else:
                        lick_rates[:,k] = np.nan
                    k = k + 1
                self.anticipatory.append(anticipatory_Licks(lick_rates[0,:], lick_rates[1,:], self.corridors[row]))



    def calc_dF_F(self):
        print('calculating dF/F and SNR...')

        self.cell_SDs = np.zeros(self.N_cells) # a vector with the SD of the cells
        self.cell_SNR = np.zeros(self.N_cells) # a vector with the signal to noise ratio of the cells (max F / SD)
        self.cell_baselines = np.zeros(self.N_cells) # a vector with the baseline F of the cells

        ## to calculate the SD and SNR, we need baseline periods with no spikes for at least 1 sec
        frame_rate = int(np.ceil(1/self.frame_period))
        sp_threshold = 20 # 
        T_after_spike = 3 #s 
        T_before_spike = 0.5 #s 
        Tmin_no_spike = 1 #s 
        L_after_spike = int(round(T_after_spike  * frame_rate))
        L_before_spike = int(round(T_before_spike  * frame_rate))
        Lmin_no_spike = int(round(Tmin_no_spike * frame_rate ))

        N_frames = len(self.frame_times) 
        filt = np.ones(frame_rate)

        #calculate baseline
        for i_cell in range(self.N_cells):
            
            # baseline: mode of the histogram
            trace=self.F[i_cell,]
            hist=np.histogram(trace, bins=100)
            max_index = np.where(hist[0] == max(hist[0]))[0][0]
            baseline = hist[1][max_index]
            # if (baseline == 0): 
            #     baseline = hist[1][max_index+1]            

            self.dF_F[i_cell,] = (self.F[i_cell,] - baseline) / baseline

            ### 1. find places where there are no spikes for a long interval 
            ### 1.1. we add all spikes in a 1s window by convolving it with a 1s box car function
            # fig, ax = plt.subplots(figsize=(12,8))
            # i_plot = 0  
            # cells = np.sort(np.random.randint(0, self.N_cells, 6))
            # cells[0] = 4
            # cells[5] = 1064

            allspikes_1s = np.hstack([np.repeat(sp_threshold, frame_rate-1), np.convolve(self.raw_spks[i_cell,:], filt, mode='valid')])

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
            if (len(fall_index) == 0):
                fall_index = np.array([int(0)])


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
            bases = np.zeros(len(rise_ind))
            for k in range(len(rise_ind)):
                i_start = fall_ind[k] + L_after_spike
                i_end = rise_ind[k] - L_before_spike
                sds[k] = np.sqrt(np.var(self.dF_F[i_cell,i_start:i_end]))
                bases[k] = np.average(self.dF_F[i_cell,i_start:i_end])
            self.cell_baselines[i_cell] = np.mean(bases)
            self.cell_SDs[i_cell] = np.mean(sds)
            self.cell_SNR[i_cell] = max(self.dF_F[i_cell,:]) / np.mean(sds)
            self.spks[i_cell,:] = self.spks[i_cell,:]# / baseline / self.cell_SDs[i_cell]

        print('SNR done')

        print('dF/F calculated for cell ROI-s')


    def calc_SD_SNR_elfiz(self):
        for i_cell in range(self.N_cells):
            self.cell_SDs[i_cell] = np.random.rand()
            self.cell_SNR[i_cell] = np.random.rand()

    def calc_active_elfiz(self):
        self.active_cells=np.array([0])

    def calc_active(self, events_per_ten_m = 10, sd_times = 3, refract_seconds = 5):
        #events_per_ten_m - we want at least this many events per 10 minutes to consider a cell active
        #sd_times - events should be above this many times the baseline sd
        #refract_seconds - refractoryness of event detection in seconds
        
        
        # creating the gaussian filter
        sdfilt = 3
        N = 10
        sampling_time = 1
        xfilt = np.arange(-N*sdfilt, N*sdfilt + sampling_time, sampling_time)
        filt = np.exp(-(xfilt**2) / (2*(sdfilt**2)))
        filt = filt/sum(filt)
        
        # calculating events
        #if not all laps are loaded we need to adjust the threshold accordingly!
        #if all laps are used:
        if self.selected_laps is None:
            len_imaging = self.frame_times[-1]-self.frame_times[0]
        #if not all laps are used:
        else:
            len_imaging = self.ImLaps[-1].raw_time[-1]-self.ImLaps[0].raw_time[0]
            
        active_threshold = len_imaging/(10*60)*events_per_ten_m
        if active_threshold < 1:
            print('active_cells may be unreliable due to the shortness of the analysed period')
        
        # print('active threshold: ', active_threshold)
        refractoriness = int(refract_seconds/self.frame_period) 
        
        n_events = np.zeros([self.N_cells])
        self.events=np.zeros(self.F.shape)
        
        for i in range(self.N_cells):
            temp = np.hstack([np.repeat(self.dF_F[i,0], N*sdfilt),self.dF_F[i,:], np.repeat(self.dF_F[i,-1], N*sdfilt)])
            dF_F_s = np.convolve(temp, filt, mode = 'valid')
            threshold=self.cell_baselines[i]+self.cell_SDs[i]*sd_times
            rises = np.nonzero((dF_F_s[0:-1] < threshold) & (dF_F_s[1:]>= threshold))[0]
            
            self.events[i,rises]=1 #here we do not take refractoriness into account
            valid = np.ones_like(rises)
            for j in range(rises.size-1):
                if valid[j]:
                    ind = np.nonzero((rises > rises[j]) & ((rises-rises[j]) < refractoriness))[0]
                    valid[ind] = 0
            n_events[i]=np.size(np.nonzero(valid==1)[0])
        #return
        self.active_cells = np.nonzero(n_events>active_threshold)[0]
        self.N_events = np.array(n_events)
        
    def detect_events(self,sd_times = 3):\
        # detecting significant events in the fluorescence signal
        # an event is significant, if the Gaussian filtered (SD = 3 x Interframe interval ) dF/F 
        # 
        # this function is redundant with the calc_active function in some parts. We need this in order to be able to pass events to individual laps
        sdfilt = 3
        N = 10
        sampling_time = 1
        xfilt = np.arange(-N*sdfilt, N*sdfilt + sampling_time, sampling_time)
        filt = np.exp(-(xfilt**2) / (2*(sdfilt**2)))
        filt = filt/sum(filt)
        self.events=np.zeros(self.F.shape)
        
        for i in range(self.N_cells):
            temp = np.hstack([np.repeat(self.dF_F[i,0], N*sdfilt),self.dF_F[i,:], np.repeat(self.dF_F[i,-1], N*sdfilt)])
            dF_F_s = np.convolve(temp, filt, mode = 'valid')
            threshold=self.cell_baselines[i]+self.cell_SDs[i]*sd_times
            rises = np.nonzero((dF_F_s[0:-1] < threshold) & (dF_F_s[1:]>= threshold))[0]
            
            self.events[i,rises]=1 #here we do not take refractoriness into account

    ##############################################################
    ## loading the LabView data
    ## separating data into laps
    ##############################################################

    def get_lapdata(self, datapath, date_time, name, task, selected_laps=None):

        time_array=[]
        lap_array=[]
        maze_array=[]
        position_array=[]
        mode_array=[]
        lick_array=[]
        action=[]
        substage=[]

        # this could be made much faster by using pandas
        data_log_file_string=datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_ExpStateMashineLog.txt'
        data_log_file=open(data_log_file_string, newline='')
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
        time_breaks = np.where(np.diff(laptime) > 1)[0]
        if (len(time_breaks) > 0):
            print('ExpStateMachineLog time interval > 1s: ', len(time_breaks), ' times')
            # print(laptime[time_breaks])

        lap = np.array(lap_array)
        logged_laps = np.unique(lap)
        all_laps = np.arange(max(lap)) + 1
        missing_laps = np.setdiff1d(all_laps, logged_laps)

        if (len(missing_laps) > 0):
            print('Some laps are not logged. Number of missing laps: ', len(missing_laps))
            print(missing_laps)
            self.n_laps = -1
            return 
        
        sstage = np.array(substage)
        current_sstage = sstage[0]

        pos = np.array(position_array)
        lick = np.array(lick_array)
        maze = np.array(maze_array)
        mode = np.array(mode_array)

        #################################################
        ## add position, and lap info into the imaging frames
        #################################################
        F = interp1d(laptime, pos) 
        self.frame_pos = np.round(F(self.frame_times), 2)
        F = interp1d(laptime, lap) 
        self.frame_laps = F(self.frame_times) 
        F = interp1d(laptime, maze) 
        self.frame_maze = F(self.frame_times) 

        print ('length of frame_times:', len(self.frame_times))
        print ('length of frame_laps:', len(self.frame_laps))
        print ('shape of dF_F:', np.shape(self.dF_F))

        self.all_corridor_start_time = []
        self.all_corridor_start_IDs = []
        ## frame_laps is NOT integer for frames between laps
        ## however, it MAY not be integer even even for frames within a lap...
        #################################################
        ## end of - add position, and lap info into the imaging frames
        #################################################
        i_ImData = [] # index of laps with imaging data
        i_corrids = [] # ID of corridor for the current lap

        self.n_laps = 0 # counting only the laps loaded for analysis
        lap_count = 0 # counting all laps except grey zone
        N_0lap = 0 # counting the non-valid laps

        grey_zone_active = False
        if (np.unique(maze)[0] == 0):
            grey_zone_active = True
            # print('grey zone is active')
            # grey_zone_duration = []
            # correct_error = []

        for i_lap in np.unique(lap): 
            y = np.flatnonzero(lap == i_lap) # index for the current lap

            mode_lap = np.prod(mode[y]) # 1 if all elements are recorded in 'Go' mode

            maze_lap = np.unique(maze[y])
            if (len(maze_lap) == 1):
                corridor = self.all_corridors[int(maze_lap)] # the maze_lap is the index of the available corridors in the given stage
            else:
                corridor = -1
            # print('corridor in lap ', self.n_laps, ':', corridor)
            
            sstage_lap = np.unique(sstage[y])
            
            if (len(sstage_lap) > 1):
                print('More than one substage in a lap before lap ', self.n_laps, 'in corridor', corridor)
                corridor = -2

            if (corridor > 0) :
                if (y.size < self.N_pos_bins):
                    print('Very short lap found, we have total ', len(y), 'datapoints recorded by the ExpStateMachine in a lap before lap', self.n_laps, 'in corridor', corridor)
                    corridor = -3

            if (corridor > 0) :
                pos_lap = pos[y]
                n_posbins = len(np.unique(pos_lap))
                if (n_posbins < (self.corridor_length_roxel * 0.9)):
                    print('Short lap found, we have total ', n_posbins, 'position bins recorded by the ExpStateMachine in a lap before lap', self.n_laps, 'in corridor', corridor)
                    corridor = -4

                # if (min(pos_lap) > 10):
                #     print('Late-start lap found, first position:', np.min(pos_lap), 'in lap', self.n_laps, 'in corridor', corridor)

                # if (max(pos_lap) < (self.corridor_length_roxel - 10)):
                #     print('Early-end lap found, last position:', np.max(pos_lap), 'in lap', self.n_laps, 'in corridor', corridor)
            
            # print('processing corridor', corridor, 'in lap', i_lap)

            t_lap = laptime[y]
            next_grey_lap_duration = None
            self.all_corridor_start_time.append(min(t_lap))          
            self.all_corridor_start_IDs.append(int(corridor))
            
            if (corridor > 0):    
                # if we select laps, then we check lap ID:
                if (selected_laps is None):
                    add_lap = True 
                else:
                    if (np.isin(lap_count, selected_laps)):
                        add_lap = True 
                    else:
                        add_lap = False

                if (add_lap):        
                    i_corrids.append(corridor) # list with the index of corridors in each run
                    pos_lap = pos[y]

                    lick_lap = lick[y] ## vector of Trues and Falses
                    t_licks = t_lap[lick_lap] # time of licks

                    if (sstage_lap != current_sstage):
                        print('############################################################')
                        print('substage change detected!')
                        print('first lap in substage ', sstage_lap, 'is lap', self.n_laps, ', which started at t', t_lap[0])
                        print('the time of the change in imaging time is: ', t_lap[0] - self.imstart_time)
                        print('############################################################')
                        current_sstage = sstage_lap
                        self.substage_change_laps.append(self.n_laps)
                        self.substage_change_time.append(t_lap[0] - self.imstart_time)


                    istart = np.min(y) # action is not a np.array, array indexing does not work
                    iend = np.max(y) + 1
                    action_lap = action[istart:iend]
        
                    reward_indices = [j for j, x in enumerate(action_lap) if x == "TrialReward"]
                    t_reward = t_lap[reward_indices]

                    ## detecting invalid laps - terminated before the animal could receive reward
                    valid_lap = False
                    if (len(t_reward) > 0): # lap is valid if the animal got reward
                        valid_lap = True
                        # print('rewarded lap', self.n_laps)
                    if (max(pos_lap) > (self.corridor_length_roxel * self.last_zone_end)): # # lap is valid if the animal left the last reward zone
                        valid_lap = True
                        # print('valid lap', self.n_laps)
                    if (valid_lap == False):
                        mode_lap = 0
                        # print('invalid lap', self.n_laps)

                    if (grey_zone_active): # we calculate the duration of the next grey zone - we can use this to double check rewarded laps
                        y_g = np.flatnonzero(lap == i_lap+1) # index for the current lap
                        if (len(y_g) > 1):
                            maze_lap_g = np.unique(maze[y_g])
                            if (len(maze_lap_g) == 1):
                                corridor_g = self.all_corridors[int(maze_lap_g)] # the maze_lap is the index of the available corridors in the given stage
                            if (corridor_g != 0):
                                print('no next grey zone found')
                            else:
                                t_g_lap = laptime[y_g]
                                next_grey_lap_duration = np.max(t_g_lap) - np.min(t_g_lap)

                    actions = []
                    for j in range(len(action_lap)):
                        if not((action_lap[j]) in ['No', 'TrialReward']):
                            actions.append([t_lap[j], action_lap[j]])

                    ## include only valid laps
                    add_ImLap = True
                    if (mode_lap == 0):
                        add_ImLap = False
                        print('lap mode = 0')

                    ### imaging data    
                    iframes = np.flatnonzero(self.frame_laps == i_lap)
                    if (len(iframes) > 1): # there is imaging data belonging to this lap...
                        lap_frames_dF_F = self.dF_F[:,iframes]
                        lap_frames_spikes = self.spks[:,iframes]
                        lap_frames_time = self.frame_times[iframes]
                        lap_frames_pos = self.frame_pos[iframes]
                        lap_frames_events = self.events[:,iframes]
                        # print(self.n_laps, np.min(lap_frames_pos), np.max(lap_frames_pos))
                        if (np.min(lap_frames_pos) > 200):
                            add_ImLap = False
                            print('Late-start lap found, first position:', np.min(lap_frames_pos), 'in lap', self.n_laps, 'in corridor', corridor)
                        if (np.max(lap_frames_pos) < (self.corridor_length_roxel - 200)):
                            add_ImLap = False
                            print('Early end lap found, last position:', np.max(lap_frames_pos), 'in lap', self.n_laps, 'in corridor', corridor)
                    else:
                        add_ImLap = False

                    if (add_ImLap): # there is imaging data belonging to this lap...
                        i_ImData.append(self.n_laps)
                        # print('frames:', min(iframes), max(iframes))
                        # print('max of iframes:', max(iframes))
                    else :
                        lap_frames_dF_F = np.nan
                        lap_frames_spikes = np.nan
                        lap_frames_time = np.nan
                        lap_frames_pos = np.nan 
                        lap_frames_events = np.nan                        
                    # print('In lap ', self.n_laps, ' we have ', len(t_lap), 'datapoints and ', len(iframes), 'frames')

                    self.ImLaps.append(Lap_ImData(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time, self.corridor_list, lap_frames_events, self.frame_period, speed_threshold=self.speed_threshold, elfiz=self.elfiz, multiplane=self.multiplane, next_grey_lap_duration=next_grey_lap_duration))
                    self.n_laps = self.n_laps + 1
                    lap_count = lap_count + 1
                else :
                    # print('lap ', lap_count, ' skipped.')
                    lap_count = lap_count + 1
                    # print(self.n_laps)
            else:
                # if (corridor == 0):
                #     lap_duration = np.max(t_lap) - np.min(t_lap)
                #     previous_correct = 'first lap'
                #     if ((lap_count > 0) & (corridor == 0)):
                #         previous_correct = self.ImLaps[self.n_laps-1].correct
                #         print('duration of grey zone after lap', self.n_laps-1, ': ', lap_duration, previous_correct)
                #         correct_error.append(previous_correct)
                #         grey_zone_duration.append(lap_duration)
                N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1) - we do not do anything with this...


        # correct_error = np.array(correct_error)
        # i_correct = np.flatnonzero(correct_error == 1)
        # i_error = np.flatnonzero(correct_error == 0)
        # grey_zone_duration = np.array(grey_zone_duration)
        # print('minimum grey zone after error:', np.min(grey_zone_duration[i_error]), ', max grey zone after correct:', np.max(grey_zone_duration[i_correct]))

        self.i_Laps_ImData = np.array(i_ImData) # index of laps with imaging data
        self.i_corridors = np.array(i_corrids) # ID of corridor for the current lap

    def combine_lapdata(self): ## fills in the cell_activity tensor
        # self.raw_activity_tensor = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials containing the spikes
        # self.raw_activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # a tensor with space x  trials containing the time spent at each location in each lap
        valid_lap = np.zeros(len(self.i_Laps_ImData))
        k_lap = 0
        for i_lap in self.i_Laps_ImData:
            if (self.ImLaps[i_lap].n_cells > 0):  # we only add imaging data when the lap is valid, so we don't need to test it again
                valid_lap[k_lap] = 1
                self.raw_activity_tensor[:,:,k_lap] = np.transpose(self.ImLaps[i_lap].spks_pos)
                self.raw_activity_tensor_time[:,k_lap] = self.ImLaps[i_lap].T_pos_fast
            k_lap = k_lap + 1

        ## smoothing - average of the 3 neighbouring bins
        # self.activity_tensor = self.raw_activity_tensor
        # self.activity_tensor_time = self.raw_activity_tensor_time
        self.activity_tensor[0,:,:] = (self.raw_activity_tensor[0,:,:] + self.raw_activity_tensor[1,:,:]) / 2
        self.activity_tensor[-1,:,:] = (self.raw_activity_tensor[-2,:,:] + self.raw_activity_tensor[-1,:,:]) / 2
        self.activity_tensor_time[0,:] = (self.raw_activity_tensor_time[0,:] + self.raw_activity_tensor_time[1,:]) / 2
        self.activity_tensor_time[-1,:] = (self.raw_activity_tensor_time[-2,:] + self.raw_activity_tensor_time[-1,:]) / 2
        for i_bin in np.arange(1, self.N_pos_bins-1):
            self.activity_tensor[i_bin,:,:] = np.average(self.raw_activity_tensor[(i_bin-1):(i_bin+2),:,:], axis=0)
            self.activity_tensor_time[i_bin,:] = np.average(self.raw_activity_tensor_time[(i_bin-1):(i_bin+2),:], axis=0)

        i_valid_laps = np.nonzero(valid_lap)[0]
        self.i_Laps_ImData = self.i_Laps_ImData[i_valid_laps]
        self.raw_activity_tensor = self.raw_activity_tensor[:,:,i_valid_laps]
        self.raw_activity_tensor_time = self.raw_activity_tensor_time[:,i_valid_laps]
        self.activity_tensor = self.activity_tensor[:,:,i_valid_laps]
        self.activity_tensor_time = self.activity_tensor_time[:,i_valid_laps]

    def speed_vs_activity(self):
#        print('IMAGED LAPS',len(self.ImLaps))
        plt.figure('events')
        plt.xlabel('speed')
        plt.ylabel('events')
        plt.title('speed vs. events')
        plt.figure('active cells')
        plt.xlabel('speed')
        plt.ylabel('active cells')
        plt.title('speed vs. N active')
        for i in range(len(self.ImLaps)):
            if self.ImLaps[i].imaging_data==True:
#                print(self.ImLaps[i].frames_dF_F.shape)
                av_speed = np.average(self.ImLaps[i].frames_speed)
                sum_events = np.sum(self.ImLaps[i].frames_events)
                n_active_cells = np.nonzero(np.sum(self.ImLaps[i].frames_events, axis=1))[0].size
                plt.figure('events')
                plt.scatter(av_speed, sum_events)
                plt.figure('active cells')
                plt.scatter(av_speed, n_active_cells)
        plt.show()
        
    def calculate_properties(self, nSD=4):
        self.cell_reliability = []
        self.cell_Fano_factor = []
        self.cell_skaggs=[]
        self.cell_activelaps=[]
        self.cell_activelaps_df=[]
        self.cell_tuning_specificity=[]

        self.cell_rates = [] # a list, each element is a 1 x n_cells matrix with the average rate of the cells in the whole corridor
        if (self.task == 'contingency_learning'):
            self.cell_pattern_rates = []
        self.ratemaps = [] # a list, each element is an array space x neurons x trials being the ratemaps of the cells in a given corridor

        self.cell_corridor_selectivity = np.zeros([2,self.N_cells]) # a matrix with the selectivity index of the cells. Second row indicates the corridor with the highers rate.
        self.cell_corridor_similarity = np.zeros(self.N_cells) # a matrix with the similarity index of the cells.

        if (self.N_corridors > 0):
            for i_corridor in np.arange(self.N_corridors): # we exclude corridor 0
                corridor = self.corridors[i_corridor]

                # select the laps in the corridor 
                # only laps with imaging data are selected - this will index the activity_tensor
                i_laps = np.flatnonzero(self.i_corridors[self.i_Laps_ImData] == corridor)
                N_laps_corr = len(i_laps)

                time_matrix_1 = self.activity_tensor_time[:,i_laps]
                total_time = np.sum(time_matrix_1, axis=1) # bins x cells -> bins; time spent in each location

                act_tensor_1 = self.activity_tensor[:,:,i_laps] ## bin x cells x laps; all activity in all laps in corridor i
                total_spikes = np.sum(act_tensor_1, axis=2) ##  bin x cells; total activity of the selected cells in corridor i

                rate_matrix = np.zeros_like(total_spikes) ## event rate 
                
                for i_cell in range(self.N_cells):
                # for i_cell in range(total_spikes.shape[1]):
                    rate_matrix[:,i_cell] = total_spikes[:,i_cell] / total_time
                self.ratemaps.append(rate_matrix)
                
                print('calculating rate, reliability and Fano factor...')
                ## average firing rate
                rates = np.sum(total_spikes, axis=0) / np.sum(total_time)
                self.cell_rates.append(rates)

                if (self.task == 'contingency_learning'):
                    zone_start = int(np.floor(self.ImLaps[0].zones[0]*self.N_pos_bins)) # 42-46 or 45-49
                    zone_end = int(np.floor(self.ImLaps[0].zones[1]*self.N_pos_bins))

                    rates_pattern1 = np.sum(total_spikes[0:14,:], axis=0) / np.sum(total_time[0:14])
                    rates_pattern2 = np.sum(total_spikes[14:28,:], axis=0) / np.sum(total_time[14:28])
                    rates_pattern3 = np.sum(total_spikes[28:42,:], axis=0) / np.sum(total_time[28:42])
                    rates_reward = np.sum(total_spikes[zone_start:zone_end,:], axis=0) / np.sum(total_time[zone_start:zone_end])
                    self.cell_pattern_rates.append(np.vstack([rates_pattern1, rates_pattern2, rates_pattern3, rates_reward]))

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
                ## Skaggs spatial info in bits per spike
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

                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps_abs = self.i_Laps_ImData[np.nonzero(icorrids == corridor)[0]] # we need a different indexing here, for the ImLaps list and not fot  the activityTensor
                k = 0
                if (self.elfiz):
                    spike_threshold = 0.75
                else:
                    spike_threshold = 25

                for i_lap in i_laps_abs:#y=ROI
                    act_cells = np.nonzero(np.amax(self.ImLaps[i_lap].frames_spikes, 1) > spike_threshold)[0]
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

                ## linear tuning specificity
                print('calculating linear tuning specificity ...')
                tuning_spec = np.zeros(self.N_cells)
                xbins = (np.arange(self.N_pos_bins) + 0.5) * self.corridor_length_cm / self.N_pos_bins
                
                for i_cell in range(self.N_cells):
                    rr = np.copy(rate_matrix[:,i_cell])
                    rr[rr < np.mean(rr)] = 0
                    Px = rr / np.sum(rr)
                    mu = np.sum(Px * xbins)
                    sigma = np.sqrt(np.sum(Px * xbins**2) - mu**2)
                    tuning_spec[i_cell] = self.corridor_length_cm / sigma

                self.cell_tuning_specificity.append(tuning_spec)

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
        ##           is near 0 for cells with uncorrelated ratemaps
        ##           is 1 if cells have identical ratemaps for all the corridors
        ## the similarity is stored in a vector of 1 element for each cell

        print('calculating corridor selectivity ...')
        rate_matrix = np.array(self.cell_rates)

        max_rate = np.max(rate_matrix, axis=0)
        i_corr_max = np.argmax(rate_matrix, axis=0)
        min_rate = np.min(rate_matrix, axis=0)
        sumrate = np.sum(rate_matrix, axis=0)

        self.cell_corridor_selectivity[0,:] = (max_rate - min_rate) / sumrate
        self.cell_corridor_selectivity[1,:] = i_corr_max
        
        # in Rita's task, we also calculate corridor selectivity in the pattern and reward zones:
        if (self.task == 'contingency_learning'):
            rate_matrix = np.array(self.cell_pattern_rates)
            max_rate = np.max(rate_matrix, axis=0)
            i_corr_max = np.argmax(rate_matrix, axis=0)
            min_rate = np.min(rate_matrix, axis=0)
            sumrate = np.sum(rate_matrix, axis=0)
            self.cell_pattern_selectivity = np.zeros(2*4*self.N_cells).reshape(2, 4, self.N_cells)
            self.cell_pattern_selectivity[0,:,:] = (max_rate - min_rate) / sumrate
            self.cell_pattern_selectivity[1,:,:] = i_corr_max

        print('calculating corridor similarity ...')
        map_mat = np.array(self.ratemaps)
        N_corridors_w_ratemaps = map_mat.shape[0]
        M = int(N_corridors_w_ratemaps * (N_corridors_w_ratemaps - 1) / 2)
        similarity_matrix = np.zeros((M, self.N_cells))
        m = 0
        for i_cor in np.arange(N_corridors_w_ratemaps-1):
            for j_cor in np.arange(i_cor+1, N_corridors_w_ratemaps):
                X = np.transpose(map_mat[i_cor,:,:])
                Y = np.transpose(map_mat[j_cor,:,:])
                similarity_matrix[m,] = Mcorrcoef(X,Y, zero_var_out=0)

        self.cell_corridor_similarity = np.mean(similarity_matrix, axis=0)

    def plot_properties(self, cellids=np.array([-1]), interactive=False):
        
        fig, ax = plt.subplots(self.N_corridors, 4, figsize=(10,5), sharex='col', sharey='col')
        plt.subplots_adjust(wspace=0.35, hspace=0.2)
        # matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        
        sc=[]
        for i in range(self.N_corridors*4):
            sc.append(0)

        ax[0, 0].axis('off')

        sc[4] = ax[1,0].scatter(self.cell_SDs, self.cell_SNR, alpha=0.5, color='w', edgecolors='C3')
        ax_list = fig.axes
        if (max(cellids) > 0):
            ax[1,0].scatter(self.cell_SDs[cellids], self.cell_SNR[cellids], alpha=0.75, color='C3')
        # sc = ax_left.scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
        title_string = 'SNR vs. SD'
        ax[1,0].set_title(title_string)
        ax[1,0].set_ylabel('SNR')
        ax[1,0].set_xlabel('SD')
        
        for i_corridor in range(self.N_corridors):
            corridor=self.corridors[i_corridor]#always plot the specified corridor
            
            rates = self.cell_rates[i_corridor]
            reliability = self.cell_reliability[i_corridor]
            skaggs_info=self.cell_skaggs[i_corridor]
            Fano_factor = self.cell_Fano_factor[i_corridor]
            specificity = self.cell_tuning_specificity[i_corridor]   
            act_laps = self.cell_activelaps[i_corridor]
#            act_laps_dF = self.cell_activelaps_df[i_corridor]
    
            sc[i_corridor*4+1] = ax[i_corridor,1].scatter(rates, reliability, alpha=0.5, s=skaggs_info*50, color='w', edgecolors='C0')
            if (max(cellids) > 0):
                ax[i_corridor,1].scatter(rates[cellids], reliability[cellids], alpha=0.75, s=skaggs_info[cellids]*50, color='C0')
            # sc = ax_left.scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
            title_string = 'corr.' + str(corridor)
            ax[i_corridor,1].set_title(title_string)
            ax[i_corridor,1].set_ylabel('reliability')
            if (i_corridor == self.N_corridors - 1): ax[i_corridor,1].set_xlabel('average event rate')
    
    
            sc[i_corridor*4+2] = ax[i_corridor,2].scatter(act_laps, specificity, alpha=0.5, s=skaggs_info*50, color='w', edgecolors='C1')
            if (max(cellids) > 0):
                ax[i_corridor,2].scatter(act_laps[cellids], specificity[cellids], alpha=0.75, s=skaggs_info[cellids]*50, color='C1')
    
            title_string = 'corr.' + str(corridor)
            ax[i_corridor,2].set_title(title_string)
            ax[i_corridor,2].set_ylabel('tuning specificity')
            if (i_corridor == self.N_corridors - 1): ax[i_corridor,2].set_xlabel('percent active laps spikes')
    
            sc[i_corridor*4+3] = ax[i_corridor,3].scatter(skaggs_info, Fano_factor, alpha=0.5, s=skaggs_info*50, color='w', edgecolors='C2')
            if (max(cellids) > 0):
                ax[i_corridor,3].scatter(skaggs_info[cellids], Fano_factor[cellids], alpha=0.75, s=skaggs_info[cellids]*50, color='C2')
            # sc = ax_right.scatter(rates, reliability, alpha=0.5, s=skaggs_info*50)
            title_string = 'corr.' + str(corridor)
            ax[i_corridor,3].set_title(title_string)
            ax[i_corridor,3].set_ylabel('Fano factor')
            if (i_corridor == self.N_corridors - 1): ax[i_corridor,3].set_xlabel('Skaggs info (bit/event)')
           
        #####################
        #add interactive annotation
        #####################
        #we need an sc list, with the given scatters in order, empty slots can be 0
        if (interactive == True):
            annot=[]
            for i in range(self.N_corridors*4):
                if sc[i] != 0:
                    annot_sub = ax_list[i].annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w"),
                                 arrowprops=dict(arrowstyle="->"))
                    annot_sub.set_visible(False)
                    annot.append(annot_sub)
                else:
                    annot.append(0)

            def update_annot(ind, active_ax):          
                pos = sc[active_ax].get_offsets()[ind["ind"][0]]
                annot[active_ax].xy = pos
                index = "{}".format(" ".join(list(map(str,ind["ind"]))))
                annot[active_ax].set_text(index)
     #            annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
                annot[active_ax].get_bbox_patch().set_alpha(0.4)
            
            def hover(event):
                for i in range(len(ax_list)):
                    if ax_list[i]== event.inaxes:
                        if sc[i] != 0:
                            cont, ind = sc[i].contains(event)
                            # print(ind)
                            if cont:
                                update_annot(ind, i)
                                annot[i].set_visible(True)
                                fig.canvas.draw_idle()
                            else:
                                annot[i].set_visible(False)
                                fig.canvas.draw_idle()
                                for i_annot in range(len(annot)):
                                    if i_annot!=i and annot[i_annot!=0]:
                                        annot[i_annot].set_visible(False)
                                        fig.canvas.draw_idle()
            
            fig.canvas.mpl_connect("motion_notify_event", hover)
        
            plt.show(block=False)
        else :
            plt.show(block=False)
            
    def plot_hist_save_data(self, prop, cellids = np.nan, 
                            N_bins = 30, bins_start = np.nan, bins_end = np.nan,normalised = False,logx_scale=False,logy_scale=False, 
                            labels=[], title = '_', saveplot_name = np.nan, save_data = False, plot = True):
        # function to make histograms of and/or save to excel any property specified, or the slice of these properties using the cellids parameter for indexing
        
        #helper function for plotting
        def draw(array, N_bins, bins_start, bins_end, labels, title): 
            if np.isnan(bins_start):
                bins_start = np.min(array)
            if np.isnan(bins_end):
                bins_end = np.max(array)        
            counts, bins = np.histogram(array, bins=np.linspace(bins_start,bins_end,N_bins))
            if normalised:
                counts = counts/len(array)
            plt.stairs(counts, bins, label = label)
            if logy_scale:
                plt.yscale('log')
            if logx_scale:
                plt.xscale('log')
        
        if save_data:
            to_save = [] #we will prepare the data to be saved in an this array
        
        cellids = np.array(cellids)
        
        #plotting
        if plot:
            plt.figure()
        if hasattr(prop, '__iter__'):
            if hasattr(prop[0], '__iter__'):
                # 2dim input
                for i_prop in range(len(prop)):
                    #celect cellids if specified
                    if np.any(np.isnan(cellids)):
                        array = prop[i_prop]
                    else:
                        array = prop[i_prop][cellids]
                        
                    #prepare data to save if specified
                    if save_data:
                        to_save.append(array)
                    
                    #prepare labels for plotting
                    if len(labels)==0:
                        label = str(i_prop)
                    else:
                        try:
                            label = labels[i_prop]
                        except IndexError:
                            print('Warning! not enough labels specified...')
                            label = str(i_prop)
                    #draw histogram using the helper function
                    if plot:
                        draw(array, N_bins, bins_start, bins_end, label, title)
            else:
                # 1dim input
                #celect cellids if specified
                if np.any(np.isnan(cellids)):
                    array = prop
                else:
                    array = prop[cellids]
                    
                #prepare data to save if specified
                if save_data:
                    to_save.append(array)
                    
                #prepare labels for plotting
                if len(labels)==0:
                    label = ' '
                else:
                    label = labels[0]
                    
                #draw histogram using the helper function
                if plot:
                    draw(array, N_bins, bins_start, bins_end, labels, title)
        else:
            print("Input not iterable - returning")
            return
        
        #plot annotation
        if plot:
            plt.title(title)
            if normalised:
                if logy_scale:
                    plt.ylabel('log fraction')
                else:
                    plt.ylabel('fraction')
            else:
                if logy_scale:
                    plt.ylabel('log N')
                else:
                    plt.ylabel('N')
            if logx_scale:
                plt.xlabel('log property value')
            else:
                plt.xlabel('property value')
            plt.legend()
            
            #save if filename is given
            if type(saveplot_name) == str:
                plt.savefig(self.suite2p_folder + saveplot_name)
        
        # save data to excel:
        if save_data:
            writer = pd.ExcelWriter(self.suite2p_folder + title + '.xlsx', engine='openpyxl') 
            wb  = writer.book
            startcol=0
            for i in range(len(to_save)):
                try:
                    header = labels[i]
                except:
                    header = 'property ' + str(i)
                df = pd.DataFrame(to_save[i], columns=[header])
                df.to_excel(writer, index=False, startcol=startcol)
                startcol+=1
            wb.save(self.suite2p_folder + title + '.xlsx')
    

    def Hainmuller_PCs(self):
        ## ratemaps: similar to the activity tensor, the laps are sorted by the corridors
        self.candidate_PCs = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping

        ## we calculate the rate matrix for all corridors - we need to use the same colors for the images
        for i_corrid in np.arange(self.N_corridors):
            corrid = self.corridors[i_corrid]

            rate_matrix = self.ratemaps[i_corrid]
            # only laps with imaging data are selected - this will index the activity_tensor

            candidate_cells = np.zeros(self.N_cells)
            
            i_laps = np.flatnonzero(self.i_corridors[self.i_Laps_ImData] == corrid)
            N_laps_corr = len(i_laps)
            act_tensor_1 = self.activity_tensor[:,:,i_laps] ## bin x cells x laps; all activity in all laps in corridor i

            for i_cell in np.arange(rate_matrix.shape[1]):
                rate_i = rate_matrix[:,i_cell]

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
                    lapsums = act_tensor_1[index_infield,i_cell,:].sum(0) # only laps in this corridor

                    if ( ( (sum(lapsums > 0.6) / float(N_laps_corr)) > 0.2)  & ( (rate_inField / rate_outField) > 7) ):
                        candidate_cells[i_cell] = 1

            self.candidate_PCs.append(candidate_cells)

    # def __init__(self, datapath, date_time, name, task, stage, raw_spikes, frame_times, frame_pos, frame_laps, N_shuffle=1000, mode='random'):
    def calc_shuffle(self, cellids, n=1000, mode='shift', batchsize=25, verbous=1, name_string=''):
        ## cellids: numpy array - the index of the cells to be included in the analysis
        ## n: integer, number of shuffles
        ## mode: 'random' or 'shift'. 
            # random: totally randomize the spike times; 
            # shift: circularly shift spike times 
        ## batchsize: integer. To make computation faster, shuffling is done in batches of size batchsize
        ## verbous: True of False: information is given about the progress
        ## name string: optional string saved in the file to specify the peculiarities of the analysis

        cellids = np.array(cellids)
        raw_spikes = self.raw_spks[cellids,:]
        calculate_shuffles = True

        ##########################################################################
        ## reading shuffling data from file
        ##########################################################################

        data_folder = self.suite2p_folder + self.data_folder
        shuffle_filename = 'shuffle_stats_' + name_string + 'n' + str(n) + '_mode_' + mode + '.csv'
        shuffle_path = data_folder + '/' + shuffle_filename
        if os.path.exists(shuffle_path):
            calculate_shuffles = False
            if (self.check_params(shuffle_filename)):
                if (verbous > 1):
                    print('loading shuffling P-values from file...')
                ## load from file
                shuffle_file=open(shuffle_path, newline='')
                shuffle_file_reader=csv.reader(shuffle_file, delimiter=' ')
                ## we skip the first few lines, as they contain the header already checked by check_params(shuffle_filename)
                file_data = next(shuffle_file_reader)
                while (file_data[0][0] == '#'):
                    file_data = next(shuffle_file_reader)
                # the statistics and corridors
                Ps_names = file_data
                if (self.N_corridors == 1):
                    expected_names = self.N_corridors * 4 + 1
                if (self.N_corridors > 1):
                    if (self.task == 'contingency_learning'):
                        expected_names = self.N_corridors * 4 + 3 + 4
                    else:
                        expected_names = self.N_corridors * 4 + 3

                if (len(Ps_names) == expected_names):
                    file_data = next(shuffle_file_reader)
                    shuffle_Pvalues = np.array(file_data, dtype='float').reshape(1, expected_names)
                    for row in shuffle_file_reader:
                        shuffle_Pvalues = np.vstack((shuffle_Pvalues, np.array(row, dtype='float')))
                    ## check the cellids
                    cellids_from_file = shuffle_Pvalues[:,0]
                    if not np.array_equal(np.sort(cellids_from_file), np.sort(cellids)):
                        if (verbous > 0):
                            print ('cellids of the saved file does not match the cellids provided. We will perform shuffling.')
                        calculate_shuffles = True
                    else :
                        if (verbous > 0):
                            print ('P-values successfully read from the saved file.')

                else :# fill in the P-value array
                    if (verbous > 0):
                        print ('number of P-values read from the saved file for each cell does not match the number expected for a given number of corridor. We will perform shuffling.')
                    calculate_shuffles = True
            if os.path.exists(data_folder + '/p95.npy'):
                self.p95 = np.load(data_folder + '/p95.npy', allow_pickle=True)
            else:
                calculate_shuffles = True

        ##########################################################################
        ## calculating shuffling - if appropriate file not found
        ##########################################################################
        if (calculate_shuffles):
            if (verbous > 0):
                print('calculating shuffles...')
            shuffle_stats = ImShuffle(self.datapath, self.date_time, self.name, self.task, self.stage, raw_spikes, self.frame_times, self.frame_pos, self.frame_laps, N_shuffle=n, cellids=cellids, mode=mode, batchsize=batchsize, randseed=self.randseed, selected_laps=self.selected_laps, elfiz=self.elfiz, min_Nlaps=self.minimum_Nlaps, multiplane=self.multiplane)
            self.p95 = shuffle_stats.p95
            np.save(data_folder + '/p95.npy', self.p95, allow_pickle=True)

            # shuffle_stats = ImShuffle(D1.datapath,   D1.date_time,   D1.name,   D1.task,   D1.stage,   raw_spikes, D1.frame_times,   D1.frame_pos,   D1.frame_laps,   N_shuffle=N_shuffle, cellids=cellids, mode='shift', batchsize=25,        randseed=D1.randseed, selected_laps=np.arange(20,80), elfiz=True)

            # NN = cellids.size
            N_corrids = len(shuffle_stats.ratemaps)
            sanity_checks_passed = True
            if ((N_corrids) != self.N_corridors):
                    print ('warning: number of corridors is different between shuffling and control!')
                    sanity_checks_passed = False            
            for i_cor in np.arange(self.N_corridors):
                if (self.elfiz == True): # we use a different time resolution for shuffling...
                    if (np.abs(shuffle_stats.cell_reliability[i_cor][0,n] - self.cell_reliability[i_cor][0]) > 0.1):
                        print ('warning: calculating reliability is different between shuffling and control!')
                        sanity_checks_passed = False
                    if (np.corrcoef(self.ratemaps[i_cor][:,0], shuffle_stats.ratemaps[i_cor][:,0,n])[0,1] < 0.75):
                        print ('warning: ratemaps are different between shuffling and control!')
                        sanity_checks_passed = False
                else:
                    if (sum(np.abs(shuffle_stats.cell_reliability[i_cor][:,n] - self.cell_reliability[i_cor][cellids]) > 0.0001) > 0):
                        print ('warning: calculating reliability is different between shuffling and control!')
                        sanity_checks_passed = False
                    if (np.sum(np.abs(self.ratemaps[i_cor][:,cellids] - shuffle_stats.ratemaps[i_cor][:,:,n]) > 1e-5)):
                        print ('warning: ratemaps are different between shuffling and control!')
                        sanity_checks_passed = False

            if (sanity_checks_passed):
                if (verbous > 0):
                    print ('Shuffling stats calculated succesfully')
            else : 
                if (verbous > 0):
                    print ('Shuffling failed, ask for help...')
                return 

            if (verbous > 1):
                print('saving shuffling data into file...')

            # shuffle_Pvalues:
            # cellids + N x Skaggs + N x specificits + N x reliability + (selectivity + similarity + (4 x pattern_selectivity)) + N x Hainmuller

            if (shuffle_stats.N_corridors > 1):# & (task == 'contingency_learning')):
                shuffle_Pvalues = cellids
                Ps_names = ['cellids']#, 
                for i in np.arange(shuffle_stats.N_corridors):
                    shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.P_skaggs[i]))
                    Ps_names = Ps_names + ['Skaggs_' + str(i)]
                for i in np.arange(shuffle_stats.N_corridors):
                    shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.P_tuning_specificity[i]))
                    Ps_names = Ps_names + ['spec_' + str(i)]
                for i in np.arange(shuffle_stats.N_corridors):
                    shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.P_reliability[i]))
                    Ps_names = Ps_names + ['reli_' + str(i)]

                shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.P_corridor_selectivity))
                Ps_names = Ps_names + ['selectivity']

                shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.P_corridor_similarity))
                Ps_names = Ps_names + ['similarity']

                if (self.task == 'contingency_learning'):
                    for kk in np.arange(4):
                        shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.P_pattern_selectivity[kk,:]))
                        Ps_names = Ps_names + ['pattern_selectivity_' + str(kk)]                

                for i in np.arange(shuffle_stats.N_corridors):
                    shuffle_Pvalues = np.vstack((shuffle_Pvalues, shuffle_stats.accepted_PCs[i]))
                    Ps_names = Ps_names + ['Hainmuller_PlaceCell_' + str(i)]
            else :
                shuffle_Pvalues = np.vstack((cellids, shuffle_stats.P_skaggs[0], shuffle_stats.P_tuning_specificity[0], shuffle_stats.P_reliability[0], shuffle_stats.accepted_PCs[0]))
                Ps_names = ['cellids', 'Skaggs_0', 'spec_0', 'reli_0', 'Hainmuller_PlaceCell_0']# 

            shuffle_Pvalues = np.transpose(shuffle_Pvalues)

            self.write_params(shuffle_filename)
            with open(shuffle_path, mode='a', newline='') as shuffle_file:
                file_writer = csv.writer(shuffle_file, **csv_kwargs)
                file_writer.writerow(Ps_names)
                for i_row in np.arange(shuffle_Pvalues.shape[0]):
                    file_writer.writerow(np.round(shuffle_Pvalues[i_row,:], 6))
            print('P values for n=', len(cellids), ' saved into file: ', shuffle_path)

        ##########################################################################
        ## processing the shuffling stats ...
        ##########################################################################

        # shuffle_Pvalues:
        # if N is the number of corridors
        # cellids + N x Skaggs + N x specificits + N x reliability + (selectivity + similarity + (4 x pattern_selectivity)) + N x Hainmuller

        self.shuffle_Pvalues = shuffle_Pvalues
        self.Ps_names = Ps_names

        # Pmatrix:
        # N x Skaggs + N x specificits + N x reliability + (selectivity + similarity + (4 x pattern_selectivity))

        if (self.N_corridors == 1): # we don't have selectivity and similarity
            Pmatrix = np.transpose(self.shuffle_Pvalues[:,1:(self.N_corridors*3+1)])
            self.ii_tuned_cells = np.transpose(HolmBonfMat(Pmatrix, 0.05))
        if (self.N_corridors > 1):
            if (self.task == 'contingency_learning'):# we have 4 + 1 selectivity and similarity
                max_col_index = self.N_corridors*3+3+4 
            else :# we have selectivity and similarity
                max_col_index = self.N_corridors*3+3                
            Pmatrix = np.transpose(self.shuffle_Pvalues[:,1:max_col_index])
            self.ii_tuned_cells = np.transpose(HolmBonfMat(Pmatrix, 0.05))

        self.accepted_PCs = []
        self.tuned_cells = []
        self.skaggs_tuned_cells = []
        self.spec_tuned_cells = []
        self.reli_tuned_cells = []
        if (self.task == 'contingency_learning'):
            self.pattern_selective_cells = []

        # print(shuffle_Pvalues)
        # print(shuffle_Pvalues.shape)
        for i_cor in np.arange(self.N_corridors):
            # print(i_cor)
            if (self.N_corridors == 1):
                self.accepted_PCs.append(cellids[np.where(shuffle_Pvalues[:,self.N_corridors*3+1+i_cor])])
            if (self.N_corridors > 1):
                if (self.task == 'contingency_learning'):
                    Hainmuller_index = self.N_corridors*3+3+4+i_cor
                else :
                    Hainmuller_index = self.N_corridors*3+3+i_cor
                self.accepted_PCs.append(cellids[np.where(shuffle_Pvalues[:,Hainmuller_index])])
            self.skaggs_tuned_cells.append(cellids[np.where(self.ii_tuned_cells[:,i_cor])])
            self.spec_tuned_cells.append(cellids[np.where(self.ii_tuned_cells[:,self.N_corridors+i_cor])])
            self.reli_tuned_cells.append(cellids[np.where(self.ii_tuned_cells[:,self.N_corridors*2+i_cor])])
            self.tuned_cells.append(np.unique(np.concatenate((self.skaggs_tuned_cells[i_cor], self.spec_tuned_cells[i_cor], self.reli_tuned_cells[i_cor]))))
        if (verbous > 1):
            print('tuned cells:', self.tuned_cells)
        if (self.N_corridors > 1):
            self.selective_cells = cellids[np.where(self.ii_tuned_cells[:, self.N_corridors*3])]
            self.similar_cells = cellids[np.where(self.ii_tuned_cells[:, self.N_corridors*3+1])]
            if (verbous > 1):
                print('selective cells:', self.selective_cells)
                print('similar cells:', self.similar_cells)
            if (self.task=='contingency_learning'):
                for kk in np.arange(4):
                    self.pattern_selective_cells.append(cellids[np.where(self.ii_tuned_cells[:, self.N_corridors*3+2+kk])])
                if (verbous > 1):
                    print('pattern selective cells:', self.pattern_selective_cells)


    def get_lap_indexes(self, corridor=-1, i_lap=-1):
        ## print the indexes of i_lap (or each lap) in a given corridor
        ## if corridor == -1 then the first corridor is used
        if (corridor == -1):
            corridor = np.unique(self.i_corridors)[0]
        # select the laps in the corridor 
        # only laps with imaging data are selected - this will index the activity_tensor
        i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor)[0] 
        N_laps_corr = len(i_laps)
        print('lap # in corridor ' + str(corridor) + ' with imaging data;    lap # within session')
        if (i_lap == -1):
            for i_lap in range(N_laps_corr):
                print (i_lap, '\t', self.i_Laps_ImData[i_laps[i_lap]])
        else :
            print (i_lap, '\t', self.i_Laps_ImData[i_laps[i_lap]])

    def plot_dF_lapstarts(self, cellid):
        corridor_types = np.unique(np.array(self.all_corridor_start_IDs))
        corridor_types = corridor_types[corridor_types >= 0]

        colors = ['coral', 'lime', 'peru', 'deepskyblue', 'olive', 'deeppink', 'teal']

        fig, ax = plt.subplots(2,1,squeeze=False, figsize=(10,6), sharex=True, sharey=True)
        ax[0,0].plot(self.frame_times - self.im_reftime, self.dF_F[cellid,:], '-k', alpha=0.5)
        ax[0,0].plot(self.frame_times - self.im_reftime, self.spks[cellid,:] * self.frame_period, '-', c='deepskyblue', alpha=0.5)
        ax[0,0].set_title('absolute time - old')
        # ax[0,0].vlines(np.array(self.all_corridor_start_time) - self.im_reftime, 0, 200, colors=np.array(self.all_corridor_start_IDs) + 1)
        ax[1,0].plot(self.frame_times, self.dF_F[cellid,:], '-k', alpha=0.5)
        ax[1,0].plot(self.frame_times, self.spks[cellid,:] * self.frame_period, '-', c='deepskyblue', alpha=0.5)
        ax[1,0].set_title('relative time - new')

        i_col = 0
        for c_type in corridor_types:
            ii = np.where(np.array(self.all_corridor_start_IDs) == c_type)[0]
            if (c_type == 0):
                corr_color = 'silver'
            else:
                corr_color = colors[int(i_col)]
                if (i_col < 7):
                    i_col = i_col + 1
            corrname = 'corridor' + str(c_type)
            ax[0,0].vlines(np.array(self.all_corridor_start_time)[ii], 0, 2, colors=corr_color)
            ax[1,0].vlines(np.array(self.all_corridor_start_time)[ii], 0, 2, colors=corr_color, label=corrname)

        ax[1,0].set_xlim(min(self.frame_times), max(self.frame_times))       
        ax[1,0].legend()
        plt.show(block=False)


    def plot_ratemaps(self, corridor=-1, normalized=False, sorted=False, corridor_sort=-1, cellids=np.array([-1]), vmax=0, ratemaps_array = [], ratemaps_title = [], filename=None):
        ## plot the average event rate of all cells in a given corridor
        ## corridor: integer or array... (corridor id)
        ##              INTEGER: if you want to plot default ratemaps
        ##                           if corridor == -1 then all corridors are used
        ##              ARRAY :  if you want plot custom-defined ratemaps
        ##                           corridor must be an array to be able to add reward zones to every ratemap defined in ratemaps_array
        ## normalized: True or False. If True then  each cell ratemap is normalized to have a max = 1
        ## sorted: sorting the ratemaps by their peaks
        ## corridor_sort: which corridor to use for sorting the ratemaps
        ##              Corridor ID of the ratemap            
        ##                          if you plot default ratemaps 
        ##              Index of ratemap 
        ##                          if you plot custom-defined ratemaps
        ##                          (in this case there can be multiple ratemaps from same corridor)
        ## cellids: np array with the indexes of the cells to be plotted. when -1: all cells are plotted
        ## vmax: float. If ratemaps are not normalised then the max range of the colors will be at least vmax. 
                # If one of the ratemaps has a higher peak, then vmax is replaced by that peak 
        ## ratemaps_array: a list containing the ratemaps as numpy arrays to visualize, if custom ratemaps are to be plotted. By default it is empy and default ratemaps are plotted.
        ## ratemaps_title: a list containing N strings which is used for annotating the corresponding ratemaps. Must have same length as ratemaps_array. Only needed if ratemaps_array is given.
        ## filename: optional string. The name of the pdf file to save the figure.
        
        # checking whether we use default ratemaps or a different set of custom-defined ratemaps - and store this info in ratemap_base variable
        if type(corridor) == int:
            ratemap_base = 'all'
        if type(corridor) == list:
            ratemap_base = 'spec'
        if (type(corridor) != int) and (type(corridor) != list):
            print('corridor variable ill-defined, returning')
            return
        
        # checking equal length of input list
        if ratemap_base == 'spec':
            ratemaps = ratemaps_array
            if len(ratemaps_array)==0:
                print('Please specify ratemaps')
                return
            if len(ratemaps_array) != len(ratemaps_title):
                print('Ratemaps and Titles must have same size!')
                return
            if len(ratemaps_array) != len(corridor):
                print('ratemaps and corridor must have same size!')
                return
            N_all_cells = ratemaps[0].shape[1]

        # checking specified corridor is valid, check if it is a singla ratemap to plot sorted in single variable
        else:
            N_all_cells = self.N_cells
            if (corridor == -1):
                ratemaps = self.ratemaps
                single = False
            else:
                if not(np.any(self.corridors == corridor)):
                    print('corridor specified is invalid, or does not have enough laps to calculate ratemaps with it')
                    return
                else:
                    single = True
                    i_corrid = int(np.nonzero(self.corridors==corridor)[0])
                    ratemaps = [self.ratemaps[i_corrid]]
                    
        #ncells
        if cellids[0] != -1:
            if len(cellids) < N_all_cells:
                title_string_base = 'some cells '
            else:
                title_string_base = 'all cells '
        else:
            cellids = np.arange(N_all_cells)
            title_string_base = 'all cells '
        
            
        #sort - selecting ratemap as template for sorting
        if sorted:
            #select ratemap to sorted according to, and set title stored in sort_title 
            sort_title = 'sorted'
            if ratemap_base == 'spec':
                if corridor_sort == -1:
                    ratemap_to_sort = np.copy(ratemaps[0][:,cellids])
                    print('Sorting unspecified, sorting according to first given ratemap')
                    sort_title = sort_title + ' by ' + str(ratemaps_title[0])
                else:
                    if corridor_sort < len(ratemaps):
                        ratemap_to_sort = np.copy(ratemaps[corridor_sort][:,cellids])
                        sort_title = sort_title + ' by ' + str(ratemaps_title[corridor_sort])
                    else:
                        print('sort index',corridor_sort, 'too large - aborting!')
                        return
            else:
                if single:
                    ratemap_to_sort = np.copy(ratemaps[0][:,cellids])
                    sort_title = sort_title + ' by ' + str(self.corridors[0])
                else:
                    if corridor_sort == -1:
                        print('Sorting unspecified, sorting according to corridor', self.corridors[0])
                        ratemap_to_sort = np.copy(ratemaps[0][:,cellids])
                        sort_title = sort_title + ' by ' + str(self.corridors[0])

                    else:
                        if not(np.any(self.corridors == corridor_sort)):
                            print('Corridor specified for sorting is not among corridors with enough laps - sorting according to first')
                            ratemap_to_sort = np.copy(ratemaps[0][:,cellids])
                            sort_title = sort_title + ' by ' + str(self.corridors[0])
                        else:
                            sort_ratemap_index = int(np.nonzero(self.corridors==corridor_sort)[0])
                            ratemap_to_sort = np.copy(ratemaps[sort_ratemap_index][:,cellids])
                            sort_title = sort_title + ' by ' + str(self.corridors[sort_ratemap_index])
                      
            #storing sort order in sort_index 
            ratemap_to_sort = np.transpose(ratemap_to_sort)
            sort_index, rmaps = self.sort_ratemaps(ratemap_to_sort)
        else:
            sort_title = 'unsorted'
            sort_index = np.arange(len(cellids))               

        #max value for plotting
        if (normalized):
            vmax = 1
        else: 
            for i in range(len(ratemaps)):
                if (np.nanmax(ratemaps[i]) > vmax):
                    vmax = np.nanmax(ratemaps[i])
                    
        #reward zones
        zone_starts = []
        zone_ends = []
        for i in range(len(ratemaps)):
            if ratemap_base == 'spec':
                if not(np.any(self.corridors == corridor[i])):
                    print('Corridor', i , 'seems to be invalid - aborting')
                    return
                else:
                    i_corrid = int(np.nonzero(self.corridors==corridor[i])[0])
                    zone_starts.append(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_starts)
                    zone_ends.append(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_ends)
            else:
                if single:
                    zone_starts.append(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_starts)
                    zone_ends.append(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_ends)
                else:
                    i_corrid = i
                    zone_starts.append(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_starts)
                    zone_ends.append(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_ends)
        
        #plotting
        fig, axs = plt.subplots(1, len(ratemaps), figsize=(len(ratemaps)*3.5,8), sharex=True, sharey=True, squeeze=False)
        ims = []
        for i in range(len(ratemaps)):
            ratemap_to_sort = np.transpose(np.copy(ratemaps[i][:,cellids]))
            ratemap_to_plot = ratemap_to_sort[sort_index,:]
            #normalising if needed
            if (normalized):
                for j in range(ratemap_to_plot.shape[0]):
                    rate_range = max(ratemap_to_plot[j,:]) - min(ratemap_to_plot[j,:])
                    ratemap_to_plot[j,:] = (ratemap_to_plot[j,:] - min(ratemap_to_plot[j,:])) / rate_range
            # subplot title
            if ratemap_base == 'spec':
                title_string = ratemaps_title[i]
            else:
                if single:
                    title_string = title_string_base + ' in corridor ' + str(corridor)
                else:
                    title_string = title_string_base + ' in corridor ' + str(self.corridors[i])
            # plot
            axs[0,i].set_title(title_string)
            ims.append(axs[0,i].matshow(ratemap_to_plot, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='binary'))
            axs[0,i].set_facecolor(matcols.CSS4_COLORS['palegreen'])
            if (len(cellids) < 30):
                axs[0,i].set_yticks(np.arange(len(cellids)))
                axs[0,i].set_yticklabels(cellids[sort_index])
            plt.colorbar(ims[i], orientation='horizontal',ax=axs[0,i])
            # add reward-zone
            bottom, top = axs[0,i].get_ylim()
            for zone in range(len(zone_starts[i])):
                left = zone_starts[i][zone] * self.N_pos_bins             
                right = zone_ends[i][zone] * self.N_pos_bins
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                axs[0,i].add_patch(polygon)
                
        fig.suptitle(sort_title)
        fig.tight_layout()
        if (filename is None):
            plt.show(block=False)
        else:
            plt.savefig(filename, format='pdf')
            plt.close()

        return sort_index

    
    def sort_ratemaps(self, rmap):
        max_loc = np.argmax(rmap, 1)
        sorted_index = np.argsort(-1*max_loc)
        sorted_rmap = rmap[sorted_index,:]
        return sorted_index, sorted_rmap
    

    #create title string for the given corridor, cell with added info 
    def CreateTitle(self, corridor, cellid):
        ##select the appropriate numpy array index to contain properties for userspecified corridor
        CI=-2#the specified corridor's index among nonzero corridors
        for corr in range(len(self.corridors)):
            if self.corridors[corr]==corridor:
                CI=corr#-1#always corridor 0 starts
                cell_info='\n'+ 'skgs: '+str(round(self.cell_skaggs[CI][cellid],2))+' %actF: '+str(round(self.cell_activelaps_df[CI][cellid],2))+' %actS: '+str(round(self.cell_activelaps[CI][cellid],2))+'\n'+'TunSp: '+str(round(self.cell_tuning_specificity[CI][cellid],2))+' FF: '+str(round(self.cell_Fano_factor[CI][cellid],2))+' rate: '+str(round(self.cell_rates[CI][cellid],2))+' rel: '+str(round(self.cell_reliability[CI][cellid],2))    
                break
        if CI==-2:
            print('Warning: specified corridor does not exist in this session!')
        return(cell_info)


    def plot_cell_laps(self, cellid, multipdf_object=-1, signal='dF', corridor=-1, reward=True, write_pdf=False, plot_laps='all', n_grey_bins=100, plot_BTSP=False):
        ## plot the activity of a single cell in all trials in a given corridor
        ## signal can be 
        ##          'dF' when dF/F and spikes are plotted as a function of time
        ##          'rate' when rate vs. position is plotted
        ## plot_laps can be either 'all', 'correct' or 'error'

        #process input corridors
        if (corridor == -1):
            corridor = self.corridors
        else:
            if corridor in self.corridors:
                corridor=np.array([corridor])
            else:
                print('Warning: specified corridor does not exist in this session!')
                return

        #plotting
        if (signal == 'dF'):
            fig, ax = plt.subplots(1,corridor.size,squeeze=False, figsize=(6*corridor.size,8), sharex=True)
            for cor_index in range(corridor.size):
                if corridor.size==1:
                    corridor_to_plot=corridor
                else:
                    corridor_to_plot=corridor[cor_index]
                cell_info=self.CreateTitle(corridor_to_plot, cellid)

                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps = self.i_Laps_ImData[np.nonzero(icorrids == corridor_to_plot)[0]]
                
                reward_times = []     
                dFs = []
                spikes = []
                times = []
                for i_lap in i_laps:
                    dFs_normalised = self.ImLaps[i_lap].frames_dF_F[cellid,:]
                    if (self.elfiz == True):
                        dFs_normalised = (dFs_normalised - np.nanmin(dFs_normalised)) / 10
                    tt = self.ImLaps[i_lap].frames_time - np.nanmin(self.ImLaps[i_lap].frames_time)
                    # print(i_lap, np.nanmin(dFs_normalised), np.nanmin(tt), np.nanmax(tt))
                    dFs.append(dFs_normalised)
                    spikes.append(self.ImLaps[i_lap].frames_spikes[cellid,:])
                    times.append(tt)
                    reward_times.append(self.ImLaps[i_lap].reward_times - np.nanmin(self.ImLaps[i_lap].frames_time))
    
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
            min_intensity=0
            max_intensity=100

            if plot_BTSP:
                is_cell_with_unreliable_pf = cellid in [pf.cellid for pf in self.btsp_analysis.unreliable_place_fields]
                is_cell_with_early_pf = cellid in [pf.cellid for pf in self.btsp_analysis.early_place_fields]
                is_cell_with_transient_pf = cellid in [pf.cellid for pf in self.btsp_analysis.transient_place_fields]
                is_cell_with_nonbtsp_novel_pf = cellid in [pf.cellid for pf in self.btsp_analysis.nonbtsp_novel_place_fields]
                is_cell_with_btsp_pf = cellid in [pf.cellid for pf in self.btsp_analysis.btsp_place_fields]

            for cor_index in range(corridor.size):
                corridor_to_plot=corridor[cor_index]
                
                #calculate rate matrix - to set the scales right
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]
                
                total_spikes = self.activity_tensor[:,cellid,i_laps]
                total_time = self.activity_tensor_time[:,i_laps]
                rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)
                
                loc_max=np.nanmax(rate_matrix[rate_matrix != np.inf])
                loc_min=np.nanmin(rate_matrix)
                if  loc_max > max_intensity :
                    max_intensity = loc_max
                if  loc_min < min_intensity:
                    min_intensity = loc_min

            # main part of ratemap plotting
            # colormap: grey for 0-100 and red for higher
            # max_intensity: 100 or higher, the highest rate
            # min_intensity: 0 or lower, the lowest rate

            colors1 = plt.cm.binary(np.linspace(0., 1, n_grey_bins)) 
            n_col2 = int(np.round(max_intensity - n_grey_bins))
            max_col2 = min(0.65, 0.25 + n_col2 / 100)
            colors2 = plt.cm.autumn(np.linspace(0, max_col2, n_col2))

            # combine them and build a new colormap
            colors = np.vstack((colors1, colors2))
            mymap = matcols.LinearSegmentedColormap.from_list('my_colormap', colors)
            nbins = self.activity_tensor.shape[0]

            if plot_BTSP:
                if any([is_cell_with_unreliable_pf, is_cell_with_early_pf, is_cell_with_transient_pf, is_cell_with_nonbtsp_novel_pf, is_cell_with_btsp_pf]):
                    fig, ax = plt.subplots(4, corridor.size, squeeze=False, figsize=(6 * corridor.size, 10))
                else:
                    fig, ax = plt.subplots(2, corridor.size, squeeze=False, sharex=True, sharey='row', figsize=(6*corridor.size,10))
            else:
                fig, ax = plt.subplots(4, corridor.size, squeeze=False, sharex=True, sharey='row', figsize=(6 * corridor.size, 10))

            for cor_index in range(corridor.size):
                if corridor.size==1:
                    corridor_to_plot=corridor
                else:
                    corridor_to_plot=corridor[cor_index]
                cell_info=self.CreateTitle(corridor_to_plot, cellid)   
                
                # getting rewarded corridors
                icorrids = self.i_corridors[self.i_Laps_ImData] # corridor ids with image data
                i_laps_beh = self.i_Laps_ImData[np.nonzero(icorrids == corridor_to_plot)[0]]
                correct_reward = np.zeros([2, len(i_laps_beh)])
                ii = 0
                for i_lap in i_laps_beh:
                    if (self.ImLaps[i_lap].correct == True):
                        correct_reward[0,ii] = 1
                    if (len(self.ImLaps[i_lap].reward_times) > 0):
                        correct_reward[1,ii] = 1
                    ii = ii + 1
                
                # select the laps in the corridor (these are different indexes from upper ones!)
                # only laps with imaging data are selected - this will index the activity_tensor
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]               
                
                #calculate rate matrix
                total_spikes = self.activity_tensor[:,cellid,i_laps]
                total_time = self.activity_tensor_time[:,i_laps]
                rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)

                if (plot_laps == 'correct'):
                    i_error = np.nonzero(correct_reward[0,:] == 0)[0]
                    rate_matrix[:,i_error] = np.nan
                if (plot_laps == 'error'):
                    i_correct = np.nonzero(correct_reward[0,:] == 1)[0]
                    rate_matrix[:,i_correct] = np.nan

                #calculate average rates for plotting
                average_firing_rate=np.nansum(rate_matrix, axis=1)/i_laps.size
                std=np.nanstd(rate_matrix, axis=1)/np.sqrt(i_laps.size)
                errorbar_x=np.arange(self.N_pos_bins)
                
                #plotting
                title_string = 'ratemap of cell ' + str(cellid) + ' in corridor ' + str(corridor_to_plot)+cell_info
                ax[0,cor_index].set_title(title_string)
                ax[1,cor_index].fill_between(errorbar_x,average_firing_rate+std, average_firing_rate-std, alpha=0.3)
                ax[1,cor_index].plot(average_firing_rate,zorder=0)

                if plot_BTSP:
                    if is_cell_with_unreliable_pf:
                        unreliable_place_fields = [pf for pf in self.btsp_analysis.unreliable_place_fields if pf.cellid == cellid and pf.cor_index == cor_index]
                        for unreliable_field in unreliable_place_fields:
                            # plot place field bounds (spatial bins)
                            lb, ub = unreliable_field.bounds
                            ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)

                            ax[0,cor_index].axvspan(lb, ub, color="black", alpha=0.15)
                            ax[1,cor_index].axvspan(lb, ub, color="black", alpha=0.15)

                    if is_cell_with_early_pf:
                        early_place_fields = [pf for pf in self.btsp_analysis.early_place_fields if pf.cellid == cellid and pf.cor_index == cor_index]
                        for early_field in early_place_fields:
                            # plot place field bounds (spatial bins)
                            lb, ub = early_field.bounds
                            ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                            fl_uc = early_field.formation_lap_uncorrected / len(i_laps)
                            fl = early_field.formation_lap / len(i_laps)
                            el = (early_field.end_lap+1) / len(i_laps)

                            # plot place field bounds (spatial bins)
                            ax[0,cor_index].axvspan(lb, ub, ymin=fl_uc, ymax=el, color="cyan", alpha=0.15)
                            ax[1,cor_index].axvspan(lb, ub, color="cyan", alpha=0.15)

                            # plot max rates in each lap between place field boundaries
                            rate_matrix_pf = early_field.rate_matrix[lb:ub,:]
                            max_rates_pf = np.nanmax(rate_matrix_pf,axis=0) / np.nanmax(rate_matrix_pf)
                            active_laps = np.where(max_rates_pf > PF.FORMATION_THRESHOLD)[0].size  # laps where max activity exceeds 10% of max activity along all laps in pf
                            ax[2,cor_index].plot(max_rates_pf, linewidth=1.5, label=f"[{lb}-{ub-1}], act.laps: {active_laps}", marker="x")
                            ax[2,cor_index].hlines(PF.FORMATION_THRESHOLD, xmin=0, xmax=len(i_laps), colors='r', linestyles="dashed")
                            if ax[2,cor_index].lines: ax[2,cor_index].legend(loc="upper right")

                            # plot uncorrected formation lap
                            ax[0, cor_index].plot(lb, early_field.formation_lap_uncorrected, marker=5, markersize=5, markeredgecolor="cyan", markerfacecolor="None", linestyle='None')
                            ax[0, cor_index].plot(ub, early_field.formation_lap_uncorrected, marker=4, markersize=5, markeredgecolor="cyan", markerfacecolor="None", linestyle='None')
                            ax[2, cor_index].axvline(early_field.formation_lap_uncorrected, color="cyan", linestyle="dashed")

                            # plot formation lap
                            ax[0, cor_index].plot(lb, early_field.formation_lap, marker=5, markersize=5, color="cyan", linestyle='None')
                            ax[0, cor_index].plot(ub, early_field.formation_lap, marker=4, markersize=5, color="cyan", linestyle='None')
                            ax[2, cor_index].axvline(early_field.formation_lap, color="cyan")

                            # plot end lap
                            ax[0, cor_index].plot(lb, early_field.end_lap, marker=5, markersize=5, color="cyan", linestyle='None')
                            ax[0, cor_index].plot(ub, early_field.end_lap, marker=4, markersize=5, color="cyan", linestyle='None')
                            ax[2, cor_index].axvline(early_field.end_lap, color="darkturquoise")

                    if is_cell_with_transient_pf:
                        transient_fields = [pf for pf in self.btsp_analysis.transient_place_fields if pf.cellid == cellid and pf.cor_index == cor_index]
                        for transient_field in transient_fields:
                            lb, ub = transient_field.bounds
                            ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                            fl_uc = transient_field.formation_lap_uncorrected / len(i_laps)
                            fl = transient_field.formation_lap / len(i_laps)
                            el = (transient_field.end_lap+1) / len(i_laps)

                            # plot place field bounds (spatial bins)
                            ax[0,cor_index].axvspan(lb, ub, ymin=fl, ymax=el, color="orange", alpha=0.15)
                            ax[1,cor_index].axvspan(lb, ub, color="orange", alpha=0.15)

                            # plot max rates in each lap between place field boundaries
                            rate_matrix_pf = transient_field.rate_matrix[lb:ub,:]
                            max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                            active_laps = np.where(max_rates_pf > PF.FORMATION_THRESHOLD)[0].size  # laps where max activity exceeds 10% of max activity along all laps in pf
                            ax[2,cor_index].plot(max_rates_pf, linewidth=1.5, label=f"[{lb}-{ub-1}], act.laps: {active_laps}", marker="x")
                            ax[2,cor_index].hlines(PF.FORMATION_THRESHOLD, xmin=0, xmax=len(i_laps), colors='r', linestyles="dashed")
                            if ax[2,cor_index].lines: ax[2,cor_index].legend(loc="upper right")

                            # plot uncorrected formation lap
                            ax[0, cor_index].plot(lb, transient_field.formation_lap_uncorrected, marker=5, markersize=5, markeredgecolor="orange", markerfacecolor="None", linestyle='None')
                            ax[0, cor_index].plot(ub, transient_field.formation_lap_uncorrected, marker=4, markersize=5, markeredgecolor="orange", markerfacecolor="None", linestyle='None')
                            ax[2, cor_index].axvline(transient_field.formation_lap_uncorrected, color="orange", linestyle="dashed")

                            # plot formation lap
                            ax[0, cor_index].plot(lb, transient_field.formation_lap, marker=5, markersize=5, color="orange", linestyle='None')
                            ax[0, cor_index].plot(ub, transient_field.formation_lap, marker=4, markersize=5, color="orange", linestyle='None')
                            ax[2, cor_index].axvline(transient_field.formation_lap, color="orange")

                            # plot end lap
                            ax[0, cor_index].plot(lb, transient_field.end_lap, marker=5, markersize=5, color="orange", linestyle='None')
                            ax[0, cor_index].plot(ub, transient_field.end_lap, marker=4, markersize=5, color="orange", linestyle='None')
                            ax[2, cor_index].axvline(transient_field.end_lap, color="darkorange")


                            output_subfolder = "transient"
                            if not os.path.exists(f"{multipdf_object}/{output_subfolder}"):
                                os.makedirs(f"{multipdf_object}/{output_subfolder}")

                    if is_cell_with_nonbtsp_novel_pf or is_cell_with_btsp_pf:
                        # plot 95th percentile from shuffling over each bin to average firing plot
                        place_cells = np.union1d(self.tuned_cells[0], self.tuned_cells[1])  # these are the cells from shuffle
                        i_cell = np.where(place_cells == cellid)[0]
                        p95_cell = self.p95[cor_index][:, i_cell]
                        ax[1, cor_index].plot(p95_cell, "r")

                        # select place fields belonging to the current cell, current corridor
                        place_fields_cell = list(filter(lambda pf: pf.cellid == cellid and pf.cor_index == cor_index, self.btsp_analysis.candidate_btsp_place_fields))
                        for place_field in place_fields_cell:
                            lb, ub = place_field.bounds
                            ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                            fl_uc = place_field.formation_lap_uncorrected / len(i_laps)
                            fl = place_field.formation_lap / len(i_laps)
                            el = (place_field.end_lap+1) / len(i_laps)

                            # plot place field bounds
                            criteria_missed = []
                            color = "darkviolet"
                            dark_color = "purple"
                            if not place_field.has_high_gain:
                                color = "red"
                                dark_color = "crimson"
                                criteria_missed.append("low gain")
                            if not place_field.has_no_drift:
                                color = "red"
                                dark_color = "crimson"
                                criteria_missed.append("drifting")
                            if not place_field.has_backwards_shift:
                                color = "red"
                                dark_color = "crimson"
                                criteria_missed.append("no backshift")
                            ax[0, cor_index].axvspan(lb, ub, ymin=fl, ymax=el, color=color, alpha=0.15)
                            ax[1, cor_index].axvspan(lb, ub, color=color, alpha=0.15)

                            # plot max rates in each lap between place field boundaries
                            rate_matrix_pf = place_field.rate_matrix[lb:ub,:]
                            max_rates_pf = np.nanmax(rate_matrix_pf,axis=0) / np.nanmax(rate_matrix_pf)
                            label = f"[{lb}-{ub-1}]"
                            for criterion_missed in criteria_missed:
                                label = f"{label}, {criterion_missed}"
                            ax[2,cor_index].plot(max_rates_pf, linewidth=1.5, label=label, marker="x")
                            ax[2,cor_index].hlines(PF.FORMATION_THRESHOLD, xmin=0, xmax=len(i_laps), colors='r', linestyles="dashed")
                            if ax[2,cor_index].lines: ax[2,cor_index].legend(loc="upper right")

                            # plot uncorrected formation lap
                            ax[0, cor_index].plot(lb, place_field.formation_lap_uncorrected, marker=5, markersize=5, markeredgecolor=color, markerfacecolor="None", linestyle='None')
                            ax[0, cor_index].plot(ub, place_field.formation_lap_uncorrected, marker=4, markersize=5, markeredgecolor=color, markerfacecolor="None", linestyle='None')
                            ax[2, cor_index].axvline(place_field.formation_lap_uncorrected, color=color, linestyle="dashed")

                            # plot formation lap
                            ax[0, cor_index].plot(lb, place_field.formation_lap, marker=5, markersize=5, color=color, linestyle='None')
                            ax[0, cor_index].plot(ub, place_field.formation_lap, marker=4, markersize=5, color=color, linestyle='None')
                            ax[2, cor_index].axvline(place_field.formation_lap, color=color)

                            # plot end lap
                            ax[0, cor_index].plot(lb, place_field.end_lap, marker=5, markersize=5, color=color, linestyle='None')
                            ax[0, cor_index].plot(ub, place_field.end_lap, marker=4, markersize=5, color=color, linestyle='None')
                            ax[2, cor_index].axvline(place_field.end_lap, color=dark_color)

                            # plot shift scores, correlation values
                            r, p = place_field.spearman_corrs
                            ax[3, cor_index].plot(place_field.com_diffs, label=f"[{lb}-{ub-1}], r={np.round(r,2)}, p={np.round(p,3)}")
                            ax[3, cor_index].axvline(x=PF.SPEARMAN_SKIP_LAPS, color="cyan")
                            if ax[3, cor_index].lines: ax[3, cor_index].legend(loc="upper right")

                            if not np.isnan(place_field.linear_coeffs[0]):
                                m, b = place_field.linear_coeffs
                                ax[3, cor_index].axline(xy1=(PF.SPEARMAN_SKIP_LAPS, b), slope=m, color="green")
                                ax[3, cor_index].plot(0, m * (-PF.SPEARMAN_SKIP_LAPS) + b, marker="x", markersize=10,color="green")
                                first_laps_avg = np.mean(place_field.com_diffs[:PF.SPEARMAN_SKIP_LAPS + 1])
                                ax[3, cor_index].plot(0, first_laps_avg, marker="_", markersize=8, markerfacecolor='none', color="red")
                            ax[3, cor_index].axvline(0, color="k", linestyle="dotted")
                            ax[3, cor_index].axhline(0, color="k", linestyle="dotted")

                        # if no place field in a corridor, hide subplots
                        if not ax[2, cor_index].lines: ax[2, cor_index].set_visible(False)
                        if not ax[3, cor_index].lines: ax[3, cor_index].set_visible(False)

                n_laps = rate_matrix.shape[1]

                if corridor.size>1:
                    im1 = ax[0,cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower',vmin=min_intensity, vmax=max_intensity, cmap=mymap)
                else:
                    im1 = ax[0,cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower',vmin=min_intensity, vmax=max_intensity, cmap=mymap)
                if (reward == True):
                    i_cor = np.nonzero(correct_reward[0,:])[0]
                    n_cor = len(i_cor)
                    ax[0, cor_index].scatter(np.ones(n_cor)*nbins, i_cor, marker="_", color='C0')
                    i_rew = np.nonzero(correct_reward[1,:])[0]
                    n_cor = len(i_rew)
                    ax[0, cor_index].scatter(np.ones(n_cor)*(nbins+0.5), i_rew, marker="_", color='C1')

                plt.colorbar(im1, orientation='horizontal',ax=ax[1,cor_index])
                ax[0,cor_index].set_xlim(0, nbins+1)
                ax[1,cor_index].set_xlim(0, nbins+1)
                ax[0,cor_index].set_facecolor(matcols.CSS4_COLORS['palegreen'])

            ## add reward zones - rewardZones
            for cor_index in range(corridor.size):
                zone_starts = self.corridor_list.corridors[corridor[cor_index]].reward_zone_starts
                if (len(zone_starts) > 0):
                    zone_ends = self.corridor_list.corridors[corridor[cor_index]].reward_zone_ends
                    bottom, top = ax[0,cor_index].get_ylim()
                    for i_zone in range(len(zone_starts)):
                        left = zone_starts[i_zone] * nbins             
                        right = zone_ends[i_zone] * nbins              
                        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                        ax[0,cor_index].add_patch(polygon)
                        # print('adding reward zone to the ', cor_index, 'th corridor, ', self.corridors[cor_index+1])

            #write pdf if asked
            if write_pdf==True and multipdf_object!=-1:
                if plot_BTSP:
                    if is_cell_with_unreliable_pf:
                        if not os.path.exists(f"{multipdf_object}/unreliable"):
                            os.makedirs(f"{multipdf_object}/unreliable")
                        filename = f"{multipdf_object}/unreliable/{cellid}.pdf"
                        plt.savefig(filename, format='pdf')
                    if is_cell_with_early_pf:
                        if not os.path.exists(f"{multipdf_object}/early"):
                            os.makedirs(f"{multipdf_object}/early")
                        filename = f"{multipdf_object}/early/{cellid}.pdf"
                        plt.savefig(filename, format='pdf')
                    if is_cell_with_transient_pf:
                        if not os.path.exists(f"{multipdf_object}/transient"):
                            os.makedirs(f"{multipdf_object}/transient")
                        filename = f"{multipdf_object}/transient/{cellid}.pdf"
                        plt.savefig(filename, format='pdf')
                    if is_cell_with_nonbtsp_novel_pf:
                        if not os.path.exists(f"{multipdf_object}/non_btsp"):
                            os.makedirs(f"{multipdf_object}/non_btsp")
                        filename = f"{multipdf_object}/non_btsp/{cellid}.pdf"
                        plt.savefig(filename, format='pdf')
                    if is_cell_with_btsp_pf:
                        if not os.path.exists(f"{multipdf_object}/btsp"):
                            os.makedirs(f"{multipdf_object}/btsp")
                        filename = f"{multipdf_object}/btsp/{cellid}.pdf"
                        plt.savefig(filename, format='pdf')
                    if not is_cell_with_unreliable_pf and \
                       not is_cell_with_early_pf and \
                       not is_cell_with_transient_pf and \
                       not is_cell_with_nonbtsp_novel_pf and \
                       not is_cell_with_btsp_pf:
                        if not os.path.exists(f"{multipdf_object}/uncategorized"):
                            os.makedirs(f"{multipdf_object}/uncategorized")
                        filename = f"{multipdf_object}/uncategorized/{cellid}.pdf"
                        plt.savefig(filename, format='pdf')
                else:
                    plt.savefig(multipdf_object, format='pdf')

            fig.tight_layout()
            #plt.show(block=False)

            if write_pdf==True:
                plt.close()

    def plot_cell_laps_poster(self, area, cellid, corridor_abs=-1):
        # add font
        from matplotlib import font_manager
        font_dirs = ['C:\\home\\phd\\']
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = 'Trade Gothic Next LT Pro BdCn'

        if (corridor_abs == -1):
            corridor = self.corridors
        else:
            if corridor_abs in self.corridors:
                corridor = np.array([corridor_abs])
            else:
                print('Warning: specified corridor does not exist in this session!')
                return

        is_cell_with_unreliable_pf = cellid in [pf.cellid for pf in self.btsp_analysis.unreliable_place_fields]
        is_cell_with_early_pf = cellid in [pf.cellid for pf in self.btsp_analysis.early_place_fields]
        is_cell_with_transient_pf = cellid in [pf.cellid for pf in self.btsp_analysis.transient_place_fields]
        is_cell_with_nonbtsp_novel_pf = cellid in [pf.cellid for pf in self.btsp_analysis.nonbtsp_novel_place_fields]
        is_cell_with_btsp_pf = cellid in [pf.cellid for pf in self.btsp_analysis.btsp_place_fields]

        fig, ax = plt.subplots(1, corridor.size, squeeze=False, figsize=(2.4, 1.3), dpi=200)
        nbins = self.activity_tensor.shape[0]
        for cor_index in range(corridor.size):
            if corridor.size == 1:
                corridor_to_plot = corridor
            else:
                corridor_to_plot = corridor[cor_index]
            cell_info = self.CreateTitle(corridor_to_plot, cellid)

            # select the laps in the corridor (these are different indexes from upper ones!)
            # only laps with imaging data are selected - this will index the activity_tensor
            i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]

            # calculate rate matrix
            total_spikes = self.activity_tensor[:, cellid, i_laps]
            total_time = self.activity_tensor_time[:, i_laps]
            rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)

            # calculate average rates for plotting
            average_firing_rate = np.nansum(rate_matrix, axis=1) / i_laps.size
            std = np.nanstd(rate_matrix, axis=1) / np.sqrt(i_laps.size)
            errorbar_x = np.arange(self.N_pos_bins)

            # plotting
            #ax[1, cor_index].fill_between(errorbar_x, average_firing_rate + std, average_firing_rate - std, alpha=0.3)
            #ax[1, cor_index].plot(average_firing_rate, zorder=0)

            if is_cell_with_unreliable_pf:
                unreliable_place_fields = [pf for pf in self.btsp_analysis.unreliable_place_fields if
                                           pf.cellid == cellid and pf.cor_index == np.where(self.corridors == corridor_abs)[0]]
                for unreliable_field in unreliable_place_fields:
                    # plot place field bounds (spatial bins)
                    lb, ub = unreliable_field.bounds
                    ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)

                    ax[0, cor_index].axvspan(lb, ub, color="black", alpha=0.15)
                    #ax[1, cor_index].axvspan(lb, ub, color="black", alpha=0.15)

            if is_cell_with_early_pf:
                early_place_fields = [pf for pf in self.btsp_analysis.early_place_fields if
                                      pf.cellid == cellid and pf.cor_index == np.where(self.corridors == corridor_abs)[0]]
                for early_field in early_place_fields:
                    # plot place field bounds (spatial bins)
                    lb, ub = early_field.bounds
                    ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                    fl_uc = early_field.formation_lap_uncorrected / len(i_laps)
                    fl = early_field.formation_lap / len(i_laps)
                    el = (early_field.end_lap + 1) / len(i_laps)

                    # plot place field bounds (spatial bins)
                    ax[0, cor_index].axvspan(lb, ub, ymin=fl_uc, ymax=el, color="cyan", alpha=0.15)
                    #ax[1, cor_index].axvspan(lb, ub, color="cyan", alpha=0.15)

                    # plot max rates in each lap between place field boundaries
                    rate_matrix_pf = early_field.rate_matrix[lb:ub, :]
                    max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                    active_laps = np.where(max_rates_pf > PF.FORMATION_THRESHOLD)[
                        0].size  # laps where max activity exceeds 10% of max activity along all laps in pf

                    # plot uncorrected formation lap
                    #ax[0, cor_index].plot(lb, early_field.formation_lap_uncorrected, marker=5, markersize=5,
                    #                      markeredgecolor="cyan", markerfacecolor="None", linestyle='None')
                    #ax[0, cor_index].plot(ub, early_field.formation_lap_uncorrected, marker=4, markersize=5,
                    #                      markeredgecolor="cyan", markerfacecolor="None", linestyle='None')

                    # plot formation lap
                    #ax[0, cor_index].plot(lb, early_field.formation_lap, marker=5, markersize=5, color="cyan",
                    #                      linestyle='None')
                    #ax[0, cor_index].plot(ub, early_field.formation_lap, marker=4, markersize=5, color="cyan",
                    #                      linestyle='None')

                    # plot end lap
                    #ax[0, cor_index].plot(lb, early_field.end_lap, marker=5, markersize=5, color="cyan",
                    #                      linestyle='None')
                    #ax[0, cor_index].plot(ub, early_field.end_lap, marker=4, markersize=5, color="cyan",
                    #                      linestyle='None')

            if is_cell_with_transient_pf:
                transient_fields = [pf for pf in self.btsp_analysis.transient_place_fields if
                                    pf.cellid == cellid and pf.cor_index == np.where(self.corridors == corridor_abs)[0]]
                for transient_field in transient_fields:
                    lb, ub = transient_field.bounds
                    ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                    fl_uc = transient_field.formation_lap_uncorrected / len(i_laps)
                    fl = transient_field.formation_lap / len(i_laps)
                    el = (transient_field.end_lap + 1) / len(i_laps)

                    # plot place field bounds (spatial bins)
                    ax[0, cor_index].axvspan(lb, ub, ymin=fl, ymax=el, color="orange", alpha=0.15)
                    #ax[1, cor_index].axvspan(lb, ub, color="orange", alpha=0.15)

                    # plot max rates in each lap between place field boundaries
                    rate_matrix_pf = transient_field.rate_matrix[lb:ub, :]
                    max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                    active_laps = np.where(max_rates_pf > PF.FORMATION_THRESHOLD)[
                        0].size  # laps where max activity exceeds 10% of max activity along all laps in pf

                    # plot uncorrected formation lap
                    #ax[0, cor_index].plot(lb, transient_field.formation_lap_uncorrected, marker=5, markersize=5,
                    #                      markeredgecolor="orange", markerfacecolor="None", linestyle='None')
                    #ax[0, cor_index].plot(ub, transient_field.formation_lap_uncorrected, marker=4, markersize=5,
                    #                      markeredgecolor="orange", markerfacecolor="None", linestyle='None')

                    # plot formation lap
                    #ax[0, cor_index].plot(lb, transient_field.formation_lap, marker=5, markersize=5, color="orange",
                    #                      linestyle='None')
                    #ax[0, cor_index].plot(ub, transient_field.formation_lap, marker=4, markersize=5, color="orange",
                    #                      linestyle='None')

                    # plot end lap
                    #ax[0, cor_index].plot(lb, transient_field.end_lap, marker=5, markersize=5, color="orange",
                    #                      linestyle='None')
                    #ax[0, cor_index].plot(ub, transient_field.end_lap, marker=4, markersize=5, color="orange",
                    #                      linestyle='None')

            if is_cell_with_nonbtsp_novel_pf or is_cell_with_btsp_pf:
                # plot 95th percentile from shuffling over each bin to average firing plot
                place_cells = np.union1d(self.tuned_cells[0],
                                         self.tuned_cells[1])  # these are the cells from shuffle
                i_cell = np.where(place_cells == cellid)[0]
                p95_cell = self.p95[cor_index][:, i_cell]
                #ax[1, cor_index].plot(p95_cell, "r")

                # select place fields belonging to the current cell, current corridor
                place_fields_cell = [pf for pf in self.btsp_analysis.candidate_btsp_place_fields if \
                                     pf.cellid == cellid and pf.cor_index == np.where(self.corridors == corridor_abs)[0]]
                for place_field in place_fields_cell:
                    lb, ub = place_field.bounds
                    ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                    fl_uc = place_field.formation_lap_uncorrected / len(i_laps)
                    fl = place_field.formation_lap / len(i_laps)
                    el = (place_field.end_lap + 1) / len(i_laps)

                    # plot place field bounds
                    criteria_missed = []
                    color = "darkviolet"
                    dark_color = "purple"
                    if not place_field.has_high_gain:
                        color = "red"
                        dark_color = "crimson"
                        criteria_missed.append("low gain")
                    if not place_field.has_no_drift:
                        color = "red"
                        dark_color = "crimson"
                        criteria_missed.append("drifting")
                    if not place_field.has_backwards_shift:
                        color = "red"
                        dark_color = "crimson"
                        criteria_missed.append("no backshift")
                    ax[0, cor_index].axvspan(lb, ub, ymin=fl, ymax=el, color=color, alpha=0.15)
                    #ax[1, cor_index].axvspan(lb, ub, color=color, alpha=0.15)

                    # plot max rates in each lap between place field boundaries
                    rate_matrix_pf = place_field.rate_matrix[lb:ub, :]
                    max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                    label = f"[{lb}-{ub - 1}]"
                    for criterion_missed in criteria_missed:
                        label = f"{label}, {criterion_missed}"

                    # plot uncorrected formation lap
                    #ax[0, cor_index].plot(lb, place_field.formation_lap_uncorrected, marker=5, markersize=5,
                    #                      markeredgecolor=color, markerfacecolor="None", linestyle='None')
                    #ax[0, cor_index].plot(ub, place_field.formation_lap_uncorrected, marker=4, markersize=5,
                    #                      markeredgecolor=color, markerfacecolor="None", linestyle='None')

                    # plot formation lap
                    #ax[0, cor_index].plot(lb, place_field.formation_lap, marker=5, markersize=5, color=color,
                    #                      linestyle='None')
                    #ax[0, cor_index].plot(ub, place_field.formation_lap, marker=4, markersize=5, color=color,
                    #                      linestyle='None')

                    # plot end lap
                    #ax[0, cor_index].plot(lb, place_field.end_lap, marker=5, markersize=5, color=color,
                    #                      linestyle='None')
                    #ax[0, cor_index].plot(ub, place_field.end_lap, marker=4, markersize=5, color=color,
                    #                      linestyle='None')

            n_laps = rate_matrix.shape[1]

            if corridor.size > 1:
                im1 = ax[0, cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower', cmap='binary', interpolation="none")
            else:
                im1 = ax[0, cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower', cmap='binary', interpolation="none")

            plt.ylabel("laps")
            plt.xlabel("spatial position")
            plt.colorbar(im1, orientation='vertical', ax=ax[0, cor_index])
            ax[0, cor_index].set_xlim(0, nbins)
            #ax[1, cor_index].set_xlim(0, nbins + 1)
            #ax[0, cor_index].set_facecolor(matcols.CSS4_COLORS['palegreen'])

        ## add reward zones - rewardZones
        for cor_index in range(corridor.size):
            zone_starts = self.corridor_list.corridors[corridor[cor_index]].reward_zone_starts
            if (len(zone_starts) > 0):
                zone_ends = self.corridor_list.corridors[corridor[cor_index]].reward_zone_ends
                bottom, top = ax[0, cor_index].get_ylim()
                for i_zone in range(len(zone_starts)):
                    left = zone_starts[i_zone] * nbins
                    right = zone_ends[i_zone] * nbins
                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True,
                                      color='green', alpha=0.15)
                    ax[0, cor_index].add_patch(polygon)
                    # print('adding reward zone to the ', cor_index, 'th corridor, ', self.corridors[cor_index+1])

        fig.tight_layout()
        filename = f"C:/home/phd/btsp_poster/pf_examples/{self.sessionID}_cell{cellid}.svg"
        plt.savefig(filename)
        plt.close()

    def run_btsp_analysis(self, cellid, shift_criterion=True):
        for cor_index in range(self.corridors.size):
            current_corridor = self.corridors[cor_index]
            i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == current_corridor)[0]

            # calculate rate matrix
            total_spikes = self.activity_tensor[:, cellid, i_laps]
            total_time = self.activity_tensor_time[:, i_laps]
            rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)
            average_firing_rate = np.nansum(rate_matrix, axis=1) / i_laps.size

            place_cells = np.union1d(self.tuned_cells[0], self.tuned_cells[1])  # these are the cells from shuffle  # TODO: generalize across corridors
            i_cell = np.where(place_cells == cellid)[0]
            p95_cell = self.p95[cor_index][:, i_cell]
            bins_p95_geq_afr = [i for i in range(self.N_pos_bins) if average_firing_rate[i] >= p95_cell[i]]
            
            btsp_analysis_corridor = BtspAnalysis(self.sessionID, cellid, rate_matrix,
                                                  corridors=self.corridors, bins_p95_geq_afr=bins_p95_geq_afr,
                                                  i_laps=i_laps, shift_criterion=shift_criterion)
            btsp_analysis_corridor.categorize_place_fields(cor_index)

            if isinstance(self.btsp_analysis, type(None)):  # first corridor in iteration
                self.btsp_analysis = btsp_analysis_corridor
            else:
                self.btsp_analysis += btsp_analysis_corridor

    def plot_popact(self, cellids, corridor=-1, name_string='selected_cells', bylaps=False, set_ymax=None):
        ## plot the total population activity in all trials in a given corridor
        ## cellids: numpy array. The index of the cells included in the population activity
        ## corridor: integer, the ID of a valid corridor
        ##      if corridor == -1 then all corridors are used
        ## bylaps: True or False. If True then population activity is plotted lap by lap


        #process input corridors
        if (corridor == -1):
            corridor = self.corridors
        else:
            if corridor in self.corridors:
                corridor=np.array(corridor)
            else:
                print('Warning: specified corridor does not exist in this session!')
                return
        nbins = self.activity_tensor.shape[0]
                
        ymax = 0              
        fig, ax = plt.subplots(1,corridor.size, squeeze=False, sharey='row', figsize=(6*corridor.size,4), sharex=True)
        for cor_index in range(corridor.size):
            if corridor.size==1:
                corridor_to_plot=corridor
            else:
                corridor_to_plot=corridor[cor_index]
                        
            # select the laps in the corridor (these are different indexes from upper ones!)
            # only laps with imaging data are selected - this will index the activity_tensor
            i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]               
            
            #calculate rate matrix
            sp_laps = self.activity_tensor[:,:,i_laps]
            sp = sp_laps[:,cellids,:]
            total_time = self.activity_tensor_time[:,i_laps]
            popact_laps = nan_divide(np.nansum(sp, 1), total_time, where=total_time > 0.025)
            

            #calculate for plotting average rates
            xmids=np.arange(self.N_pos_bins)
            title_string = 'total activity in corridor ' + str(corridor_to_plot)
            if (bylaps == True):
                if (cor_index==0):
                    scale_factor = np.round(np.nanmean(popact_laps))
                ax[0,cor_index].plot(popact_laps + np.arange(popact_laps.shape[1])*scale_factor, color='dodgerblue')
                ax[0,cor_index].set_title(title_string)
                ylab_text = 'lap number x ' + str(scale_factor) + ' / total activity'
                ax[0,cor_index].set_ylabel(ylab_text)
                ax[0,cor_index].set_xlabel('position bin')
            else:
                mean_rate=np.nansum(popact_laps, axis=1)/np.sum(1 - np.isnan(popact_laps), axis=1)
                se_rate=np.nanstd(popact_laps, axis=1)/np.sqrt(np.sum(1 - np.isnan(popact_laps), axis=1))
            
                if (max(mean_rate + se_rate) > ymax):
                    ymax = max(mean_rate + se_rate)

                #plotting
                ax[0,cor_index].fill_between(xmids,mean_rate+se_rate, mean_rate-se_rate, alpha=0.3)
                ax[0,cor_index].plot(mean_rate,zorder=0)
                # n_laps = popact_laps.shape[1]

                ax[0,cor_index].set_xlim(0, self.N_pos_bins)
                ax[0,cor_index].set_title(title_string)
                if (set_ymax is None):
                    ax[0,0].set_ylim(0, ymax)
                else :
                    ax[0,0].set_ylim(0, set_ymax)


            ## add reward zones
            zone_starts = self.corridor_list.corridors[corridor_to_plot].reward_zone_starts
            if (len(zone_starts) > 0):
                zone_ends = self.corridor_list.corridors[corridor_to_plot].reward_zone_ends
                bottom, top = ax[0,cor_index].get_ylim()
                for i_zone in range(len(zone_starts)):
                    left = zone_starts[i_zone] * nbins             
                    right = zone_ends[i_zone] * nbins              
                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                    ax[0,cor_index].add_patch(polygon)

        plt.show(block=False)

    def calc_correct_lap_proportions(self):
        selected_laps = np.arange(self.n_laps)
        if (self.n_laps > 0):
            corridor_types = np.unique(self.i_corridors[selected_laps])
            nrow = len(corridor_types)

            for row in range(nrow): # for each corridor...
                ids_all = np.where(self.i_corridors == corridor_types[row])
                ids = np.intersect1d(ids_all, selected_laps)
                if (len(ids) > 2):
                    n_correct = 0
                    n_valid = 0
                    for lap in ids:
                        n_correct = n_correct + self.ImLaps[lap].correct
                        n_valid = n_valid + 1
                    P_correct = np.round(nan_divide(float(n_correct), float(n_valid)),3)
                    self.Ps_correct[self.corridors[row]] = P_correct

    def calc_speed_and_lick_selectivity(self):
        selected_laps = np.arange(self.n_laps)
        if (self.n_laps > 0):
            corridor_types = np.unique(self.i_corridors[selected_laps])
            nrow = len(corridor_types)
            nbins = len(self.ImLaps[0].bincenters)

            avg_speeds_by_corridor = {}
            avg_licks_by_corridor = {}
            for row in range(nrow): # for each corridor...
                ids_all = np.where(self.i_corridors == corridor_types[row])
                ids = np.intersect1d(ids_all, selected_laps)

                if (len(ids) > 2):
                    ########################################
                    ## speed
                    speed_matrix = np.zeros((len(ids), nbins))
                    i_lap = 0
                    for lap in ids:
                        if (self.ImLaps[lap].mode == 1): # only use the lap if it was a valid lap
                            speed_matrix[i_lap,:] =  np.round(self.ImLaps[lap].ave_speed, 2)
                        else:
                            speed_matrix[i_lap,:] =  np.nan
                        i_lap = i_lap + 1

                    ########################################
                    ## lick
                    lick_matrix = np.zeros((len(ids), nbins))
                    i_lap = 0
                    for lap in ids:
                        if (self.ImLaps[lap].mode == 1): # only use the lap if it was a valid lap
                            lick_matrix[i_lap,:] =  np.round(self.ImLaps[lap].lick_rate, 2)
                        i_lap = i_lap + 1

                    # 5 bins long region right before reward zone
                    RZ_start_bin = np.round(self.corridor_list.corridors[self.corridors[row]].reward_zone_starts * self.N_pos_bins)
                    preRZ_ub = int(RZ_start_bin-1)
                    preRZ_lb = preRZ_ub - 5

                    # 5 bins long control region 1000 roxels before preRZ region
                    ctrl_ub = preRZ_lb - int(np.round(1000 / self.corridor_length_roxel * self.N_pos_bins))
                    ctrl_lb = ctrl_ub - 5

                    # within-corridor selectivities
                    avg_speed_preRZ = np.nanmean(speed_matrix[:, preRZ_lb:preRZ_ub], axis=1)
                    avg_speed_ctrl = np.nanmean(speed_matrix[:,ctrl_lb:ctrl_ub], axis=1)
                    speed_selectivity_laps = nan_divide(avg_speed_preRZ - avg_speed_ctrl,
                                                        avg_speed_preRZ + avg_speed_ctrl)
                    self.speed_selectivity_laps[self.corridors[row]] = speed_selectivity_laps

                    avg_lick_preRZ = np.nanmean(lick_matrix[:, preRZ_lb:preRZ_ub], axis=1)
                    avg_lick_ctrl = np.nanmean(lick_matrix[:, ctrl_lb:ctrl_ub], axis=1)
                    lick_selectivity_laps = nan_divide(avg_lick_preRZ - avg_lick_ctrl,
                                                        avg_lick_preRZ + avg_lick_ctrl)
                    self.lick_selectivity_laps[self.corridors[row]] = lick_selectivity_laps

                    # cross-corridor selectivities
                    current_corr_id = self.corridors[row]
                    if current_corr_id in [14, 15]:  # we ignore block and new env for now
                        RZ_corr14_start_bin = np.round(self.corridor_list.corridors[14].reward_zone_starts * self.N_pos_bins)
                        corr14_preRZ_ub = int(RZ_corr14_start_bin - 1)
                        corr14_preRZ_lb = int(RZ_corr14_start_bin - 5)

                        avg_speed_corr14_preRZ = np.nanmean(speed_matrix[:, corr14_preRZ_lb:corr14_preRZ_ub])
                        avg_speeds_by_corridor[current_corr_id] = avg_speed_corr14_preRZ
                        avg_lick_corr14_preRZ = np.nanmean(lick_matrix[:, corr14_preRZ_lb:corr14_preRZ_ub])
                        avg_licks_by_corridor[current_corr_id] = avg_lick_corr14_preRZ
            self.speed_selectivity_cross_corridor = nan_divide(avg_speeds_by_corridor[14] - avg_speeds_by_corridor[15],
                                                               avg_speeds_by_corridor[14] + avg_speeds_by_corridor[15])
            self.lick_selectivity_cross_corridor = nan_divide(avg_licks_by_corridor[14] - avg_licks_by_corridor[15],
                                                              avg_licks_by_corridor[14] + avg_licks_by_corridor[15])

    def plot_session(self, selected_laps=None, average=True, filename=None):
        ## plot the behavioral data during one session. 
            # - speed
            # - lick rate
        ## selected laps: numpy array indexing the laps to be included in the plot
        ## average: boolean. If True (default) session averages are plotted. If False, individual alps will be plotted as image plot.
        ## filename: name of a file to save the image in pdf format

        ## find the number of different corridors
        if (selected_laps is None):
            selected_laps = np.arange(self.n_laps)
            add_anticipatory_test = True
        else:
            add_anticipatory_test = False

        if (self.n_laps > 0):
            corridor_types = np.unique(self.i_corridors[selected_laps])
            nrow = len(corridor_types)
            nbins = len(self.ImLaps[0].bincenters)
            cmap = plt.cm.get_cmap('jet')   
            rowHeight = 2
            if (nrow > 4):
                rowHeight = 1.5
            # plt.figure(figsize=(5,2*nrow))

            if (average):
                reward_zone_color = 'green'
                fig, axs = plt.subplots(nrows=nrow, ncols=1, figsize=(8,rowHeight*nrow), squeeze=False)
            else:
                reward_zone_color = 'fuchsia'
                fig, axs = plt.subplots(nrows=nrow, ncols=1, figsize=(8,rowHeight*nrow*3), squeeze=False)

            speed_color = cmap(30)
            speed_color_trial = (speed_color[0], speed_color[1], speed_color[2], (0.05))

            lick_color = cmap(200)
            lick_color_trial = (lick_color[0], lick_color[1], lick_color[2], (0.05))

            for row in range(nrow): # for each corridor...
                # ax = plt.subplot(nrow, 1, row+1)
                ids_all = np.where(self.i_corridors == corridor_types[row])
                ids = np.intersect1d(ids_all, selected_laps)

                if (len(ids) > 2):
                    ########################################
                    ## speed
                    avespeed = np.zeros(nbins)
                    n_lap_bins = np.zeros(nbins) # number of laps in a given bin (data might be NAN for some laps)
                    n_laps = len(ids)
                    n_correct = 0
                    n_valid = 0
                    maxspeed = 10

                    speed_matrix = np.zeros((len(ids), nbins))

                    i_lap = 0
                    for lap in ids:
                        if (self.ImLaps[lap].mode == 1): # only use the lap if it was a valid lap
                            if (average):
                                axs[row,0].step(self.ImLaps[lap].bincenters, self.ImLaps[lap].ave_speed, where='mid', c=speed_color_trial)
                            speed_matrix[i_lap,:] =  np.round(self.ImLaps[lap].ave_speed, 2)
                            nans_lap = np.isnan(self.ImLaps[lap].ave_speed)
                            avespeed = nan_add(avespeed, self.ImLaps[lap].ave_speed)
                            n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                            if (max(self.ImLaps[lap].ave_speed) > maxspeed): maxspeed = max(self.ImLaps[lap].ave_speed)
                            n_correct = n_correct + self.ImLaps[lap].correct
                            n_valid = n_valid + 1
                        else:
                            speed_matrix[i_lap,:] =  np.nan
                        i_lap = i_lap + 1
                    maxspeed = min(maxspeed, 60)
                    P_correct = np.round(float(n_correct) / float(n_valid), 3)

                    if (average):
                        avespeed = nan_divide(avespeed, n_lap_bins, n_lap_bins > 0)
                        axs[row,0].step(self.ImLaps[lap].bincenters, avespeed, where='mid', c=speed_color)
                        axs[row,0].set_ylim([-1,1.2*maxspeed])
                    else:
                        im = axs[row,0].imshow(speed_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=maxspeed, extent=(0, self.corridor_length_roxel, 0, i_lap), origin='lower')
                        plt.colorbar(im, orientation='vertical',ax=axs[row, 0])

                    if (row == 0):
                        if self.sessionID:
                            plot_title = 'session:' + str(self.sessionID) + ': ' + str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)
                        else:
                            plot_title = str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)
                    else:
                        plot_title = str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)

                    ########################################
                    ## reward zones

                    if (self.ImLaps[lap].zones.shape[1] > 0):
                        bottom, top = axs[row,0].get_ylim()
                        left = np.round(self.ImLaps[lap].zones[0,0] * self.corridor_length_roxel, -1) - 4.5 # threshold of the position rounded to 10s - this is what LabView does
                        right = np.round(self.ImLaps[lap].zones[1,0] * self.corridor_length_roxel, -1) - 4.5

                        if (average):
                            polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color=reward_zone_color, alpha=0.15)
                            axs[row,0].add_patch(polygon)
                        else :
                            axs[row,0].vlines((left, right), ymin=bottom, ymax=top, colors=reward_zone_color, lw=3)
                        n_zones = np.shape(self.ImLaps[lap].zones)[1]
                        if (n_zones > 1):
                            for i in range(1, n_zones):
                                left = np.round(self.ImLaps[lap].zones[0,i] * self.corridor_length_roxel, -1) - 4.5 # threshold of the position rounded to 10s - this is what LabView does
                                right = np.round(self.ImLaps[lap].zones[1,i] * self.corridor_length_roxel, -1) - 4.5
                                if (average):
                                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color=reward_zone_color, alpha=0.15)
                                    axs[row,0].add_patch(polygon)
                                else :
                                    axs[row,0].vlines((left, right), ymin=bottom, ymax=top, colors=reward_zone_color, lw=3)
                        else: # we look for anticipatory licking tests
                            P_statement = ', anticipatory P value not tested'
                            for k in range(len(self.anticipatory)):
                                if (self.anticipatory[k].corridor == corridor_types[row]):
                                    P_statement = ', anticipatory P = ' + str(round(self.anticipatory[k].test[1],5))
                            if (add_anticipatory_test == True):
                                plot_title = plot_title + P_statement

                    axs[row,0].set_title(plot_title)

                    ########################################
                    ## lick
                    if (average):
                        ax2 = axs[row,0].twinx()
                    n_lap_bins = np.zeros(nbins) # number of laps in a given bin (data might be NAN for some laps)
                    maxrate = 10
                    avelick = np.zeros(nbins)

                    lick_matrix = np.zeros((len(ids), nbins))
                    i_lap = 0
                    i_sstage = 0
                    if (len(self.substage_change_laps) > 1): # we have substage switch
                        # the substage of the first lap: 
                        # which is the largast switch before the fist lap?
                        i_sstage = max(np.flatnonzero(self.substage_change_laps <= ids[0])) + 1


                    for lap in ids:
                        if (self.ImLaps[lap].mode == 1): # only use the lap if it was a valid lap
                            if (average): 
                                ax2.step(self.ImLaps[lap].bincenters, self.ImLaps[lap].lick_rate, where='mid', c=lick_color_trial, linewidth=1)
                            else:
                                if (len(self.ImLaps[lap].reward_times) > 0):
                                    axs[row,0].plot(self.ImLaps[lap].reward_position, np.ones(len(self.ImLaps[lap].reward_position)) * i_lap + 0.5, 'o', ms=4, color='deepskyblue')
                                if (len(self.ImLaps[lap].lick_times) > 0):
                                    axs[row,0].plot(self.ImLaps[lap].lick_position, np.ones(len(self.ImLaps[lap].lick_position)) * i_lap + 0.5, 'oC1', ms=1)
                                    if (len(self.substage_change_laps) > i_sstage):
                                        if (lap >= self.substage_change_laps[i_sstage]):
                                            axs[row,0].hlines(i_lap, xmin=0, xmax=self.corridor_length_roxel, colors='crimson', lw=3)
                                            i_sstage = i_sstage + 1
                            lick_matrix[i_lap,:] =  np.round(self.ImLaps[lap].lick_rate, 2)
                            nans_lap = np.isnan(self.ImLaps[lap].lick_rate)
                            avelick = nan_add(avelick, self.ImLaps[lap].lick_rate)
                            n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                            if (np.nanmax(self.ImLaps[lap].lick_rate) > maxrate): maxrate = np.nanmax(self.ImLaps[lap].lick_rate)
                        i_lap = i_lap + 1
                    maxrate = min(maxrate, 20)

                    avelick = nan_divide(avelick, n_lap_bins, n_lap_bins > 0)
                    if (average):
                        ax2.step(self.ImLaps[lap].bincenters, avelick, where='mid', c=lick_color)
                        ax2.set_ylim([-1,1.2*maxrate])


                    if (row==(nrow-1)):
                        if (average): 
                            axs[row,0].set_ylabel('speed (cm/s)', color=speed_color)
                            axs[row,0].tick_params(axis='y', labelcolor=speed_color)
                            ax2.set_ylabel('lick rate (lick/s)', color=lick_color)
                            ax2.tick_params(axis='y', labelcolor=lick_color)
                        else:
                            axs[row,0].set_ylabel('laps')
                        axs[row,0].set_xlabel('position (roxel)')
                    else:
                        axs[row,0].set_xticklabels([])
                        if (average): 
                            axs[row,0].tick_params(axis='y', labelcolor=speed_color)
                            ax2.tick_params(axis='y', labelcolor=lick_color)
                        else:
                            axs[row,0].set_ylabel('laps')

            if (filename is None):
                plt.show(block=False)
            else:
                plt.savefig(filename, format='pdf')
                plt.close()

        else:
            fig = plt.figure(figsize=(8,3))
            # plt.scatter([-4, -3, -2], [2,3,4])
            plt.title('No data to show')
            plt.show(block=False)

    def plot_speed_and_lick_selectivity(self, filename):
        n_corridors = len(self.speed_selectivity_laps)
        fig, axs = plt.subplots(2, n_corridors, sharey=True)
        for i_cor, cor in enumerate(self.speed_selectivity_laps):
            axs[0,i_cor].hist(self.speed_selectivity_laps[cor], color="blue")
            axs[0,i_cor].set_title(f"corridor {cor}")
            axs[0,i_cor].set_xlim([-1,1])
            axs[0,i_cor].set_xlabel("speed selectivity")
            axs[0,i_cor].set_ylabel("laps")
            axs[0,i_cor].axvline(0, color="black", linestyle="--")
            axs[0,i_cor].axvline(np.nanmean(self.speed_selectivity_laps[cor]), color="red")

            axs[1,i_cor].hist(self.lick_selectivity_laps[cor], color="orange")
            axs[1,i_cor].set_title(f"corridor {cor}")
            axs[1,i_cor].set_xlim([-1, 1])
            axs[1,i_cor].set_xlabel("lick selectivity")
            axs[1,i_cor].set_ylabel("laps")
            axs[1,i_cor].axvline(0, color="black", linestyle="--")
            axs[1,i_cor].axvline(np.nanmean(self.lick_selectivity_laps[cor]), color="red")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_masks(self, cellids, cell_property=np.array([np.nan]), title_string=''):
        if (version_info.major == 2):
            print ('Mask data can not be loaded in python2. Switch to python 3 and try again.')
            return
        elif (version_info.major == 3):
            stat = np.load(self.stat_string, allow_pickle=True)
            ops = np.load(self.ops_string, allow_pickle=True).item()
            print(type(cell_property))
            if (np.isnan(cell_property[0])):
                flag=False
            else:
                flag=True

            #initialise picture
            im = np.ones((ops['Ly'], ops['Lx']))
            im_nonover = np.ones((ops['Ly'], ops['Lx']))
            im[:] = np.nan
            im_nonover[:] = np.nan
            fig, [left, right]=plt.subplots(1,2)
            
            #select intensities if specifies
            if flag==True:
                intens=cell_property[cellids]
            
            #create image
            for i in range(np.size(cellids)):
                cellid=cellids[i]
                n = self.neuron_index[cellid]# n is the suite2p ID for the given cell
                ypix = stat[n]['ypix']#[~stat[n]['overlap']]
                xpix = stat[n]['xpix']#[~stat[n]['overlap']]
                ypix_nonover = stat[n]['ypix'][~stat[n]['overlap']]
                xpix_nonover = stat[n]['xpix'][~stat[n]['overlap']]
                if flag == False:
                    im[ypix,xpix] = n+1
                    im_nonover[ypix_nonover,xpix_nonover] = n+1
                else:
                    im[ypix,xpix] = intens[i]
                    im_nonover[ypix_nonover,xpix_nonover] = intens[i]
            #plotting
            full_image=left.imshow(im)
            non_overlap_image = right.imshow(im_nonover)
            if flag==True:
                title_left = 'Full ROI' +'\n' +  title_string
                title_right = 'Nonoverlapping parts' +'\n' +  title_string 
            else:
                title_left = 'Full ROI' + '\n' + ' colors = suite2p index' +'\n' +  title_string 
                title_right = 'Nonoverlapping parts' + '\n' + ' colors = suite2p index' +'\n' +  title_string 
            left.set_title(title_left)
            right.set_title(title_right)
            plt.colorbar(full_image, orientation='horizontal',ax=left)
            plt.colorbar(non_overlap_image, orientation='horizontal',ax=right)

            plt.show(block=False)


    def save_cell_stats(self, path=""):
        if self.sessionID is None:
            animalID = None
        elif "_" not in self.sessionID:
            animalID = None
        else:
            animalID, _, _ = self.sessionID.partition("_")

        N_active_cells = len(self.active_cells)
        N_tuned_cells = len(functools.reduce(np.union1d, self.tuned_cells))  # handles multiple corridors by taking union of cellids

        cell_stats = {
            "animalID": animalID,
            "sessionID": self.sessionID,
            "total cells": self.N_cells,
            "active cells": N_active_cells,
            "tuned cells": N_tuned_cells
        }
        df = pd.DataFrame.from_dict([cell_stats])
        if path:
            df.to_pickle(path)
        #with open("C:/home/makaralab/tuned.csv", "a") as tuned_file:
        #    tuned_file.writelines([self.sessionID, ",", str(N_active_cells), ",", str(N_tuned_cells), ",", str(len(self.tuned_cells[0])), ",", str(len(self.tuned_cells[1])), "\n"])
        return df

    def save_data(self, save_properties=True, save_ratemaps=True, save_laptime=True, save_lick_speed_stats=True, save_place_code_stats = True, plot=False):
        # Saves the primary data into a folder in csv format.
        # separate file is created for each corridor and lap.
        # 
        # save_properties: True or False. If True, the cell properties are saved. 
        # save_ratemaps: True or False. If True, the ratemaps are saved.
        # save_laptime: True or False. If True, the raw data is saved for each lap.
        # save_lick_speed_stats: True or False, prepares an array that contains all the brhavioral measures of the session. Each row is a separate lap. 
        #           First 5 columns are: 1: lap number, 2: corridor ID, 3: correct, 4: reward, 5: imaging available. 
        #           The remaining columns are speed and lick rate in the spatial bins
        # save_place_code_stats: True or False, 


        data_folder = self.suite2p_folder + self.data_folder
        if (self.elfiz == True):
            data_folder = self.suite2p_folder + self.data_folder + '_' + self.imaging_logfile_name

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        N_corridors = len(self.corridors)

        if (self.elfiz):
            save_laptime = False

        # saving the cell properties - all cells included, a separate file is given for each corridor
        # each line is a different cell
        if (save_properties):
            # cell_number   rate1   rate2   reliability1    reliability2    Fano1   Fano2   Skaggs1     Skaggs2     activeLaps1    activLaps2  activeLaps_dF1   activeLaps_dF2  specificity1    specificity2
            colnames = ['#cell_number', 'rate', 'reliability', 'Fano_factor', 'Skaggs_info', 'active_laps', 'active_laps_dF', 'specificity']
            for i_cor in np.arange(N_corridors):
                allprops = np.array([np.arange(self.N_cells), self.cell_rates[i_cor], self.cell_reliability[i_cor], self.cell_Fano_factor[i_cor], self.cell_skaggs[i_cor], self.cell_activelaps[i_cor], self.cell_activelaps_df[i_cor], self.cell_tuning_specificity[i_cor]])
                corridor_file = 'cell_properties_corridor_' + str(self.corridors[i_cor]) + '_N' + str(self.N_cells) + '.csv'
                self.write_params(corridor_file)
                filename = data_folder + '/' + corridor_file
                with open(filename, mode='a', newline='') as property_file:
                    file_writer = csv.writer(property_file, **csv_kwargs)
                    file_writer.writerow(colnames)
                    for i_col in np.arange(allprops.shape[1]):
                        file_writer.writerow(np.round(allprops[:,i_col], 4))
                print('cell properties for corridor', self.corridors[i_cor], ' saved into file: ', filename)

        # saving the ratemaps - all cells included, separate matrix for each of the corridors
        # each line is a different cell
        if (save_ratemaps):
            for i_cor in np.arange(self.N_corridors):
                ratemap_file = 'ratemaps_corridor_' + str(self.corridors[i_cor]) + '_N' + str(self.N_cells) + '.csv'
                self.write_params(ratemap_file)
                filename = data_folder + '/' + ratemap_file
                with open(filename, mode='a', newline='') as rate_file:
                    file_writer = csv.writer(rate_file, **csv_kwargs)
                    file_writer.writerow(('rows: firing rate of the ' + str(self.N_cells) + ' cells;',   'columns: position bins'))
                    for i_col in np.arange(self.N_cells):
                        file_writer.writerow(np.round(self.ratemaps[i_cor][:,i_col], 4))
                print('ratemap for corridor', self.corridors[i_cor], ' saved into file: ', filename)


        # saving behavioral and imaging data for all laps
        # for each frame we save time, position, speed, lick, reward and spikes for each cells
        if (save_laptime):
            dt_im = self.ImLaps[1].dt_imaging
            for i_lap in self.i_Laps_ImData:
                lapfile = 'lapdata_lap_' + str(i_lap) + '_N' + str(self.N_cells) + '.csv'
                self.write_params(lapfile)
                filename = data_folder + '/' + lapfile
                nframes = len(self.ImLaps[i_lap].frames_pos)
                frames_lick = np.zeros(nframes)
                for tt in self.ImLaps[i_lap].lick_times:
                    if ((tt <= max(self.ImLaps[i_lap].frames_time) + dt_im) & (tt >= min(self.ImLaps[i_lap].frames_time) - dt_im)):
                        i_frame = np.argmin((tt - self.ImLaps[i_lap].frames_time)**2)
                        frames_lick[i_frame] = frames_lick[i_frame] + 1

                frames_reward = np.zeros(nframes)
                for tt in self.ImLaps[i_lap].reward_times:
                    if ((tt <= max(self.ImLaps[i_lap].frames_time) + dt_im) & (tt >= min(self.ImLaps[i_lap].frames_time) - dt_im)):
                        i_frame = np.argmin((tt - self.ImLaps[i_lap].frames_time)**2)
                        frames_reward[i_frame] = frames_reward[i_frame] + 1


                lapdata = np.vstack((self.ImLaps[i_lap].frames_time, self.ImLaps[i_lap].frames_pos, self.ImLaps[i_lap].frames_speed, frames_lick, frames_reward, self.ImLaps[i_lap].frames_spikes))
                with open(filename, mode='a', newline='') as lap_file:
                    file_writer = csv.writer(lap_file, **csv_kwargs)
                    file_writer.writerow(('rows: time, position, speed, lick, reward and spikes of the ' + str(self.N_cells) + ' cells;', 'columns: time bins'))
                    for i_row in np.arange(lapdata.shape[0]):
                        file_writer.writerow(np.round(lapdata[i_row,:], 4))
            print('lapdata saved into file: ' + filename)
            
            
        if (save_lick_speed_stats):
        # save_lick_speed_stats: True or False, prepares an array that contains all the behavioral measures of the session. Each row is a separate lap. 
        #           First 5 columns are: 0: lap number, 1: corridor ID, 2: correct, 3: reward, 4: imaging available. 
        #           The remaining columns are speed and lick rate in the spatial bins
        

            self.behavior_stats = np.zeros((self.n_laps, self.N_pos_bins * 2 + 5))
            colnames = ['#lap number', 'corridor', 'correct', 'reward', 'imaging', 'lick 0 - ' + str(self.N_pos_bins), 'speed 0 - ' + str(self.N_pos_bins)]
            
            for i_lap in np.arange(self.n_laps):
                self.behavior_stats[i_lap,0:5] = [i_lap, self.ImLaps[i_lap].corridor, self.ImLaps[i_lap].correct, len(self.ImLaps[i_lap].reward_position), self.ImLaps[i_lap].imaging_data]
                b = 5
                c = self.N_pos_bins + 5
                d = 2 * self.N_pos_bins + 5
                self.behavior_stats[i_lap, b:c] = self.ImLaps[i_lap].lick_rate
                self.behavior_stats[i_lap, c:d] = self.ImLaps[i_lap].ave_speed

            behavior_file = 'behavior_data.csv'
            filename = data_folder + '/' + behavior_file
            with open(filename, mode='w', newline='') as behav_file:
                file_writer = csv.writer(behav_file, **csv_kwargs)
                file_writer.writerow(colnames)
                for i_row in np.arange(self.n_laps):
                    file_writer.writerow(np.round(self.behavior_stats[i_row,:], 4))
            print('behavioral data saved into file: ', filename)

            # for i_corrid in np.arange(self.corridors.size):

            #     ids_all = np.nonzero(self.i_corridors == self.corridors[i_corrid])[0]
            #     i_laps = np.intersect1d(ids_all, self.i_Laps_ImData)

            #     last_prezone_bin = int(np.floor(self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_ends * self.N_pos_bins ))-1
            #     n = i_laps.size
                
            #     lickrate_matrix = np.zeros((n,self.N_pos_bins))
            #     speed_matrix = np.zeros((n,self.N_pos_bins))
            #     for i in range(i_laps.size):
            #         lickrate_matrix[i,:] = self.ImLaps[i_laps[i]].lick_rate
            #         speed_matrix[i,:]  = self.ImLaps[i_laps[i]].ave_speed
            #     ave_speed = np.nanmean(speed_matrix, axis=0)
            #     ave_lick = np.nanmean(lickrate_matrix, axis=0)

            #     self.reference_lickrate.append(np.mean(ave_lick[8:10]))
            #     self.reference_speed.append(np.mean(ave_speed[8:10]))
            #     self.prezone_lickrate.append(np.mean(ave_lick[(last_prezone_bin-3):last_prezone_bin]))
            #     self.prezone_speed.append(np.mean(ave_speed[(last_prezone_bin-3):last_prezone_bin]))

            if plot:
                self.plot_session(selected_laps=self.i_Laps_ImData)

            print('lickrate- and speed difference calculated')
            #TODO save to file
            
        if (save_place_code_stats):
            # outputs:
            self.PC_per_bin = []
            for i_corrid in np.arange(self.corridors.size):
                cellids = self.tuned_cells[i_corrid]
                # cellids = self.accepted_PCs[i]
                
                #calculate
                N_pos_bins = self.ratemaps[i_corrid].shape[0]
                PC_in_bin = np.zeros(N_pos_bins)
                PC_in_smooth = np.zeros(N_pos_bins)
                for i_tuned_cell in range(cellids.size):
                    i_cell = cellids[i_tuned_cell]
                    index = np.argmax(self.ratemaps[i_corrid][:,i_cell])
                    PC_in_bin[index] += 1                    

                #convert to percentage, smooth
                PC_in_bin = PC_in_bin/cellids.size*100
                PC_in_bin_smooth_p2=np.convolve(PC_in_bin, np.ones(3)/3)
                PC_in_bin_smooth = PC_in_bin_smooth_p2[1:(N_pos_bins+1)]
                PC_in_bin_smooth[0] = PC_in_bin_smooth[0] + PC_in_bin_smooth_p2[0]
                PC_in_bin_smooth[-1] = PC_in_bin_smooth[-1] + PC_in_bin_smooth_p2[-1]
                bins=np.arange(N_pos_bins)
                self.PC_per_bin.append(PC_in_bin_smooth)
                
                #plotting...
            print('Place Cells per spatial bin calculated')

            # here we calculate cross corr between first two corridor to make it easy to automate
            try:
                cellids = np.unique(np.concatenate((self.tuned_cells[0],self.tuned_cells[1])))
            except AttributeError:
                cellids = np.arange(0, self.N_cells)
                print('tuned_cells attribute does not exist - all cells used for pop vector corr! You probably want to run shuffling first!')
            a=self.ratemaps[0][:,cellids]
            b=self.ratemaps[1][:,cellids]
            
            popp_full = np.corrcoef(a,b)
            self.ratemap_corr = popp_full[:self.N_pos_bins, self.N_pos_bins:]
            
            # diag = np.diagonal(popp)
            #for plotting
            if plot:

                corr1label = 'corridor ' + str(self.corridors[0])
                corr2label = 'corridor ' + str(self.corridors[1])

                fig, ax1 = plt.subplots()

                ax1.set_xlabel('spatial bin')
                ax1.set_ylabel('percent of tuned cells', color='teal')
                ax1.plot(bins,self.PC_per_bin[0],color='teal', label=corr1label)
                ax1.plot(bins,self.PC_per_bin[1],color='darkviolet', label=corr2label)
                ax1.tick_params(axis='y', labelcolor='teal')

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('correlation', color='firebrick')  # we already handled the x-label with ax1
                ax2.plot(bins,np.diagonal(self.ratemap_corr),color='firebrick', label='ratemap correlation')
                ax2.tick_params(axis='y', labelcolor='firebrick')

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()

                print('pop vector correlation calculated')

            ## saving these vectors into file
            place_code_stat_file = 'place_code_stat' + '_N' + str(len(cellids)) + '.csv'
            self.write_params(place_code_stat_file)
            filename = data_folder + '/' + place_code_stat_file
            with open(filename, mode='a', newline='') as stat_file:
                file_writer = csv.writer(stat_file, **csv_kwargs)
                file_writer.writerow(('first row:', 'percent of tuned cells per position bins in corridor ' + str(self.corridors[0])))
                file_writer.writerow(('second row', 'percent of tuned cells per position bins in corridor ' + str(self.corridors[1])))
                file_writer.writerow(('third row', 'correlation between the ratemaps of all tuned cells across the two corridors', str(self.corridors[0]) + ' and ' + str(self.corridors[1])))
                file_writer.writerow(np.round(self.PC_per_bin[0], 4))
                file_writer.writerow(np.round(self.PC_per_bin[1], 4))
                file_writer.writerow(np.round(np.diagonal(self.ratemap_corr), 4))
            print('place code statistics saved into file: ', filename)
   
    def calc_rate(self, i_laps, cellids=None):
        #calculate ratemaps for the given laps
        if (cellids is None):
            cellids = np.arange(self.N_cells)
        N_cells = len(cellids)
        ratemap = np.zeros((N_cells,self.N_pos_bins))
        
        total_time = self.activity_tensor_time[:,i_laps]
        for i in range(N_cells):
            total_spikes = self.activity_tensor[:,i,i_laps]
            rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)
            av_rate = np.nanmean(rate_matrix, axis=1)
            ratemap[i,:] = av_rate
            
        return np.transpose(ratemap)

    
    def calc_even_odd_rates(self):
        #calculate ratemaps for even and ott laps for every corridor with enough laps
        if self.even_odd_rate_calculated == False:
            self.even_odd_rate_calculated = True
            self.ratemaps_even = []
            self.ratemaps_odd = []
            for i in range(self.corridors.size):
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == self.corridors[i])[0]
                i_laps_even = i_laps[0::2]
                i_laps_odd = i_laps[1::2]
                
                ratemap_even = self.calc_rate(i_laps_even)
                ratemap_odd = self.calc_rate(i_laps_odd)
                
                self.ratemaps_even.append(ratemap_even)
                self.ratemaps_odd.append(ratemap_odd)
                
            print('even/odd ratemaps calculated')
        else:
            print('even/odd ratemaps already calculated')
        
    
    def calc_start_end_rates(self, n_used = -1):
        #calculate ratemaps for laps t the begining and at the end for every corridor with enough laps
        # with the n_used parameter the last n_used and first n_used laps are used 
        if type(n_used) != int:
            print('Invaid n_used parameter - give an integer')
            return
        else:
            self.start_end_rate_calculated = True
            self.ratemaps_start = []
            self.ratemaps_end = []
            
            for i in range(self.corridors.size):
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == self.corridors[i])[0]
                # print(i_laps.size)
                if n_used == -1:
                    i_laps_start = i_laps[0:int(i_laps.size/2)]
                    i_laps_end = i_laps[int(i_laps.size/2):]
                else:
                    if (n_used > i_laps.size):
                        i_laps_start = i_laps[0:int(i_laps.size/2)]
                        i_laps_end = i_laps[int(i_laps.size/2):]
                        print('Specified n_used is too large, using half-half the laps in corridor ', self.corridors[i])
                    else:
                        i_laps_start = i_laps[0:n_used]
                        i_laps_end = i_laps[i_laps.size-n_used:]
                ratemap_start = self.calc_rate(i_laps_start)
                ratemap_end = self.calc_rate(i_laps_end)
                
                self.ratemaps_start.append(ratemap_start)
                self.ratemaps_end.append(ratemap_end)
                
            print('start/end ratemaps calculated')
            
    def calc_previous_based_rates(self, corr_a, corr_b):
        #calculate ratemaps based on the id of current and previous maze
        #warning: this code is developed for two-corridor task - for more corridors it may give unexpected results
        if (corr_a in self.corridors) and (corr_b in self.corridors):
            i_stay_a = []
            i_stay_b = []
            i_changeto_a = []
            i_changeto_b = []
            
            imaged_laps = self.i_corridors[self.i_Laps_ImData]
            for i in np.arange(1,imaged_laps.size):
                if imaged_laps[i] == corr_a:
                    if imaged_laps[i-1] == corr_a:
                        i_stay_a.append(i)
                    else:
                        #TODO
                        i_changeto_a.append(i)
                if imaged_laps[i] == corr_b:
                    if imaged_laps[i-1] == corr_b:
                        i_stay_b.append(i)
                    else:
                        i_changeto_b.append(i)
            self.ratemap_a_a = self.calc_rate(i_stay_a)
            self.ratemap_b_b = self.calc_rate(i_stay_b)
            self.ratemap_a_b = self.calc_rate(i_changeto_b)
            self.ratemap_b_a = self.calc_rate(i_changeto_a)
            
            print('previous-based ratemaps calculated - !Previous overwritten!')

    def show_crosscorr(self, ratemap1, ratemap2, cellids=None, ratemap1_annot='map 1', ratemap2_annot='map 2', main_title='Cross correlation', return_matrix=False, plot_ccm=True):
        #plot cross-correlation matrix between two ratemaps, only for specified cellids
        if (cellids is None):
            cellids = np.arange(self.N_cells)

        popp_full = np.corrcoef(ratemap1[:,cellids], ratemap2[:,cellids])
        popp = popp_full[:self.N_pos_bins, self.N_pos_bins:]
        
        if (plot_ccm == True):
            fig, ax = plt.subplots()
            im = ax.imshow(popp, cmap = 'seismic', vmin = -1, vmax = 1, origin='lower')
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", linewidth = '0.5', c='k')
            
            plt.colorbar(im)
            plt.ylabel(ratemap1_annot)
            plt.xlabel(ratemap2_annot)
            plt.title(main_title)
            plt.show()

        if (return_matrix == True):
            return popp_full
        else :
            return
        
    def show_autocorr(self, ratemap, cellids=None, title='autocorrelation'):
        #show autucorrelation matrix for a given ratemap, cellids
        if (cellids is None):
            cellids = np.arange(self.N_cells)

        popp = np.corrcoef(ratemap[:,cellids])
        
        fig, ax = plt.subplots()
        im = plt.imshow(popp, cmap = 'seismic', vmin = -1, vmax = 1, origin='lower')
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", linewidth = '0.5', c='k')

        plt.colorbar(im)
        plt.title(title)
        plt.show()
        
    def lap_decode(self, cellids, ratemaps=None, labels=None, title=''):
        ## D1.lap_decode(cellids, D1.ratemaps, D1.corridors, '')
        add_true_corridor_ids = False
        if (ratemaps is None):
            ratemaps = self.ratemaps
            labels = self.corridors
            add_true_corridor_ids = True

        results = np.zeros((len(ratemaps), self.i_Laps_ImData.size))
        ratemap = np.zeros((self.N_pos_bins, self.N_cells))
        
        speed = []
                        
        for i in range(self.i_Laps_ImData.size):
            speed.append(np.nanmean(self.ImLaps[self.i_Laps_ImData[i]].ave_speed))
            
            for j in range(self.N_cells):
                total_spikes = self.activity_tensor[:,j,i]
                total_time = self.activity_tensor_time[:,i]
                rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)
#                av_rate = np.nanmean(rate_matrix, axis=1)
                ratemap[:,j] = rate_matrix

            for k in range(len(ratemaps)):
                results[k, i] = np.mean(np.diagonal(np.corrcoef(ratemap[:,cellids], ratemaps[k][:,cellids])[0:self.N_pos_bins,self.N_pos_bins:]))
                # print(ratemap.shape, ratemaps[k].shape)
                # print(results[k ,i])
                
        fig, ax = plt.subplots()
        x=np.arange(0, self.i_Laps_ImData.size)
        for  k in range(len(ratemaps)):
            ax.scatter(x, results[k,:], label=labels[k])
        if (add_true_corridor_ids):
            for k in range(self.i_Laps_ImData.size):
                i_corridor = np.flatnonzero(self.corridors == self.i_corridors[self.i_Laps_ImData[k]])
                ax.scatter(x[k], results[i_corridor,k], facecolor='w', s=5)

        substage_change_laps = np.array(self.substage_change_laps)
        i_substage_change_ImData = (substage_change_laps > np.min(self.i_Laps_ImData)) & (substage_change_laps < np.max(self.i_Laps_ImData))
        if (sum(i_substage_change_ImData) > 0):
            i_lap_stage_change = substage_change_laps[i_substage_change_ImData]
            ii_lap_stage_change = np.flatnonzero(self.i_Laps_ImData == i_lap_stage_change) - .5

            ax.vlines(ii_lap_stage_change, 0, 1, linewidth=2)

        ax2=ax.twinx()
        ax2.plot(speed, label='average speed', linewidth= 0.5)

        ax.legend()
        ax2.legend()
        ax2.set_ylabel('speed (cm/s)')
        ax.set_xlabel('lap')
        ax.set_ylabel('correlation')

        plt.title(title)
        plt.show()

        return results
        
    def lap_correlate(self, cellids, filename=None, corridors=None, normalize_rates=False, add_switch_ordered=False):
        lap2lap_corr = np.zeros((self.i_Laps_ImData.size, self.i_Laps_ImData.size))
        ratemaps = np.zeros((self.i_Laps_ImData.size, self.N_pos_bins, self.N_cells))
        
        for j in range(self.N_cells):
            for i in range(self.i_Laps_ImData.size):
                total_spikes = self.activity_tensor[:,j,i]
                total_time = self.activity_tensor_time[:,i]
                rate_vector = nan_divide(total_spikes, total_time, where=total_time > 0.025)
                ratemaps[i,:,j] = rate_vector
            ## as in Low et al., 2021, we clip the max rate to the 99th percentile
            clap_rate_j = np.nanquantile(ratemaps[:,:,j], 0.99) # I suggest to use 99% instead of 90% as we do Ca imaging and not neurpixels, and these are place cells not grid cells
            ratemaps[:,:,j][np.where(ratemaps[:,:,j] > clap_rate_j)] = clap_rate_j
            if (normalize_rates):
                ratemaps[:,:,j] = (ratemaps[:,:,j] - np.nanmin(ratemaps[:,:,j])) / (np.nanmax(ratemaps[:,:,j]) - np.nanmin(ratemaps[:,:,j]))
    

        for i in np.arange(self.i_Laps_ImData.size):
            for j in np.arange(i):
                ri = ratemaps[i,:,cellids].reshape(len(cellids)* self.N_pos_bins)
                rj = ratemaps[j,:,cellids].reshape(len(cellids)* self.N_pos_bins)
                rr12 = np.corrcoef(ri, rj)[0,1]
                lap2lap_corr[i,j] = rr12
                lap2lap_corr[j,i] = rr12

                # lap2lap_corr[i,j] = np.nanmean(Mcorrcoef(np.transpose(ratemaps[i,:,cellids]), np.transpose(ratemaps[j,:,cellids])))
                # lap2lap_corr[i,j] = np.nanmean(np.diagonal(np.corrcoef(np.transpose(ratemaps[i,:,cellids]), np.transpose(ratemaps[j,:,cellids]))[0:self.N_pos_bins,self.N_pos_bins:]))
                # if np.isnan(lap2lap_corr[i,j]) == True:
                #     print(i,j)
    
        fig, axs = plt.subplots(1,2)
        im0 = axs[0].imshow(lap2lap_corr, cmap = 'seismic', vmin = -1, vmax = 1, origin='lower')
        plt.colorbar(im0, orientation='horizontal',ax=axs[0])
        
        #############################################################################
        ## indicate the substage change lap with vertical and horizontal lines

        substage_change_laps = np.array(self.substage_change_laps)
        i_substage_change_ImData = (substage_change_laps > np.min(self.i_Laps_ImData)) & (substage_change_laps < np.max(self.i_Laps_ImData))
        if (sum(i_substage_change_ImData) > 0):
            i_lap_stage_change = substage_change_laps[i_substage_change_ImData]
            ii_lap_stage_change = np.flatnonzero(self.i_Laps_ImData == i_lap_stage_change) - .5

            axs[0].vlines(ii_lap_stage_change, 0, len(self.i_Laps_ImData)-1, linewidth=.5)
            axs[0].hlines(ii_lap_stage_change, 0, len(self.i_Laps_ImData)-1, linewidth=.5)


        #############################################################################
        ## reorder the laps to have the laps in the same corridors next to each other

        if (corridors is None):
            corridors = self.corridors
        else :
            if (len(np.intersect1d(corridors, self.corridors)) < len(corridors)):
                print('Error: some corridors given are not used in this session.')
                corridors = np.intersect1d(corridors, self.corridors)
        
        ###################################################
        ## calculating the number of laps in each corridor and their ordering
        N_laps_corr = np.zeros(len(corridors))
        i_laps = np.array((1)).reshape(1,)
        i = 0
        N_corridors = len(corridors)
        for i_corrid in corridors:
            i_laps_corridor = np.flatnonzero(self.i_corridors[self.i_Laps_ImData] == i_corrid)
            i_laps = np.concatenate((i_laps, i_laps_corridor))
            N_laps_corr[i] = len(i_laps_corridor)
            i = i + 1
        order = i_laps[1:]
        i_corr_bounds = np.cumsum(N_laps_corr)-0.5
        i_corr_mids = N_laps_corr / 2 + np.concatenate(([0], i_corr_bounds[0:(N_corridors-1)]))

        lap2lap_corr_ordered = lap2lap_corr[order, :]
        lap2lap_corr_ordered = lap2lap_corr_ordered[:,order]
        # print(results2.shape)
        
        im1 = axs[1].imshow(lap2lap_corr_ordered, cmap = 'seismic', vmin = -1, vmax = 1, origin='lower')
        plt.colorbar(im1, orientation='horizontal',ax=axs[1])

        axs[1].vlines(i_corr_bounds, 0, len(self.i_Laps_ImData)-1, linewidth=.5, colors='g')
        axs[1].hlines(i_corr_bounds, 0, len(self.i_Laps_ImData)-1, linewidth=.5, colors='g')
        for i_cor in np.arange(N_corridors):
            axs[1].text(i_corr_mids[i_cor], -20, 'cor ' + str(corridors[i_cor]), fontsize=10, ha='center', weight='600')

        if ((sum(i_substage_change_ImData) > 0) & add_switch_ordered):
            axs[1].vlines(ii_lap_stage_change, 0, len(self.i_Laps_ImData)-1, linewidth=.5)
            axs[1].hlines(ii_lap_stage_change, 0, len(self.i_Laps_ImData)-1, linewidth=.5)

        if (filename is None):
            plt.show(block=False)
        else:
            plt.savefig(filename, format='pdf')
            plt.close()
        
        # cov = np.copy(results)
        # eigenvalues, eigenvectors = np.linalg.eig(cov)
        # plt.figure()
        # plt.plot(eigenvalues)
        
        # sortindex = np.argsort(eigenvalues*-1)
        # eigenvalues_sorted = eigenvalues[sortindex]
        # print(np.sum(eigenvalues_sorted[0:3])/np.sum(eigenvalues_sorted))
        
        # plt.plot(eigenvalues_sorted)
        # plt.show()
        
        # plt.figure()
        # plt.plot(eigenvectors[:,0:4])
        # plt.show()
        
        # print(ratemaps.shape, eigenvectors.shape)
        # self.eig1_ratemaps = np.copy(ratemaps)
        # self.eig2_ratemaps = np.copy(ratemaps)
        # for i in range(self.i_Laps_ImData.size):
        #     self.eig1_ratemaps[i,:,:] = ratemaps[i,:,:]*eigenvectors[i,0]
        #     self.eig2_ratemaps[i,:,:] = ratemaps[i,:,:]*eigenvectors[i,1]
        # self.eig1_ratemaps=np.nanmean(self.eig1_ratemaps, axis=0)
        # self.eig2_ratemaps=np.nanmean(self.eig2_ratemaps, axis=0)        
        # print(self.eig1_ratemaps.shape)


class Lap_ImData:
    'common base class for individual laps'

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time, corridor_list, lap_frames_events, frame_period, printout=False, speed_threshold=5, elfiz=False, verbous=0, multiplane=False, next_grey_lap_duration=None):
        if (verbous > 0):
            print('ImData initialised')
        # begin_time = datetime.now()

        self.name = name
        self.lap = lap
        self.multiplane = multiplane

        self.correct = False
        self.raw_time = laptime
        self.raw_position = position
        self.lick_times = lick_times
        self.reward_times = reward_times
        self.corridor = corridor # the ID of the corridor in the given stage; This indexes the corridors in the list called self.corridor_list.corridors
        self.corridor_list = corridor_list # a list containing the corridor properties in the given stage
        self.mode = mode # 1 if all elements are recorded in 'Go' mode
        self.actions = actions
        self.elfiz=elfiz

        self.speed_threshold = speed_threshold ## cm / s 106 cm - 3500 roxels; roxel/s * 106.5/3500 = cm/s
        self.corridor_length_roxel = (self.corridor_list.corridors[self.corridor].length - 1024.0) / (7168.0 - 1024.0) * 3500
        self.N_pos_bins = int(np.round(self.corridor_length_roxel / 70))
        self.speed_factor = 106.5 / 3500 ## constant to convert distance from pixel to cm
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor # cm

        self.last_zone_start = max(self.corridor_list.corridors[self.corridor].reward_zone_starts)
        self.last_zone_end = max(self.corridor_list.corridors[self.corridor].reward_zone_ends)
        self.zones = np.vstack([np.array(self.corridor_list.corridors[self.corridor].reward_zone_starts), np.array(self.corridor_list.corridors[self.corridor].reward_zone_ends)])        
        self.n_zones = np.shape(self.zones)[1]
        self.preZoneRate = [None, None] # only if 1 lick zone; Compare the 210 roxels just before the zone with the preceeding 210 


        self.frame_period = frame_period
        self.frames_dF_F = lap_frames_dF_F
        self.frames_spikes = lap_frames_spikes
        self.frames_pos = lap_frames_pos
        self.frames_time = lap_frames_time
        self.frames_events = lap_frames_events

        self.n_cells = 1 # we still create the same np arrays even if there are no cells
        self.bincenters = np.arange(0, self.corridor_length_roxel, 70) + 70 / 2.0

        if (verbous > 0):
            print('ImData established')

        ##################################################################################
        ## lick position and reward position
        ##################################################################################

        F = interp1d(self.raw_time,self.raw_position)
        self.lick_position = F(self.lick_times)
        self.reward_position = F(self.reward_times)

        # correct: if rewarded
        if (len(self.reward_times) > 0):
            self.correct = True
        # correct: if no licking in the zone
        if (len(self.lick_times) > 0):
            lick_in_zone = np.nonzero((self.lick_position > self.last_zone_start * self.corridor_length_roxel) & (self.lick_position <= self.last_zone_end * self.corridor_length_roxel + 1))[0]
        else:
            lick_in_zone = np.array([])
        if (self.corridor_list.corridors[self.corridor].reward == 'Left'):
            if (len(lick_in_zone) == 0):
                self.correct = True
        else :
            if ((len(lick_in_zone) == 0) & self.correct):
                print ('Warning: rewarded lap with no lick in zone! lap number:' + str(self.lap))

        if (next_grey_lap_duration is not None):
            self.grey_threshold = 0.7 # max duration of grey zone after correct
            if (self.correct):
                if (next_grey_lap_duration > self.grey_threshold):
                    raise ValueError('Error: too long grey zone duration after correct trial:', next_grey_lap_duration)                    
            else:
                if (next_grey_lap_duration < self.grey_threshold):
                    raise ValueError('Error: too short grey zone duration after incorrect trial:', next_grey_lap_duration)

        if (verbous > 0):
            print('lick and reward position calculated')

        ##################################################################################
        ## speed vs. time
        ##################################################################################
        self.imaging_data = True

        if (np.isnan(self.frames_time).any()): # we don't have imaging data
            self.imaging_data = False
            ## resample time uniformly for calculating speed
            start_time = np.ceil(self.raw_time.min()/self.frame_period)*self.frame_period
            end_time = np.floor(self.raw_time.max()/self.frame_period)*self.frame_period
            Ntimes = int(round((end_time - start_time) / self.frame_period)) + 1
            self.frames_time = np.linspace(start_time, end_time, Ntimes)
            self.frames_pos = F(self.frames_time)
        else:
            self.n_cells = self.frames_dF_F.shape[0]

        # if (max(self.frames_time) < self.raw_time.max() - self.frame_period): # imiging finished before end of lap...
        #     ## we need to amend the frames_time, frames_pos and frames_spikes and frames_dF
        #     start_time = max(self.frames_time) + self.frame_period
        #     end_time = np.floor(self.raw_time.max()/self.frame_period)*self.frame_period
        #     Ntimes = int(round((end_time - start_time) / self.frame_period)) + 1
        #     new_frames_time = np.linspace(start_time, end_time, Ntimes)
        #     L_new_frames = len(new_frames_time)
        #     self.frames_pos = np.hstack((self.frames_pos, F(new_frames_time)))
        #     self.frames_spikes = np.hstack((self.frames_spikes,  np.full((self.n_cells, L_new_frames), np.nan)))
        #     self.frames_dF_F = np.hstack((self.frames_dF_F,  np.full((self.n_cells, L_new_frames), np.nan)))
        #     self.frames_time = np.hstack((self.frames_time, new_frames_time))

        ## calculate the speed during the frames
        speed = np.diff(self.frames_pos) * self.speed_factor / self.frame_period # cm / s       
        speed_first = 2 * speed[0] - speed[1] # linear extrapolation: x1 - (x2 - x1)
        self.frames_speed = np.hstack([speed_first, speed])

        if (verbous > 0):
            print('speed calculated')
        
        ##################################################################################
        ## speed, lick and spiking vs. position
        ##################################################################################

        ####################################################################
        ## calculate the lick-rate and the average speed versus location    
        bin_counts = np.zeros(self.N_pos_bins)
        total_speed = np.zeros(self.N_pos_bins)

        last_bin_number = 0 # each spike is assigned to all position bins since the last imaging frame 
        for i_frame in range(len(self.frames_pos)):
            next_bin_number = int(self.frames_pos[i_frame] // 70) 
            if ((next_bin_number > last_bin_number + 1) & self.multiplane):
                bin_number = np.arange(last_bin_number+1, next_bin_number+1) # the sequence ends at next_bun_number
                n_bins = len(bin_number)
                speed_to_add = np.repeat(self.frames_speed[i_frame] * self.frame_period/n_bins, n_bins)
            else:
                bin_number = next_bin_number
                n_bins = 1
                speed_to_add = self.frames_speed[i_frame] * self.frame_period
            bin_counts[bin_number] += 1 / n_bins
            total_speed[bin_number] = total_speed[bin_number] + speed_to_add
            last_bin_number = next_bin_number

        self.T_pos = bin_counts * self.frame_period           # used for lick rate and average speed

        # total_speed = total_speed * self.frame_period
        self.ave_speed = nan_divide(total_speed, self.T_pos, where=(self.T_pos > 0.025))

        lbin_counts = np.zeros(self.N_pos_bins)
        for lpos in self.lick_position:
            lbin_number = int(lpos // 70)
            lbin_counts[lbin_number] += 1
        self.N_licks = lbin_counts
        self.lick_rate = nan_divide(self.N_licks, self.T_pos, where=(self.T_pos > 0.025))
        if (verbous > 0):
            print('lick rate calculated')

        ####################################################################
        ## calculate the cell activations (spike rate) as a function of position
        if (self.imaging_data == True):
            fast_bin_counts = np.zeros(self.N_pos_bins)
            self.spks_pos = np.zeros((self.n_cells, self.N_pos_bins)) # sum of spike counts measured at a given position
            self.event_rate = np.zeros((self.n_cells, self.N_pos_bins)) # spike rate 

            last_bin_number = 0 # each spike is assigned to all position bins since the last imaging frame 
            for i_frame in range(len(self.frames_pos)):
                next_bin_number = int(self.frames_pos[i_frame] // 70) 
                if ((next_bin_number > last_bin_number + 1) & self.multiplane):
                    bin_number = np.arange(last_bin_number+1, next_bin_number+1) # the sequence ends at next_bun_number
                    n_bins = len(bin_number)
                    added_spikes = np.tile(self.frames_spikes[:,i_frame], (n_bins, 1)).T # prepare a matrix with the spikes to add at multiple spatial bins
                    # print('multiple position bins: ', self.lap, i_frame, n_bins)
                else:
                    bin_number = next_bin_number
                    n_bins = 1
                    added_spikes = self.frames_spikes[:,i_frame]
                if (self.frames_speed[i_frame] > self.speed_threshold):
                    fast_bin_counts[bin_number] += 1 / n_bins
                    if (self.elfiz):
                        self.spks_pos[:,bin_number] = self.spks_pos[:,bin_number] + added_spikes
                    else: ### we need to multiply the values with dt_imaging as this converts probilities to expected counts
                        self.spks_pos[:,bin_number] = self.spks_pos[:,bin_number] + added_spikes * self.frame_period / n_bins
                last_bin_number = next_bin_number

            self.T_pos_fast = fast_bin_counts * self.frame_period # used for spike rate calculations
            for bin_number in range(self.N_pos_bins):
                if (self.T_pos_fast[bin_number] > 0): # otherwise the rate will remain 0
                    self.event_rate[:,bin_number] = self.spks_pos[:,bin_number] / self.T_pos_fast[bin_number]

        if (verbous > 0):
            print('ratemaps calculated')

        ####################################################################
        ## Calculate the lick rate before the reward zone - anticipatory licks 210 roxels before zone start
        ## only when the number of zones is 1!

        if (self.n_zones == 1):
            zone_start = np.round(self.zones[0,0] * self.corridor_length_roxel, -1) - 4.5 # threshold of the position rounded to 10s - this is what LabView does
            zone_end = np.round(self.zones[1,0] * self.corridor_length_roxel, -1) - 4.5
            if (len(self.lick_position) > 0):
                lz_posbins = np.array([np.min((np.min(self.frames_pos)-1, np.min(self.lick_position)-1, 0)), zone_start-420, zone_start-210, zone_start, zone_end, self.corridor_length_roxel])
            else :
                lz_posbins = np.array([np.min((np.min(self.frames_pos)-1, 0)), zone_start-420, zone_start-210, zone_start, zone_end, self.corridor_length_roxel])


            lz_bin_counts = np.zeros(5)
            for pos in self.frames_pos:
                bin_number = np.max(np.where(pos > lz_posbins))
                lz_bin_counts[bin_number] += 1
            T_lz_pos = lz_bin_counts * self.frame_period

            lz_lbin_counts = np.zeros(5)
            for lpos in self.lick_position:
                lbin_number = np.max(np.where(lpos > lz_posbins))
                lz_lbin_counts[lbin_number] += 1
            lz_lick_rate = nan_divide(lz_lbin_counts, T_lz_pos, where=(T_lz_pos>0.025))
            self.preZoneRate = [lz_lick_rate[1], lz_lick_rate[2]]
            
            if (verbous > 0):
                print('Zone-Rates calculated')

    def plot_tx(self, fluo=False, th=25):
        ## fluo: True or Fasle, whether fluoresnece data should be plotted.
        ## th: threshold for plotting the fluorescence data - only cells with spikes > th are shown
        ##      when plotting elphys data, th should be -0.5 to show the voltage trace

        colmap = plt.cm.get_cmap('jet')   
        vshift = 0
        colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

        ## first, plot position versus time
        ax_top.plot(self.frames_time, self.frames_pos, color=colmap(90))
        ax_top.plot(self.raw_time, self.raw_position, color=colmap(30))

        ax_top.scatter(self.lick_times, np.repeat(self.frames_pos.min(), len(self.lick_times)), marker="|", s=100, c=colmap(180))
        ax_top.scatter(self.reward_times, np.repeat(self.frames_pos.min()+100, len(self.reward_times)), marker="|", s=100, c=colmap(230))
        ax_top.set_ylabel('position')
        ax_top.set_xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)
        ax_top.set_ylim(0, self.corridor_length_roxel)

        ## next, plot dF/F versus time (or spikes)
        if (self.n_cells > 1):
            # act_cells = np.nonzero(np.amax(self.frames_dF_F, 1) > th)[0]
            act_cells = np.nonzero(np.amax(self.frames_spikes, 1) > th)[0]
            max_range = np.nanmax(self.event_rate)
            for i in range(self.n_cells):
            # for i in (252, 258, 275):
                if (fluo & (i in act_cells)):
                    dFs_normalised = self.frames_dF_F[i,:]
                    if (self.elfiz):
                        dFs_normalised = (dFs_normalised - min(dFs_normalised)) / 10
                        vshift = 2
                    ax_bottom.plot(self.frames_time, dFs_normalised + i, alpha=0.5, color=colmap(np.remainder(i, 255)))
                events = self.frames_spikes[i,:]
                events = 50 * events / max_range
                ii_events = np.nonzero(events)[0]
                ax_bottom.scatter(self.frames_time[ii_events], np.ones(len(ii_events)) * i + vshift, s=events[ii_events], cmap=colmap, c=(np.ones(len(ii_events)) * np.remainder(i, 255)), norm=colnorm)

            ylab_string = 'dF_F, spikes (max: ' + str(np.round(max_range, 1)) +  ' )'
            ax_bottom.set_ylabel(ylab_string)
            ax_bottom.set_xlabel('time (s)')
            plot_title = 'dF/F of all neurons  in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
            ax_bottom.set_title(plot_title)
            ax_bottom.set_ylim(0, self.n_cells+5)
            # ax_bottom.set_ylim(250, 280)

        plt.show(block=False)
       

    def plot_xv(self):
        colmap = plt.cm.get_cmap('jet')   
        colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,8), sharex=True,  gridspec_kw={'height_ratios': [1, 3]})
        ax_top.plot(self.frames_pos, self.frames_speed, c=colmap(90))
        ax_top.step(self.bincenters, self.ave_speed, where='mid', c=colmap(30))
        ax_top.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, color=colmap(180))
        ax_top.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, color=colmap(230))
        ax_top.set_ylabel('speed (cm/s)')
        ax_top.set_ylim([min(0, self.frames_speed.min()), max(self.frames_speed.max(), 30)])
        ax_top.set_xlabel('position')
        plot_title = 'Mouse: ' + self.name + ' speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)


        bottom, top = ax_top.get_ylim()
        left = self.zones[0,0] * self.corridor_length_roxel
        right = self.zones[1,0] * self.corridor_length_roxel

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        ax_top.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * self.corridor_length_roxel
                right = self.zones[1,i] * self.corridor_length_roxel
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                ax_top.add_patch(polygon)

        ax2 = ax_top.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', c=colmap(200), linewidth=1)
        ax2.set_ylabel('lick rate (lick/s)', color=colmap(200))
        ax2.tick_params(axis='y', labelcolor=colmap(200))
        ax2.set_ylim([-1,max(2*np.nanmax(self.lick_rate), 20)])

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


        # colmap = plt.cm.get_cmap('jet')
        # colnorm = matcols.Normalize(vmin=0, vmax=255, clip=False)
        # x = np.random.rand(4)
        # y = np.random.rand(4)
        # area = (np.abs(x/max(x))*30)**2
        # colors = 232

    def plot_txv(self):
        cmap = plt.cm.get_cmap('jet')   
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,6))

        ## first, plot position versus time
        ax_top.plot(self.frames_time, self.frames_pos, c=cmap(50))
        ax_top.plot(self.raw_time, self.raw_position, c=cmap(90))

        ax_top.scatter(self.lick_times, np.repeat(200, len(self.lick_times)), marker="|", s=100, color=cmap(180))
        ax_top.scatter(self.reward_times, np.repeat(400, len(self.reward_times)), marker="|", s=100, color=cmap(230))
        ax_top.set_ylabel('position')
        ax_top.set_xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position and speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)
        ax_top.set_ylim(0, self.corridor_length_roxel + 100)


        ## next, plot speed versus position
        ax_bottom.plot(self.frames_pos, self.frames_speed, c=cmap(80))
        ax_bottom.step(self.bincenters, self.ave_speed, where='mid', c=cmap(30))
        ax_bottom.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, color=cmap(180))
        ax_bottom.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, color=cmap(230))
        ax_bottom.set_ylabel('speed (cm/s)')
        ax_bottom.set_xlabel('position')
        ax_bottom.set_ylim([min(0, self.frames_speed.min()), max(self.frames_speed.max(), 30)])

        bottom, top = plt.ylim()
        left = self.zones[0,0] * self.corridor_length_roxel
        right = self.zones[1,0] * self.corridor_length_roxel

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        ax_bottom.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * self.corridor_length_roxel
                right = self.zones[1,i] * self.corridor_length_roxel
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                ax_bottom.add_patch(polygon)

        ax2 = ax_bottom.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', c=cmap(180), linewidth=1)
        ax2.set_ylabel('lick rate (lick/s)', color=cmap(180))
        ax2.tick_params(axis='y', labelcolor=cmap(180))
        ax2.set_ylim([-1,max(2*np.nanmax(self.lick_rate), 20)])

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
        self.anti = False
        if (self.m_anti > 0):
            self.test = scipy.stats.wilcoxon(self.baseline, self.anti_rate)
            if ((self.test[1] < 0.01 ) & (greater == True)):
                self.anti = True
        else:
            self.test = [np.nan, 1]


def HolmBonfMat(P_mat, p_val):
    ## P_mat is a matrix with the 
    ## P-values of different tests in each row and
    ## Cells in each column

    # m = P_mat.shape[0]
    n_cells = P_mat.shape[1]
    i_significant = np.zeros_like(P_mat)
    for i_cell in np.arange(n_cells):
        i_significant[:,i_cell] = HolmBonf(P_mat[:,i_cell], 0.05)
    return i_significant

def HolmBonf(P_vec, p_val):
    m = len(P_vec)
    Pindex = np.argsort(P_vec)
    i_significant = np.zeros(m)
    for i_test in np.arange(m):
        if ( P_vec[Pindex[i_test]] < p_val / (m - i_test)):
            i_significant[Pindex[i_test]] = 1
        else:
            break

    return i_significant


##########################################################
def LocateImaging(trigger_log_file_string, TRIGGER_VOLTAGE_FILENAME):
    # 1. TRIGGER_DATA  = np.array, 4 x N_triggers, each row is the 1. start time, 2, end time, duration, ITT
    # 2. select only trigger data with ITT > 10 ms
    # 3. find the shortest trigger
    # 4. find candidate trigger times in the log_trigger by matching trigger duration
    # 5. check the next ITTS in voltage recording and log trigger
    # intputs: 
    #   self.trigger_log_starts,        normal LETTERS: variables defined with LabView time axis
    #   self.trigger_log_lengths, 
    #   self.TRIGGER_VOLTAGE_VALUE,      CAPITAL LETTERS: variables defined with IMAGING time axis
    #   self.TRIGGER_VOLTAGE_TIMES
    #
    # output:
    #   self.imstart_time: singe scalar [s]: Labview time of the start of voltage-imaging recordings
    #
    # only works for 1 imaging session...
    # self.imstart_time = 537.133055 # Bazsi's best guess
    # print('Imaging time axis guessed by Bazsi...')
    
    #0)load recorded trigger 
    trigger_log_starts = [] ## s
    trigger_log_lengths = []      
    trigger_log_file=open(trigger_log_file_string, newline='')
    log_file_reader=csv.reader(trigger_log_file, delimiter=',')
    next(log_file_reader, None)#skip the headers
    for line in log_file_reader:             
        trigger_log_starts.append(float(line[0])) # seconds
        trigger_log_lengths.append(float(line[1]) / 1000) # convert to seconds from ms
    print('trigger logfile loaded')
    trigger_starts = np.array(trigger_log_starts)
    trigger_lengths = np.array(trigger_log_lengths)


    TRIGGER_VOLTAGE_VALUE = [] 
    TRIGGER_VOLTAGE_TIMES = [] ## ms
    trigger_signal_file=open(TRIGGER_VOLTAGE_FILENAME, 'r', newline='')
    trigger_reader=csv.reader(trigger_signal_file, delimiter=',')
    next(trigger_reader, None)
    for line in trigger_reader:
        TRIGGER_VOLTAGE_VALUE.append(float(line[1])) 
        TRIGGER_VOLTAGE_TIMES.append(float(line[0]) / 1000) # converting it to seconds
    TRIGGER_VOLTAGE=np.array(TRIGGER_VOLTAGE_VALUE)
    TRIGGER_TIMES=np.array(TRIGGER_VOLTAGE_TIMES)
    print('trigger voltage signal loaded')
    
    ## find trigger start and end times
    rise_index=np.nonzero((TRIGGER_VOLTAGE[0:-1] < 1)&(TRIGGER_VOLTAGE[1:]>= 1))[0]+1#+1 needed otherwise we are pointing to the index just before the trigger
    RISE_T=TRIGGER_TIMES[rise_index]
    
    fall_index=np.nonzero((TRIGGER_VOLTAGE[0:-1] > 1)&(TRIGGER_VOLTAGE[1:]<= 1))[0]+1
    FALL_T=TRIGGER_TIMES[fall_index]
    
    # pairing rises with falls
    if (RISE_T[0]>FALL_T[0]):
        print('deleting first fall')
        FALL_T = np.delete(FALL_T,0)
    if (RISE_T[-1] > FALL_T[-1]):
        print('deleting last rise')
        RISE_T=np.delete(RISE_T,-1)

    if np.size(RISE_T)!=np.size(FALL_T):
        print(np.size(RISE_T),np.size(FALL_T))
        print('trigger ascending and desending edges do not match! unable to locate imaging part')
        imstart_time = np.nan
        return imstart_time

    #1) filling up TRIGGER_DATA array:
    #TRIGGER_DATA: 0. start time, 1. end time, 2. duration, 3.ITT, 4. index
    TRIGGER_DATA = np.zeros((np.size(RISE_T),5))
    TRIGGER_DATA[:,0] = RISE_T
    TRIGGER_DATA[:,1] = FALL_T
    TRIGGER_DATA[:,2]=FALL_T-RISE_T # duration
    TEMP_FALL = np.concatenate([[0],FALL_T]) 
    TEMP_FALL = np.delete(TEMP_FALL,-1)
    TRIGGER_DATA[:,3] = RISE_T - TEMP_FALL # previous down duration - Inter Trigger Time
    TRIGGER_DATA[:,4] = np.arange(0,np.size(RISE_T))
        
    #2) keeping only triggers with ITT > 10 ms    
    valid_indexes=np.nonzero(TRIGGER_DATA[:,3] > 0.010)[0]
    TRIGGER_DATA_sub=TRIGGER_DATA[valid_indexes,:]
    
    #3) find the valid shortest trigger
    minindex = np.argmin(TRIGGER_DATA_sub[:,2])
    used_index = int(TRIGGER_DATA_sub[minindex][4])
    n_extra_indexes = min(5,TRIGGER_DATA.shape[0]-used_index)
    print('triggers after:',TRIGGER_DATA.shape[0]-used_index)
    print('n_extra_indexes',n_extra_indexes)

    #4)find the candidate trigger times
    candidate_log_indexes = []
    for i in range(len(trigger_lengths)):
        if (abs(trigger_lengths[i] - TRIGGER_DATA[used_index][2]) < 0.007):
            candidate_log_indexes.append(i)
    print('candidate log indexes',candidate_log_indexes)

    #5)check the next ITT-s, locate relevant behavior
    print('min recorded trigger length:',TRIGGER_DATA[used_index,2])
    if TRIGGER_DATA[used_index,2] > 0.600:
        print('Warning! No short enough trigger in this recording! Unable to locate imaging')
        imstart_time = np.nan
        # return
    else:
        match_found = False
        for i in range(len(candidate_log_indexes)):    
            log_reference_index=candidate_log_indexes[i]
            difs=[]
            if len(trigger_starts) > log_reference_index + n_extra_indexes:
                for j in range(n_extra_indexes):
                    dif_log = trigger_starts[log_reference_index + j] - trigger_starts[log_reference_index]
                    dif_mes = TRIGGER_DATA[used_index+j,0] - TRIGGER_DATA[used_index,0]
                    delta = abs(dif_log - dif_mes)
                    difs.append(delta)
                # print(trigger_log_lengths[candidate_log_indexes[i]],'log',dif_log,'mes', dif_mes,'dif', delta)
                if max(difs) < 0.009:
                    if match_found==False:                      
                        lap_time_of_first_frame = trigger_starts[log_reference_index] - TRIGGER_DATA[used_index,0]
                        print('relevant behavior located, lap time of the first frame:',lap_time_of_first_frame, ', log reference index:', log_reference_index)
                        match_found=True
                    else:
                        print('Warning! More than one trigger matches found!')
            else:
                print('slight warning - testing some late candidates failed')

        if match_found==True:
            imstart_time = lap_time_of_first_frame
        else:
            sys.exit('no precise trigger mach found: need to refine code or check device')
    return imstart_time
