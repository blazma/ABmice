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
from xml.dom import minidom
from matplotlib.backends.backend_pdf import PdfPages


from utils import *
from Stages import *
from Corridors import *
from ImShuffle import *

if (platform == 'darwin'):
    csv_kwargs = {'delimiter':' '}
else:
    csv_kwargs = {'delimiter':' ', 'newline':''}


class ImagingSessionData:
    'Base structure for both imaging and behaviour data'
    def __init__(self, datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=np.nan, selected_laps=None, speed_threshold=5, randseed=123, elfiz=False):
        self.datapath = datapath
        self.date_time = date_time
        self.name = name
        self.task = task
        self.suite2p_folder = suite2p_folder
        self.imaging_logfile_name = imaging_logfile_name

        self.stage = 0
        self.stages = []
        self.sessionID = sessionID
        self.randseed = randseed
        self.selected_laps = selected_laps
        self.speed_threshold = speed_threshold
        self.elfiz = elfiz

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
        if (self.elfiz):
            F_string = self.suite2p_folder + 'Vm_' + self.imaging_logfile_name + '.npy'
            spks_string = self.suite2p_folder + 'spikes_' + self.imaging_logfile_name + '.npy'
            time_string = self.suite2p_folder + 'frame_times_' + self.imaging_logfile_name + '.npy'

            self.F = np.load(F_string) # npy array, N_ROI x N_frames, fluorescence traces of ROIs from suite2p
            self.raw_spks = np.load(spks_string) # npy array, N_ROI x N_frames, spike events detected from suite2p
            print('elfiz data loaded')       

            self.frame_times = np.load(time_string)[0] + self.imstart_time
            print('elfiz time axis loaded')       
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
        else :
            F_string = self.suite2p_folder + 'F.npy'
            # Fneu_string = self.suite2p_folder + 'Fneu.npy'
            spks_string = self.suite2p_folder + 'spks.npy'
            iscell_string = self.suite2p_folder + 'iscell.npy'
            
            self.F_all = np.load(F_string) # npy array, N_ROI x N_frames, fluorescence traces of ROIs from suite2p
            # self.Fneu = np.load(Fneu_string) # npy array, N_ROI x N_frames, fluorescence traces of neuropil from suite2p
            self.spks_all = np.load(spks_string) # npy array, N_ROI x N_frames, spike events detected from suite2p
            self.iscell = np.load(iscell_string) # np array, N_ROI x 2, 1st col: binary classified as cell. 2nd P_cell?
            self.stat_string = self.suite2p_folder + 'stat.npy' # we may load these later if needed
            self.ops_string = self.suite2p_folder + 'ops.npy'
            print('suite2p data loaded')               

            self.frame_times = np.nan # labview coordinates
            self.LoadImaging_times(self.imstart_time)
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

          
        ##################################################
        ## loading behavioral data
        ##################################################

        self.ImLaps = [] # list containing a special class for storing the imaging and behavioral data for single laps
        self.n_laps = 0 # total number of laps
        self.i_Laps_ImData = np.zeros(1) # np array with the index of laps with imaging
        self.i_corridors = np.zeros(1) # np array with the index of corridors in each run

        self.get_lapdata(self.datapath, self.date_time, self.name, self.task, selected_laps=self.selected_laps) # collects all behavioral and imaging data and sort it into laps, storing each in a Lap_ImData object
        if (self.n_laps == -1):
            print('Error: missing laps are found in the ExpStateMachineLog file! No analysis was performed, check the logfiles!')
            return

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
        print('number of laps with imaging data:', self.N_ImLaps)
        self.raw_activity_tensor = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # a tensor with space x neurons x trials containing the spikes
        self.raw_activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # a tensor with space x trials containing the time spent at each location in each lap
        self.activity_tensor = np.zeros((self.N_pos_bins, self.N_cells, self.N_ImLaps)) # same as the activity tensor spatially smoothed
        self.activity_tensor_time = np.zeros((self.N_pos_bins, self.N_ImLaps)) # same as the activity tensor time spatially smoothed
        print('laps with image data:')
        print(self.i_Laps_ImData)
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

    def write_params(self, filename):
        # write the parameters of the current ImagingSessionData into the given file

        data_folder = self.suite2p_folder + 'analysed_data'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        param_filename = data_folder + '/' + filename
        with open(param_filename, mode='w') as paramfile:
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
        data_folder = self.suite2p_folder + 'analysed_data'
        if not os.path.exists(data_folder):
            print ('Error: data directory', data_folder, 'does NOT exist!')    
            return False

        param_filename = data_folder + '/' + filename
        param_file=open(param_filename)
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
        if (datapath[1] != self.datapath):
            print('Error: datapath read from file not equals the datapath in the loaded session!')
            return False

        suite2p_folder = next(param_file_reader)
        if (suite2p_folder[1] != self.suite2p_folder):
            print('Error: suite2p_folder read from file not equals the suite2p_folder in the loaded session!')
            return False

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
        action_log_file=open(action_log_file_string)
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

        frames = imaging_logfile.getElementsByTagName('Frame')
        self.frame_times = np.zeros(len(frames)) # this is already in labview time
        for i in range(len(frames)):
            self.frame_times[i] = float(frames[i].attributes['relativeTime'].value) + corrected_offset
        if (len(self.frame_times) != self.F_all.shape[1]):
            print('Warning: imaging frame number does not match suite2p frame number! Something is wrong!')
            N_frames = min([len(self.frame_times), self.F_all.shape[1]])
            if (len(self.frame_times) < self.F_all.shape[1]):
                self.F_all = self.F_all[:, 1:N_frames]
                self.spks_all = self.spks_all[:, 1:N_frames]
                # self.Fneu = self.Fneu[:, 1:N_frames]
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
        self.anticipatory = []

        for row in range(self.N_corridors):
            ids = np.where(self.i_corridors == self.corridors[row])
            n_laps = np.shape(ids)[1]
            n_zones = np.shape(self.ImLaps[ids[0][0]].zones)[1]
            if ((n_zones == 1) & (n_laps > 2)):
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

        ## to calculate the SD and SNR, we need baseline periods with no spikes for at least 1 sec
        frame_rate = int(np.ceil(1/np.median(np.diff(self.frame_times))))
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
            for k in range(len(rise_ind)):
                i_start = fall_ind[k] + L_after_spike
                i_end = rise_ind[k] - L_before_spike
                sds[k] = np.sqrt(np.var(self.dF_F[i_cell,i_start:i_end]))

            self.cell_SDs[i_cell] = np.mean(sds)
            self.cell_SNR[i_cell] = max(self.dF_F[i_cell,:]) / np.mean(sds)
            self.spks[i_cell,:] = self.spks[i_cell,:]# / baseline / self.cell_SDs[i_cell]

        print('SNR done')

        print('dF/F calculated for cell ROI-s')


    def calc_SD_SNR_elfiz(self):
        for i_cell in range(self.N_cells):
            self.cell_SDs[i_cell] = np.random.rand()
            self.cell_SNR[i_cell] = np.random.rand()



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
        N_0lap = 0 # Counting the non-valid laps

        #################################################
        ## add position, and lap info into the imaging frames
        #################################################
        F = interp1d(laptime, pos) 
        self.frame_pos = np.round(F(self.frame_times), 2)
        F = interp1d(laptime, lap) 
        self.frame_laps = F(self.frame_times) 
        F = interp1d(laptime, maze) 
        self.frame_maze = F(self.frame_times) 

        ## frame_laps is NOT integer for frames between laps
        ## however, it MAY not be integer even even for frames within a lap...
        #################################################
        ## end of - add position, and lap info into the imaging frames
        #################################################
        i_ImData = [] # index of laps with imaging data
        i_corrids = [] # ID of corridor for the current lap

        ### include only a subset of laps
        if selected_laps is None:
            all_laps_set = np.unique(lap)
            # print('we select all laps', all_laps_set)
        else:
            all_laps_set = np.intersect1d(np.unique(lap), selected_laps)
            # print('sel of selected all laps', all_laps_set)

        self.n_laps = 0


        for i_lap in all_laps_set: 
            y = lap == i_lap # index for the current lap

            mode_lap = np.prod(mode[y]) # 1 if all elements are recorded in 'Go' mode

            maze_lap = np.unique(maze[y])
            if (len(maze_lap) == 1):
                corridor = self.all_corridors[int(maze_lap)] # the maze_lap is the index of the available corridors in the given stage
            else:
                corridor = -1
            # print('corridor in lap ', self.n_laps, ':', corridor)

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

                if (sstage_lap != current_sstage):
                    print('############################################################')
                    print('substage change detected!')
                    print('first lap in substage ', sstage_lap, 'is lap', self.n_laps, ', which started at t', t_lap[0])
                    print('the time of the change in imaging time is: ', t_lap[0] - self.imstart_time)
                    print('############################################################')
                    current_sstage = sstage_lap

                istart = np.where(y)[0][0] # action is not a np.array, array indexing does not work
                iend = np.where(y)[0][-1] + 1
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

                actions = []
                for j in range(len(action_lap)):
                    if not((action_lap[j]) in ['No', 'TrialReward']):
                        actions.append([t_lap[j], action_lap[j]])

                ## include only valid laps
                add_lap = True
                if (mode_lap == 0):
                    add_lap = False

                ### imaging data    
                iframes = np.where(self.frame_laps == i_lap)[0]
                # print('In lap ', self.n_laps, ' we have ', len(t_lap), 'datapoints and ', len(iframes), 'frames')

                if ((len(iframes) > self.N_pos_bins) & (add_lap == True)): # there is imaging data belonging to this lap...
                    # print('frames:', min(iframes), max(iframes))
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
                    
                self.ImLaps.append(Lap_ImData(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time, self.corridor_list, speed_threshold=self.speed_threshold, elfiz=self.elfiz))
                self.n_laps = self.n_laps + 1
                # print(self.n_laps)
            else:
                N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1) - we do not do anything with this...

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
        self.activity_tensor[0,:,:] = (self.raw_activity_tensor[0,:,:] + self.raw_activity_tensor[1,:,:]) / 2
        self.activity_tensor[-1,:,:] = (self.raw_activity_tensor[-1,:,:] + self.raw_activity_tensor[-1,:,:]) / 2
        self.activity_tensor_time[0,:] = (self.raw_activity_tensor_time[0,:] + self.raw_activity_tensor_time[1,:]) / 2
        self.activity_tensor_time[-1,:] = (self.raw_activity_tensor_time[-1,:] + self.raw_activity_tensor_time[-1,:]) / 2
        for i_bin in np.arange(1, self.N_pos_bins-1):
            self.activity_tensor[i_bin,:,:] = np.average(self.raw_activity_tensor[(i_bin-1):(i_bin+2),:,:], axis=0)
            self.activity_tensor_time[i_bin,:] = np.average(self.raw_activity_tensor_time[(i_bin-1):(i_bin+2),:], axis=0)

        i_valid_laps = np.nonzero(valid_lap)[0]
        self.i_Laps_ImData = self.i_Laps_ImData[i_valid_laps]
        self.raw_activity_tensor = self.raw_activity_tensor[:,:,i_valid_laps]
        self.raw_activity_tensor_time = self.raw_activity_tensor_time[:,i_valid_laps]
        self.activity_tensor = self.activity_tensor[:,:,i_valid_laps]
        self.activity_tensor_time = self.activity_tensor_time[:,i_valid_laps]


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
                if (sum(self.i_corridors[self.i_Laps_ImData] == corridor) > 5):
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
                    self.ratemaps.append(rate_matrix)
                    
                    print('calculating rate, reliability and Fano factor...')
                    ## average firing rate
                    rates = np.sum(total_spikes, axis=0) / np.sum(total_time)
                    self.cell_rates.append(rates)

                    if (self.task == 'contingency_learning'):
                        rates_pattern1 = np.sum(total_spikes[0:14,:], axis=0) / np.sum(total_time[0:14])
                        rates_pattern2 = np.sum(total_spikes[14:28,:], axis=0) / np.sum(total_time[14:28])
                        rates_pattern3 = np.sum(total_spikes[28:42,:], axis=0) / np.sum(total_time[28:42])
                        rates_reward = np.sum(total_spikes[42:47,:], axis=0) / np.sum(total_time[42:47])
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
        M = int(self.N_corridors * (self.N_corridors - 1) / 2)
        similarity_matrix = np.zeros((M, self.N_cells))
        m = 0
        for i_cor in np.arange(self.N_corridors-1):
            for j_cor in np.arange(i_cor+1, self.N_corridors):
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

    def Hainmuller_PCs(self):
        ## ratemaps: similar to the activity tensor, the laps are sorted by the corridors
        self.candidate_PCs = [] # a list, each element is a vector of Trues and Falses of candidate place cells with at least 1 place field according to Hainmuller and Bartos 2018
        self.accepted_PCs = [] # a list, each element is a vector of Trues and Falses of accepted place cells after bootstrapping

        ## we calculate the rate matrix for all corridors - we need to use the same colors for the images
        for corrid in self.corridors:
            if (sum(self.i_corridors[self.i_Laps_ImData] == corrid) > 5):
                # select the laps in the corridor 
                i_corrid = int(np.nonzero(self.corridors == corrid)[0])
                rate_matrix = self.ratemaps[i_corrid]
                # only laps with imaging data are selected - this will index the activity_tensor

                candidate_cells = np.zeros(self.N_cells)
                
                i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corrid)[0] 
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
    def calc_shuffle(self, cellids, n=1000, mode='shift', batchsize=25, verbous=True):
        ## cellids: numpy array - the index of the cells to be included in the analysis
        ## n: integer, number of shuffles
        ## mode: 'random' or 'shift'. 
            # random: totally randomize the spike times; 
            # shift: circularly shift spike times 
        ## batchsize: integer. To make computation faster, shuffling is done in batches of size batchsize
        ## verbous: True of False: information is gizev about the progress

        cellids = np.array(cellids)
        raw_spikes = self.raw_spks[cellids,:]
        calculate_shuffles = True

        ##########################################################################
        ## reading shuffling data from file
        ##########################################################################

        data_folder = self.suite2p_folder + 'analysed_data'
        shuffle_filename = 'shuffle_stats_n' + str(n) + '_mode_' + mode + '.csv'
        shuffle_path = data_folder + '/' + shuffle_filename
        if os.path.exists(shuffle_path):
            calculate_shuffles = False
            if (self.check_params(shuffle_filename)):
                if (verbous):
                    print('loading shuffling P-values from file...')
                ## load from file
                shuffle_file=open(shuffle_path)
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
                        if (verbous):
                            print ('cellids of the saved file does not match the cellids provided. We will perform shuffling.')
                        calculate_shuffles = True

                else :# fill in the P-value array
                    if (verbous):
                        print ('number of P-values read from the saved file for each cell does not match the number expected for a given number of corridor. We will perform shuffling.')
                    calculate_shuffles = True
            else:
                calculate_shuffles = True

        ##########################################################################
        ## calculating shuffling - if appropriate file not found
        ##########################################################################
        if (calculate_shuffles):
            if (verbous):
                print('calculating shuffles...')
            shuffle_stats = ImShuffle(self.datapath, self.date_time, self.name, self.task, self.stage, raw_spikes, self.frame_times, self.frame_pos, self.frame_laps, N_shuffle=n, cellids=cellids, mode=mode,    batchsize=batchsize, randseed=self.randseed, selected_laps=self.selected_laps, elfiz=self.elfiz)
            # shuffle_stats = ImShuffle(D1.datapath,   D1.date_time,   D1.name,   D1.task,   D1.stage,   raw_spikes, D1.frame_times,   D1.frame_pos,   D1.frame_laps,   N_shuffle=N_shuffle, cellids=cellids, mode='shift', batchsize=25,        randseed=D1.randseed, selected_laps=np.arange(20,80), elfiz=True)

            NN = cellids.size
            N_corrids = len(shuffle_stats.ratemaps)
            sanity_checks_passed = True
            if ((N_corrids) != self.N_corridors):
                    print ('warning: number of corridors is different between shuffling and control!')
                    sanity_checks_passed = False            
            for i_cor in range(N_corrids):
                if (self.elfiz == True): # we use a different time resolution for shuffling...
                    if (np.abs(shuffle_stats.cell_reliability[i_cor][0,n] - self.cell_reliability[i_cor][0]) > 0.1):
                        print ('warning: calculating reliability is different between shuffling and control!')
                        sanity_checks_passed = False
                    if (np.corrcoef(self.ratemaps[0][:,0], shuffle_stats.ratemaps[0][:,0,n])[0,1] < 0.75):
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
                if (verbous):
                    print ('Shuffling stats calculated succesfully')
            else : 
                if (verbous):
                    print ('Shuffling failed, ask for help...')
                return 

            if (verbous):
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
            with open(shuffle_path, mode='a') as shuffle_file:
                file_writer = csv.writer(shuffle_file, **csv_kwargs)
                file_writer.writerow(Ps_names)
                for i_row in np.arange(shuffle_Pvalues.shape[0]):
                    file_writer.writerow(np.round(shuffle_Pvalues[i_row,:], 6))
            print('P values for n=', len(cellids), ' saved into file: ', shuffle_path)

        ##########################################################################
        ## processing the shuffling stats ...
        ##########################################################################

        # shuffle_Pvalues:
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
        if (verbous):
            print('tuned cells:', self.tuned_cells)
        if (self.N_corridors > 1):
            self.selective_cells = cellids[np.where(self.ii_tuned_cells[:, self.N_corridors*3])]
            self.similar_cells = cellids[np.where(self.ii_tuned_cells[:, self.N_corridors*3+1])]
            if (verbous):
                print('selective cells:', self.selective_cells)
                print('similar cells:', self.similar_cells)
            if (self.task=='contingency_learning'):
                for kk in np.arange(4):
                    self.pattern_selective_cells.append(cellids[np.where(self.ii_tuned_cells[:, self.N_corridors*3+2+kk])])
                if (verbous):
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



    def plot_ratemaps(self, corridor=-1, normalized=False, sorted=False, corridor_sort=-1, cellids=np.array([-1]), vmax=0):
        ## plot the average event rate of all cells in a given corridor
        ## corridor: integer, the ID of a valid corridor
        ##      if corridor == -1 then all corridors are used
        ## normalized: True or False. If True then  each cell ratemap is normalized to have a max = 1
        ## sorted: sorting the ramemaps by their peaks
        ## corridor_sort: which corridor to use for sorting the ratemaps
        ## cellids: np array with the indexes of the cells to be plotted. when -1: all cells are plotted
        ## vmax: float. If ratemaps are not normalised then the max range of the colors will be at least vmax. 
                # If one of the ratemaps has a higher peak, then vmax is replaced by that peak 
        
        if (cellids[0] == -1):
            cellids = np.array(range(self.activity_tensor.shape[1]))
        N_cells_plotted = len(cellids)

        nbins = self.activity_tensor.shape[0]
        ## we calculate the rate matrix for all corridors - we need to use the same colors for the images
        
        if (normalized):
            vmax = 1
        else: 
            for i_corrid in range(self.N_corridors):
                if (np.max(self.ratemaps[i_corrid]) > vmax):
                    vmax = np.max(self.ratemaps[i_corrid])

        if (corridor == -1):

            fig, axs = plt.subplots(1, self.N_corridors, figsize=(6+self.N_corridors*2,8), sharex=True, sharey=True)
            ims = []

            if (corridor_sort == -1): # if no corridor is geven for sorting...
                corridor_sort = self.corridors[0]

            if (sorted):
                suptitle_string = 'sorted by corridor ' + str(corridor_sort)
                i_corrid = np.nonzero(corridor_sort == self.corridors)[0][0]
                rmp = np.array(self.ratemaps[i_corrid][:,cellids], copy=True)
                ratemaps_to_sort = np.transpose(rmp)
                sort_index, rmaps = self.sort_ratemaps(ratemaps_to_sort)
            else :
                sort_index = np.arange(len(cellids))
                suptitle_string = 'unsorted'

            for i_corrid in range(self.N_corridors):
                if (N_cells_plotted == self.N_cells) :
                    title_string = 'ratemap of all cells in corridor ' + str(self.corridors[i_corrid])
                else :
                    title_string = 'ratemap of some cells in corridor ' + str(self.corridors[i_corrid])
                rmp = np.array(self.ratemaps[i_corrid][:,cellids], copy=True)
                ratemaps_to_sort = np.transpose(rmp)
                ratemaps_to_plot = ratemaps_to_sort[sort_index,:]
                if (normalized):
                    for i in range(ratemaps_to_plot.shape[0]):
                        rate_range = max(ratemaps_to_plot[i,:]) - min(ratemaps_to_plot[i,:])
                        ratemaps_to_plot[i,:] = (ratemaps_to_plot[i,:] - min(ratemaps_to_plot[i,:])) / rate_range

                axs[i_corrid].set_title(title_string)
                ims.append(axs[i_corrid].matshow(ratemaps_to_plot, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='binary'))
                axs[i_corrid].set_facecolor(matcols.CSS4_COLORS['palegreen'])
                axs[i_corrid].set_yticks(np.arange(len(cellids)))
                if (sorted):
                    axs[i_corrid].set_yticklabels(cellids[sort_index])
                else:
                    axs[i_corrid].set_yticklabels(cellids)
                plt.colorbar(ims[i_corrid], orientation='horizontal',ax=axs[i_corrid])

                ## add reward zones
                zone_starts = self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_starts 
                if (len(zone_starts) > 0):
                    zone_ends = self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_ends
                    bottom, top = axs[i_corrid].get_ylim()
                    for i_zone in range(len(zone_starts)):
                        left = zone_starts[i_zone] * nbins             
                        right = zone_ends[i_zone] * nbins              
                        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                        axs[i_corrid].add_patch(polygon)
                        # print('adding reward zone to the ', i_corrid, 'th corridor, ', self.corridors[i_corrid+1])

            fig.suptitle(suptitle_string)
            fig.tight_layout()
            plt.show(block=False)       

        else:
            if (N_cells_plotted == self.N_cells) :
                title_string = 'ratemap of all cells in corridor ' + str(corridor)
            else :
                title_string = 'ratemap of some cells in corridor ' + str(corridor)

            i_corrid = int(np.nonzero(np.unique(self.i_corridors)==corridor)[0])

            rmp = np.array(self.ratemaps[i_corrid][:,cellids], copy=True)
            ratemaps_to_sort = np.transpose(rmp)
            if (sorted):
                sort_index, ratemaps_to_plot = self.sort_ratemaps(ratemaps_to_sort)
                title_string = title_string + ' sorted'
            else:
                ratemaps_to_plot = ratemaps_to_sort
                title_string = title_string + ' unsorted'

            if (normalized):
                for i in range(ratemaps_to_plot.shape[0]):
                    rate_range = max(ratemaps_to_plot[i,:]) - min(ratemaps_to_plot[i,:])
                    ratemaps_to_plot[i,:] = (ratemaps_to_plot[i,:] - min(ratemaps_to_plot[i,:])) / rate_range

            fig, ax_bottom = plt.subplots(figsize=(6,8))
            ax_bottom.set_title(title_string)
            im1 = ax_bottom.matshow(ratemaps_to_plot, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='binary')
            ax_bottom.set_yticks(np.arange(len(cellids)))
            if (sorted):
                ax_bottom.set_yticklabels(cellids[sort_index])
            else:
                ax_bottom.set_yticklabels(cellids)

            zone_starts = self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_starts 
            if (len(zone_starts) > 0):
                zone_ends = self.corridor_list.corridors[self.corridors[i_corrid]].reward_zone_ends
                bottom, top = ax_bottom.get_ylim()
                for i_zone in range(len(zone_starts)):
                    left = zone_starts[i_zone] * nbins             
                    right = zone_ends[i_zone] * nbins              
                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                    ax_bottom.add_patch(polygon)
                    # print('adding reward zone to the ', i_corrid, 'th corridor, ', self.corridors[i_corrid+1])

            plt.colorbar(im1, orientation='horizontal')
            plt.show(block=False)       


# fig, ax_bottom = plt.subplots(figsize=(6,8))
# ax_bottom.set_title(title_string)
# im1 = ax_bottom.matshow(ratemaps_to_plot, aspect='auto', origin='lower', vmin=0, cmap='binary')
# ax_bottom.set_yticks(np.arange(len(cellids)))
# ax_bottom.set_yticklabels(cellids)

# plt.colorbar(im1)
# plt.show(block=False)       


# fig, ax = plt.subplots(figsize=(6,8))
# im1 = ax.scatter(np.arange(5), (2,5,4,0,3))
# ax.set_title('title')
# ax.set_yticks(np.arange(5))
# ax.set_yticklabels(('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))
# plt.show(block=False)       



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


    def plot_cell_laps(self, cellid, multipdf_object=-1, signal='dF', corridor=-1, reward=True, write_pdf=False, plot_laps='all'):
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

            for cor_index in range(corridor.size):
                corridor_to_plot=corridor[cor_index]
                
                #calculate rate matrix ...again :(
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

            colors1 = plt.cm.binary(np.linspace(0., 1, 100)) 
            n_col2 = int(np.round(max_intensity - 100))
            max_col2 = min(0.65, 0.25 + n_col2 / 100)
            colors2 = plt.cm.autumn(np.linspace(0, max_col2, n_col2))

            # combine them and build a new colormap
            colors = np.vstack((colors1, colors2))
            mymap = matcols.LinearSegmentedColormap.from_list('my_colormap', colors)
            nbins = self.activity_tensor.shape[0]

            fig, ax = plt.subplots(2,corridor.size, squeeze=False, sharey='row', figsize=(6*corridor.size,8),sharex=True)
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

            ## add reward zones
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
                plt.savefig(multipdf_object, format='pdf')

            fig.tight_layout()
            plt.show(block=False)
            
            if write_pdf==True:
                plt.close()
            
    

    def plot_popact(self, cellids, corridor=-1, name_string='selected_cells', bylaps=False):
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
                n_laps = popact_laps.shape[1]

                ax[0,cor_index].set_xlim(0, 51)
                ax[0,cor_index].set_title(title_string)

                ax[0,0].set_ylim(0, ymax)

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


    def plot_session(self, selected_laps=None):
        ## plot the behavioral data during one session. 
            # - speed
            # - lick rate
        ## selected laps: numpy array indexing the laps to be included in the plot

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
            fig, axs = plt.subplots(nrows=nrow, ncols=1, figsize=(8,rowHeight*nrow), squeeze=False)
            # plt.figure(figsize=(5,2*nrow))
            speed_color = cmap(30)
            speed_color_trial = (speed_color[0], speed_color[1], speed_color[2], (0.05))

            lick_color = cmap(200)
            lick_color_trial = (lick_color[0], lick_color[1], lick_color[2], (0.05))

            for row in range(nrow):
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
                            axs[row,0].step(self.ImLaps[lap].bincenters, self.ImLaps[lap].ave_speed, where='mid', c=speed_color_trial)
                            speed_matrix[i_lap,:] =  np.round(self.ImLaps[lap].ave_speed, 2)
                            nans_lap = np.isnan(self.ImLaps[lap].ave_speed)
                            avespeed = nan_add(avespeed, self.ImLaps[lap].ave_speed)
                            n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                            if (max(self.ImLaps[lap].ave_speed) > maxspeed): maxspeed = max(self.ImLaps[lap].ave_speed)
                            n_correct = n_correct + self.ImLaps[lap].correct
                            n_valid = n_valid + 1
                        i_lap = i_lap + 1
                    maxspeed = min(maxspeed, 60)
                    P_correct = np.round(np.float(n_correct) / np.float(n_valid), 3)

                    avespeed = nan_divide(avespeed, n_lap_bins, n_lap_bins > 0)
                    axs[row,0].step(self.ImLaps[lap].bincenters, avespeed, where='mid', c=speed_color)
                    axs[row,0].set_ylim([-1,1.2*maxspeed])

                    if (row == 0):
                        if (self.sessionID >= 0):
                            plot_title = 'session:' + str(self.sessionID) + ': ' + str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)
                        else:
                            plot_title = str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)
                    else:
                        plot_title = str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)

                    ########################################
                    ## reward zones

                    if (self.ImLaps[lap].zones.shape[1] > 0):
                        bottom, top = axs[row,0].get_ylim()
                        left = self.ImLaps[lap].zones[0,0] * self.corridor_length_roxel
                        right = self.ImLaps[lap].zones[1,0] * self.corridor_length_roxel

                        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                        axs[row,0].add_patch(polygon)
                        n_zones = np.shape(self.ImLaps[lap].zones)[1]
                        if (n_zones > 1):
                            for i in range(1, np.shape(self.ImLaps[lap].zones)[1]):
                                left = self.ImLaps[lap].zones[0,i] * self.corridor_length_roxel
                                right = self.ImLaps[lap].zones[1,i] * self.corridor_length_roxel
                                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                                axs[row,0].add_patch(polygon)
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
                    ax2 = axs[row,0].twinx()
                    n_lap_bins = np.zeros(nbins) # number of laps in a given bin (data might be NAN for some laps)
                    maxrate = 10
                    avelick = np.zeros(nbins)

                    lick_matrix = np.zeros((len(ids), nbins))
                    i_lap = 0

                    for lap in ids:
                        if (self.ImLaps[lap].mode == 1): # only use the lap if it was a valid lap
                            ax2.step(self.ImLaps[lap].bincenters, self.ImLaps[lap].lick_rate, where='mid', c=lick_color_trial, linewidth=1)
                            lick_matrix[i_lap,:] =  np.round(self.ImLaps[lap].lick_rate, 2)
                            nans_lap = np.isnan(self.ImLaps[lap].lick_rate)
                            avelick = nan_add(avelick, self.ImLaps[lap].lick_rate)
                            n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                            if (np.nanmax(self.ImLaps[lap].lick_rate) > maxrate): maxrate = np.nanmax(self.ImLaps[lap].lick_rate)
                        i_lap = i_lap + 1
                    maxrate = min(maxrate, 20)

                    avelick = nan_divide(avelick, n_lap_bins, n_lap_bins > 0)
                    ax2.step(self.ImLaps[lap].bincenters, avelick, where='mid', c=lick_color)
                    ax2.set_ylim([-1,1.2*maxrate])


                    if (row==(nrow-1)):
                        axs[row,0].set_ylabel('speed (cm/s)', color=speed_color)
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


    def save_data(self, save_properties=True, save_ratemaps=True, save_laptime=True):
        # Saves the primary data into a folder in csv format.
        # separate file is created for each corridor and lap.
        # 
        # save_properties: True or False. If True, the cell properties are saved. 
        # save_ratemaps: True or False. If True, the ratemaps are saved.
        # save_laptime: True or False. If True, the raw data is saved for each lap.

        data_folder = self.suite2p_folder + 'analysed_data'
        if (self.elfiz == True):
            data_folder = self.suite2p_folder + 'analysed_data_' + self.imaging_logfile_name

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
            for i_cor in np.arange(N_corridors):
                ratemap_file = 'ratemaps_corridor_' + str(self.corridors[i_cor]) + '_N' + str(self.N_cells) + '.csv'
                self.write_params(ratemap_file)
                filename = data_folder + '/' + ratemap_file
                with open(filename, mode='a') as rate_file:
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
                with open(filename, mode='a') as lap_file:
                    file_writer = csv.writer(lap_file, **csv_kwargs)
                    file_writer.writerow(('rows: time, position, speed, lick, reward and spikes of the ' + str(self.N_cells) + ' cells;', 'columns: time bins'))
                    for i_row in np.arange(lapdata.shape[0]):
                        file_writer.writerow(np.round(lapdata[i_row,:], 4))
            print('lapdata saved into file: ' + filename)


class Lap_ImData:
    'common base class for individual laps'

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, lap_frames_dF_F, lap_frames_spikes, lap_frames_pos, lap_frames_time, corridor_list, printout=False, speed_threshold=5, elfiz=False, verbous=0):
        if (verbous > 0):
            print('ImData initialised')
        self.name = name
        self.lap = lap

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

        # approximate frame period for imaging - 0.033602467
        # only use it to convert spikes to rates and to prepare uniform time axis!
        if (self.elfiz==True):
            self.dt_imaging = 0.0002 # s - 5000 Hz
        else: 
            self.dt_imaging = 0.033602467 # s - 33 Hz

        self.frames_dF_F = lap_frames_dF_F
        self.frames_spikes = lap_frames_spikes
        self.frames_pos = lap_frames_pos
        self.frames_time = lap_frames_time

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

        if (verbous > 0):
            print('lick and reward position calculated')
        ##################################################################################
        ## speed vs. time
        ##################################################################################
        self.imaging_data = True

        if (np.isnan(self.frames_time).any()): # we don't have imaging data
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

        if (verbous > 0):
            print('speed calculated')
        ##################################################################################
        ## speed, lick and spiking vs. position
        ##################################################################################

        ####################################################################
        ## calculate the lick-rate and the average speed versus location    
        bin_counts = np.zeros(self.N_pos_bins)
        fast_bin_counts = np.zeros(self.N_pos_bins)
        total_speed = np.zeros(self.N_pos_bins)

        for i_frame in range(len(self.frames_pos)):
            bin_number = int(self.frames_pos[i_frame] // 70)
            bin_counts[bin_number] += 1
            if (self.frames_speed[i_frame] > self.speed_threshold):
                fast_bin_counts[bin_number] += 1
            total_speed[bin_number] = total_speed[bin_number] + self.frames_speed[i_frame]

        self.T_pos = bin_counts * self.dt_imaging           # used for lick rate and average speed
        self.T_pos_fast = fast_bin_counts * self.dt_imaging # used for spike rate calculations

        total_speed = total_speed * self.dt_imaging
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
            self.n_cells = self.frames_dF_F.shape[0]
            self.spks_pos = np.zeros((self.n_cells, self.N_pos_bins)) # sum of spike counts measured at a given position
            self.event_rate = np.zeros((self.n_cells, self.N_pos_bins)) # spike rate 

            for i_frame in range(len(self.frames_pos)):
                bin_number = int(self.frames_pos[i_frame] // 70)
                if (self.frames_speed[i_frame] > self.speed_threshold):
                    if (self.elfiz):
                        self.spks_pos[:,bin_number] = self.spks_pos[:,bin_number] + self.frames_spikes[:,i_frame]
                    else: ### we need to multiply the values with dt_imaging as this converts probilities to expected counts
                        self.spks_pos[:,bin_number] = self.spks_pos[:,bin_number] + self.frames_spikes[:,i_frame] * self.dt_imaging
            for bin_number in range(self.N_pos_bins):
                if (self.T_pos_fast[bin_number] > 0): # otherwise the rate will remain 0
                    self.event_rate[:,bin_number] = self.spks_pos[:,bin_number] / self.T_pos_fast[bin_number]

        if (verbous > 0):
            print('ratemaps calculated')

        ####################################################################
        ## Calculate the lick rate befor the reward zone - anticipatory licks 210 roxels before zone start
        ## only when the number of zones is 1!

        if (self.n_zones == 1):

            zone_start = int(self.zones[0][0] * self.corridor_length_roxel)
            zone_end = int(self.zones[1][0] * self.corridor_length_roxel)
            if (len(self.lick_position) > 0):
                lz_posbins = np.array([np.min((np.min(self.frames_pos)-1, np.min(self.lick_position)-1, 0)), zone_start-420, zone_start-210, zone_start, zone_end, self.corridor_length_roxel])
            else :
                lz_posbins = np.array([np.min((np.min(self.frames_pos)-1, 0)), zone_start-420, zone_start-210, zone_start, zone_end, self.corridor_length_roxel])


            lz_bin_counts = np.zeros(5)
            for pos in self.frames_pos:
                bin_number = np.max(np.where(pos > lz_posbins))
                lz_bin_counts[bin_number] += 1
            T_lz_pos = lz_bin_counts * self.dt_imaging

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
        if (self.n_cells > 0):
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
        if (self.n_cells > 0):
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
        self.test = scipy.stats.wilcoxon(self.baseline, self.anti_rate)
        self.anti = False
        if ((self.test[1] < 0.01 ) & (greater == True)):
            self.anti = True


def HolmBonfMat(P_mat, p_val):
    ## P_mat is a matrix with the 
    ## P-values of different tests in each row and
    ## Cells in each column

    m = P_mat.shape[0]
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
    trigger_log_file=open(trigger_log_file_string)
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
    trigger_signal_file=open(TRIGGER_VOLTAGE_FILENAME, 'r')
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
