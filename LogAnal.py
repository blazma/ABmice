# -*- coding: utf-8 -*-
"""
Created in Mar 2019
@author: bbujfalussy - ubalazs317@gmail.com
A script to read behavioral log files in mouse in vivo virtual reality experiments

"""

import numpy as np
from string import *
import datetime
import time
import os
import pickle
import warnings
import scipy.stats
from scipy.interpolate import interp1d   
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import sys
from sys import version_info
import csv

from Stages import *
from Corridors import *

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

class Lap_Data:
    'common base class for individual laps'

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, corridor_list, printout=False, speed_threshold=5):
        self.name = name
        self.lap = lap

        self.correct = False
        self.raw_time = laptime
        self.raw_position = position
        self.lick_times = lick_times
        self.reward_times = reward_times
        self.corridor = corridor # the ID of the corridor in the given stage; This indexes the corridors in the vector called self.corridors
        self.corridor_list = corridor_list 
        self.mode = mode # 1 if all elements are recorded in 'Go' mode
        self.actions = actions
        
        self.speed_threshold = speed_threshold ## cm / s 106 cm - 3500 roxels; roxel/s * 106.5/3500 = cm/s
        self.corridor_length_roxel = (self.corridor_list.corridors[self.corridor].length - 1024.0) / (7168.0 - 1024.0) * 3500
        self.N_pos_bins = int(np.round(self.corridor_length_roxel / 70))
        self.speed_factor = 106.5 / 3500 ## constant to convert distance from pixel to cm
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor # cm

        self.zones = np.vstack([np.array(self.corridor_list.corridors[self.corridor].reward_zone_starts), np.array(self.corridor_list.corridors[self.corridor].reward_zone_ends)])
        self.n_zones = np.shape(self.zones)[1]
        self.preZoneRate = [None, None] # only if 1 lick zone; Compare the 210 roxels just before the zone with the preceeding 210 

        self.last_zone_start = max(self.corridor_list.corridors[self.corridor].reward_zone_starts)
        self.last_zone_end = max(self.corridor_list.corridors[self.corridor].reward_zone_ends)

        # approximate frame period for imaging - 0.033602467
        # only use it to prepare uniform time axis
        self.dt_imaging = 0.033602467
        self.bincenters = np.arange(0, self.corridor_length_roxel, 70) + 70 / 2.0

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
            
        ##################################################################################
        ## speed vs. time
        ##################################################################################

        start_time = np.ceil(self.raw_time.min()/self.dt_imaging)*self.dt_imaging
        end_time = np.floor(self.raw_time.max()/self.dt_imaging)*self.dt_imaging
        Ntimes = int(round((end_time - start_time) / self.dt_imaging)) + 1
        self.frames_time = np.linspace(start_time, end_time, Ntimes)
        self.frames_pos = F(self.frames_time)

        ## calculate the speed during the frames
        speed = np.diff(self.frames_pos) * self.speed_factor / self.dt_imaging # cm / s
        try:
            speed_first = 2 * speed[0] - speed[1] # linear extrapolation: x1 - (x2 - x1)
            
        except IndexError:
            speed_first = np.nan
        self.frames_speed = np.hstack([speed_first, speed])

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
                

    def plot_tx(self):
        cmap = plt.cm.get_cmap('jet')   
        plt.figure(figsize=(6,4))
        plt.plot(self.frames_time, self.frames_pos, color=cmap(50))
        plt.plot(self.raw_time, self.raw_position, color=cmap(90))

        plt.scatter(self.lick_times, np.repeat(self.frames_pos.min(), len(self.lick_times)), marker="|", s=100, color=cmap(180))
        plt.scatter(self.reward_times, np.repeat(self.frames_pos.min()+100, len(self.reward_times)), marker="|", s=100, color=cmap(230))
        plt.ylabel('position')
        plt.xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        plt.title(plot_title)
        plt.ylim(0, self.corridor_length_roxel)

        plt.show(block=False)
       
        # time = mm.Laps[55].time
        # frames_pos = mm.Laps[55].frames_pos
        # lick_times = mm.Laps[55].lick_times
        # reward_times = mm.Laps[55].reward_times
        # lap = mm.Laps[55].lap
        # corridor = mm.Laps[55].corridor
        # lick_rate = mm.Laps[55].lick_rate
        # bincenters = np.arange(0, 3500, 175) + 175 / 2.0

        # plt.figure(figsize=(6,4))
        # plt.plot(laptime, frames_pos, c='g')

        # plt.scatter(lick_times, np.repeat(frames_pos.min(), len(lick_times)), marker="|", s=100)
        # plt.scatter(reward_times, np.repeat(frames_pos.min()+100, len(reward_times)), marker="|", s=100, c='r')
        # plt.ylabel('position')
        # plt.xlabel('time (s)')
        # plot_title = 'Mouse: ' + name + ' position in lap ' + str(lap) + ' in corridor ' + str(corridor)
        # plt.title(plot_title)

        # plt.show(block=False)

    def plot_xv(self):
        cmap = plt.cm.get_cmap('jet')   

        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(self.frames_pos, self.frames_speed, c=cmap(80))
        plt.step(self.bincenters, self.ave_speed, where='mid', c=cmap(30))
        plt.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, color=cmap(180))
        plt.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, color=cmap(230))
        plt.ylabel('speed (cm/s)')
        plt.ylim([min(0, self.frames_speed.min()), max(self.frames_speed.max(), 30)])
        plt.xlabel('position')
        plot_title = 'Mouse: ' + self.name + ' speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        plt.title(plot_title)


        bottom, top = plt.ylim()
        left = self.zones[0,0] * self.corridor_length_roxel
        right = self.zones[1,0] * self.corridor_length_roxel

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color='green', alpha=0.15)
        ax.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * self.corridor_length_roxel
                right = self.zones[1,i] * self.corridor_length_roxel
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed= True, color='green', alpha=0.15)
                ax.add_patch(polygon)

        ax2 = plt.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', c=cmap(200), linewidth=1)
        ax2.set_ylabel('lick rate (lick/s)', color=cmap(200))
        ax2.tick_params(axis='y', labelcolor=cmap(200))
        ax2.set_ylim([-1,max(2*np.nanmax(self.lick_rate), 20)])

        plt.show(block=False)       


        # cmap = plt.cm.get_cmap('jet')   
        # frames_pos = mm.Laps[55].frames_pos
        # speed = mm.Laps[55].speed
        # lick_position = mm.Laps[55].lick_position
        # lick_times = mm.Laps[55].lick_times
        # reward_position = mm.Laps[55].reward_position
        # reward_times = mm.Laps[55].reward_times
        # lap = mm.Laps[55].lap
        # corridor = mm.Laps[55].corridor
        # lick_rate = mm.Laps[55].lick_rate
        # ave_speed = mm.Laps[55].ave_speed
        # zones = mm.Laps[0].zones
        # bincenters = np.arange(0, 3500, 175) + 175 / 2.0

        # fig, ax = plt.subplots(figsize=(6,4))
        # ax.plot(frames_pos, speed, c=cmap(80))
        # ax.plot(bincenters, ave_speed, c=cmap(30))
        # ax.scatter(lick_position, np.repeat(speed.min(), len(lick_position)), marker="|", s=100, c=cmap(180))
        # ax.scatter(reward_position, np.repeat(speed.min(), len(reward_position)), marker="|", s=100, c=cmap(230))
        # plt.ylabel('speed (roxel/s)')
        # plt.xlabel('position')
        # plot_title = 'Mouse: ' + name + ' speed in lap ' + str(lap) + ' in corridor ' + str(corridor)
        # plt.title(plot_title)

        # bottom, top = plt.ylim()
        # left = zones[0,0] * 3500
        # right = zones[1,0] * 3500

        # polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color='green', alpha=0.15)
        # if (np.shape(zones)[1] > 1):
        #     for i in range(1, np.shape(zones)[1]):
        #         left = zones[0,i] * 3500
        #         right = zones[1,i] * 3500
        #         polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color='green', alpha=0.15)
        #         ax.add_patch(polygon)


        # ax2 = plt.twinx()
        # ax2.plot(bincenters, lick_rate, c=cmap(200), linewidth=1)
        # ax2.set_ylabel('lick rate', color=cmap(200))
        # ax2.tick_params(axis='y', labelcolor=cmap(200))
        # ax2.set_ylim([-1,2*max(lick_rate)])

        # plt.show(block=False)       


    def plot_txv(self):
        cmap = plt.cm.get_cmap('jet')   
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6,6))

        ## first, plot position versus time
        ax_top.plot(self.frames_time, self.frames_pos, color=cmap(50))
        ax_top.plot(self.raw_time, self.raw_position, color=cmap(90))

        ax_top.scatter(self.lick_times, np.repeat(200, len(self.lick_times)), marker="|", s=100, color=cmap(180))
        ax_top.scatter(self.reward_times, np.repeat(400, len(self.reward_times)), marker="|", s=100, color=cmap(230))
        ax_top.set_ylabel('position')
        ax_top.set_xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position and speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)
        ax_top.set_ylim(0, self.corridor_length_roxel + 100)


        ## next, plot speed versus position
        ax_bottom.plot(self.frames_pos, self.frames_speed, color=cmap(80))
        ax_bottom.step(self.bincenters, self.ave_speed, where='mid', color=cmap(30))
        ax_bottom.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, color=cmap(180))
        ax_bottom.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, color=cmap(230))
        ax_bottom.set_ylabel('speed (cm/s)')
        ax_bottom.set_xlabel('position')
        ax_bottom.set_ylim([min(0, self.frames_speed.min()), max(self.frames_speed.max(), 30)])

        bottom, top = plt.ylim()
        left = self.zones[0,0] * self.corridor_length_roxel
        right = self.zones[1,0] * self.corridor_length_roxel

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color='green', alpha=0.15)
        ax_bottom.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * self.corridor_length_roxel
                right = self.zones[1,i] * self.corridor_length_roxel
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color='green', alpha=0.15)
                ax_bottom.add_patch(polygon)

        ax2 = ax_bottom.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', color=cmap(180), linewidth=1)
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
        if ((self.m_base > 0) & (self.m_anti > 0)):
            self.test = scipy.stats.wilcoxon(self.baseline, self.anti_rate)
            if ((self.test[1] < 0.01 ) & (greater == True)):
                self.anti = True
        else:
            self.test = [0, 1]


class Session:
    'common base class for low level position and licksensor data in a given session'

    def __init__(self, datapath, date_time, name, task, sessionID=-1, printout=False):
        self.datapath = datapath
        self.date_time = date_time
        self.name = name
        self.task = task

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

        ## in certain tasks, the same corridor may appear multiple times in different substages
        ## Labview uses different indexes for corridors in different substages, therefore 
        ## we need to keep this corridor in the list self.corridors for running self.get_lapdata()
        ## but we should remove the redundancy after the data is loaded

        self.get_stage(datapath, date_time, name, task)
        self.corridors = np.hstack([0, np.array(self.stage_list.stages[self.stage].corridors)])

        self.last_zone_start = 0
        self.last_zone_end = 0
        for i_corridor in self.corridors:
            if (i_corridor > 0):
                if (max(self.corridor_list.corridors[i_corridor].reward_zone_starts) > self.last_zone_start):
                    self.last_zone_start = max(self.corridor_list.corridors[i_corridor].reward_zone_starts)
                if (max(self.corridor_list.corridors[i_corridor].reward_zone_ends) > self.last_zone_end):
                    self.last_zone_end = max(self.corridor_list.corridors[i_corridor].reward_zone_ends)
                    
        self.substage_change_laps = [0]
        self.speed_factor = 106.5 / 3500.0 ## constant to convert distance from pixel to cm
        self.corridor_length_roxel = (self.corridor_list.corridors[self.corridors[1]].length - 1024.0) / (7168.0 - 1024.0) * 3500
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor # cm
        self.N_pos_bins = int(np.round(self.corridor_length_roxel / 70))


        self.Laps = [] # list containing a special class for storing the behavioral data for single laps
        self.n_laps = 0
        self.i_corridors = np.zeros(1) # np array with the index of corridors in each run

        self.get_lapdata(datapath, date_time, name, task)
        if (self.n_laps == -1):
            print('Error: missing laps are found in the ExpStateMachineLog file! No analysis was performed, check the logfiles!')
            return

        self.test_anticipatory()

        # behavior score -- call calc_behavior_score to fill up
        self.Ps_correct = {}  # corridor_id : P_correct
        self.VSEL_NORMALIZATION = -0.45  # speed selectivity normalization constant: negative so that speed selectivity is "good" if close to 1 instead of -1, just like lick
        self.speed_index = {}
        self.lick_index = {}
        self.speed_selectivity = np.nan
        self.lick_selectivity = np.nan
        self.behavior_score = None
        self.behavior_score_components = {}

        self.preRZ = {}  # pre reward zone
        self.ctrlZ = {}  # control zone (1000 roxels before preRZ)
        self.crossZ = [] # cross-corridor zone (for selectivities)

    def get_lapdata(self, datapath, date_time, name, task):

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
        time_breaks = np.where(np.diff(laptime) > 1)[0]
        if (len(time_breaks) > 0):
            print('ExpStateMachineLog time interval > 1s: ', len(time_breaks), ' times')
            print(laptime[time_breaks])

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
        i_corrids = [] # ID of corridor for the current lap
        self.n_laps = 0

        ### include only a subset of laps

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
    
                lick_lap = lick[y]
                t_licks = t_lap[lick_lap]
    
                sstage_lap = np.unique(sstage[y])
                if (len(sstage_lap) > 1):
                    print('More than one substage in lap ', self.n_laps)
                    return 

                if (sstage_lap != current_sstage):
                    print('############################################################')
                    print('substage change detected!')
                    print('first lap in substage ', sstage_lap, 'is lap', self.n_laps, ', which started at t', t_lap[0])
                    print('############################################################')
                    current_sstage = sstage_lap
                    self.substage_change_laps.append(self.n_laps)

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
    
                # sessions.append(Lap_Data(name, i, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions))
				# print(self.n_laps, i_lap, len(pos_lap))
                self.Laps.append(Lap_Data(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, self.corridor_list))
                self.n_laps = self.n_laps + 1
            else:
	            N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1)
        
        self.i_corridors = np.array(i_corrids) # ID of corridor for the current lap

    def get_stage(self, datapath, date_time, name, task):
        action_log_file_string=datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_UserActionLog.txt'
        action_log_file=open(action_log_file_string)
        log_file_reader=csv.reader(action_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            if (line[1] == 'Stage'):
                self.stage = int(round(float(line[2])))

    def test_anticipatory(self):
        corridor_types = np.unique(self.i_corridors)
        nrow = len(corridor_types)
        self.anticipatory = []

        for row in range(nrow):
            ids = np.where(self.i_corridors == corridor_types[row])
            n_laps = np.shape(ids)[1]
            n_zones = np.shape(self.Laps[ids[0][0]].zones)[1]
            if ((n_zones == 1) & (n_laps > 2)):
                lick_rates = np.zeros([2,n_laps])
                k = 0
                for lap in np.nditer(ids):
                    if (self.Laps[lap].mode == 1):
                        lick_rates[:,k] = self.Laps[lap].preZoneRate
                    else:
                        lick_rates[:,k] = np.nan
                    k = k + 1
                self.anticipatory.append(anticipatory_Licks(lick_rates[0,:], lick_rates[1,:], corridor_types[row]))

    def calc_correct_lap_proportions(self, corrA, corrB, selected_laps=None):
        if selected_laps is None:
            selected_laps = np.arange(self.n_laps)
        if (self.n_laps > 0):
            for corridor in [corrA, corrB]: # for each corridor...
                if corridor == corrB and corrB is None:
                    self.Ps_correct[corridor] = np.nan
                    continue
                ids_all = np.where(self.i_corridors == corridor)
                ids = np.intersect1d(ids_all, selected_laps)
                if (len(ids) > 2):
                    n_correct = 0
                    n_valid = 0
                    for lap in ids:
                        if self.Laps[lap].mode != 1:  # only valid laps should be included
                            continue
                        n_correct = n_correct + self.Laps[lap].correct
                        n_valid = n_valid + 1
                    P_correct = np.round(nan_divide(float(n_correct), float(n_valid)),3)
                    self.Ps_correct[corridor] = P_correct

    def calc_speed_and_lick_selectivity(self, corrA, corrB, selected_laps=None):
        print("calculating speed and lick selectivities, behavior score...")
        if selected_laps is None:
            selected_laps = np.arange(self.n_laps)
        if (self.n_laps > 0):
            nbins = len(self.Laps[0].bincenters)

            avg_speeds_by_corridor = {}
            avg_licks_by_corridor = {}
            for i_cor, corridor in enumerate([corrA, corrB]):
                if corridor == corrB and corrB is None:
                    self.preRZ[corrB] = [np.nan, np.nan]
                    self.ctrlZ[corrB] = [np.nan, np.nan]
                    self.crossZ = [np.nan, np.nan]
                    self.speed_index[corrB] = np.nan
                    self.lick_index[corrB] = np.nan
                    continue
                ids_all = np.where(self.i_corridors == corridor)
                ids = np.intersect1d(ids_all, selected_laps)

                if (len(ids) > 2):
                    ########################################
                    ## speed
                    speed_matrix = np.zeros((len(ids), nbins))
                    i_lap = 0
                    for lap in ids:
                        if (self.Laps[lap].mode == 1): # only use the lap if it was a valid lap
                            speed_matrix[i_lap,:] =  np.round(self.Laps[lap].ave_speed, 2)
                        else:
                            speed_matrix[i_lap,:] =  np.nan
                        i_lap = i_lap + 1

                    ########################################
                    ## lick
                    lick_matrix = np.zeros((len(ids), nbins))
                    i_lap = 0
                    for lap in ids:
                        if (self.Laps[lap].mode == 1): # only use the lap if it was a valid lap
                            lick_matrix[i_lap,:] =  np.round(self.Laps[lap].lick_rate, 2)
                        i_lap = i_lap + 1

                    # 5 bins long region right before reward zone
                    RZ_start_bin = np.round(self.corridor_list.corridors[corridor].reward_zone_starts * self.N_pos_bins)
                    preRZ_ub = int(RZ_start_bin) - 1
                    preRZ_lb = preRZ_ub - 5
                    roxel_per_bin = self.corridor_length_roxel / self.N_pos_bins
                    self.preRZ[corridor] = [preRZ_lb * roxel_per_bin, preRZ_ub * roxel_per_bin]

                    # 5 bins long control region 1000 roxels before preRZ region
                    ctrl_ub = preRZ_lb - int(np.round(1000 / self.corridor_length_roxel * self.N_pos_bins))
                    ctrl_lb = ctrl_ub - 5
                    self.ctrlZ[corridor] = [ctrl_lb * roxel_per_bin, ctrl_ub * roxel_per_bin]

                    # within-corridor selectivities
                    with warnings.catch_warnings(action="ignore"):
                        avg_speed_preRZ = np.nanmean(speed_matrix[:, preRZ_lb:preRZ_ub], axis=1)
                        avg_speed_ctrl = np.nanmean(speed_matrix[:,ctrl_lb:ctrl_ub], axis=1)
                        speed_selectivity_laps = nan_divide(avg_speed_preRZ - avg_speed_ctrl,
                                                            avg_speed_preRZ + avg_speed_ctrl)
                        self.speed_index[corridor] = speed_selectivity_laps

                        avg_lick_preRZ = np.nanmean(lick_matrix[:, preRZ_lb:preRZ_ub], axis=1)
                        avg_lick_ctrl = np.nanmean(lick_matrix[:, ctrl_lb:ctrl_ub], axis=1)
                        lick_selectivity_laps = nan_divide(avg_lick_preRZ - avg_lick_ctrl,
                                                            avg_lick_preRZ + avg_lick_ctrl)
                        self.lick_index[corridor] = lick_selectivity_laps

                    # cross-corridor selectivities
                    RZ_corrA_start_bin = np.round(self.corridor_list.corridors[corrA].reward_zone_starts * self.N_pos_bins)
                    corrA_preRZ_ub = int(RZ_corrA_start_bin) - 1
                    corrA_preRZ_lb = corrA_preRZ_ub - 5
                    self.crossZ = [corrA_preRZ_lb * roxel_per_bin, corrA_preRZ_ub * roxel_per_bin]

                    avg_speed_corrA_preRZ = np.nanmean(speed_matrix[:, corrA_preRZ_lb:corrA_preRZ_ub])
                    avg_speeds_by_corridor[corridor] = avg_speed_corrA_preRZ
                    avg_lick_corrA_preRZ = np.nanmean(lick_matrix[:, corrA_preRZ_lb:corrA_preRZ_ub])
                    avg_licks_by_corridor[corridor] = avg_lick_corrA_preRZ
            with warnings.catch_warnings(action="ignore"):  # nan_divide throws RuntimeWarning but it works as
                if corrB is not None:
                    self.speed_selectivity = nan_divide(avg_speeds_by_corridor[corrA] - avg_speeds_by_corridor[corrB],
                                                        avg_speeds_by_corridor[corrA] + avg_speeds_by_corridor[corrB])
                    self.lick_selectivity = nan_divide(avg_licks_by_corridor[corrA] - avg_licks_by_corridor[corrB],
                                                       avg_licks_by_corridor[corrA] + avg_licks_by_corridor[corrB])
                else:
                    self.speed_selectivity = np.nan
                    self.lick_selectivity = np.nan

    def calc_behavior_score(self, corrA, corrB=None, selected_laps=None):
        if selected_laps is None:
            selected_laps = np.arange(self.n_laps)
        corridor_types = np.unique(self.i_corridors[selected_laps])
        if corrB is not None:
            if corrA not in corridor_types and corrB not in corridor_types:
                raise Exception("corridors"+str(corrA)+" and/ or"+str(corrB)+" not found in session, aborting behavior score calc.")
        else:
            if corrA not in corridor_types:
                raise Exception("corridor"+str(corrA)+" not found in session, aborting behavior score calc.")

        self.calc_correct_lap_proportions(corrA, corrB, selected_laps)
        self.calc_speed_and_lick_selectivity(corrA, corrB, selected_laps)

        mean_vidx_A = np.round(np.nanmean(self.speed_index[corrA]), 2)
        mean_vidx_B = np.round(np.nanmean(self.speed_index[corrB]), 2)
        mean_lidx_A = np.round(np.nanmean(self.lick_index[corrA]), 2)
        mean_lidx_B = np.round(np.nanmean(self.lick_index[corrB]), 2)
        self.behavior_score_components = {
            "P correct ("+str(corrA)+")": self.Ps_correct[corrA],
            "P correct ("+str(corrB)+")": self.Ps_correct[corrB],
            "Speed index ("+str(corrA)+")": mean_vidx_A / self.VSEL_NORMALIZATION,
            "Speed index ("+str(corrB)+")": mean_vidx_B / self.VSEL_NORMALIZATION,
            "Speed selectivity": float(self.speed_selectivity / self.VSEL_NORMALIZATION),
            "Lick index ("+str(corrA)+")": mean_lidx_A,
            "Lick index ("+str(corrB)+")": mean_lidx_B,
            "Lick selectivity": float(self.lick_selectivity),
        }
        self.behavior_score = sum(list(self.behavior_score_components.values()))

    def plot_session(self, corrA=None, corrB=None, selected_laps=None, filename=None):
        self.plot_session_engine(corrA = corrA, corrB = corrB, selected_laps=selected_laps, average=True, filename=None)
        self.plot_session_engine(corrA = corrA, corrB = corrB, selected_laps=selected_laps, average=False, filename=None)    

    def plot_session_engine(self, corrA=None, corrB=None, selected_laps=None, average = True, filename=None):
        # corrA, corrB are needed for behavior score calculation
        ## find the number of different corridors
        if (selected_laps is None):
            selected_laps = np.arange(self.n_laps)
            add_anticipatory_test = True
        else:
            add_anticipatory_test = False

        if (self.n_laps > 0):
            corridor_types = np.unique(self.i_corridors[selected_laps])
            nrow = len(corridor_types)
            nbins = len(self.Laps[0].bincenters)
            cmap = plt.cm.get_cmap('jet')   

            rowHeight = 2
            if (nrow > 4):
                rowHeight = 1.5
                
            if (average):
                reward_zone_color = 'green'
                if self.behavior_score is not None:
                    fig, axs = plt.subplots(nrows=nrow, ncols=2, figsize=(8,rowHeight*nrow), squeeze=False, width_ratios=[2,1])
                else:
                    fig, axs = plt.subplots(nrows=nrow, ncols=1, figsize=(8,rowHeight*nrow), squeeze=False)
            else:
                reward_zone_color = 'fuchsia'
                fig, axs = plt.subplots(nrows=nrow, ncols=1, figsize=(8,rowHeight*nrow*3), squeeze=False)

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
                        if (self.Laps[lap].mode == 1): # only use the lap if it was a valid lap
                            if (average):
                                axs[row,0].step(self.Laps[lap].bincenters, self.Laps[lap].ave_speed, where='mid', c=speed_color_trial)
                            speed_matrix[i_lap,:] =  np.round(self.Laps[lap].ave_speed, 2)
                            nans_lap = np.isnan(self.Laps[lap].ave_speed)
                            avespeed = nan_add(avespeed, self.Laps[lap].ave_speed)
                            n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                            if (max(self.Laps[lap].ave_speed) > maxspeed): maxspeed = max(self.Laps[lap].ave_speed)
                            n_correct = n_correct + self.Laps[lap].correct
                            n_valid = n_valid + 1
                        else:
                            speed_matrix[i_lap,:] =  np.nan
                        i_lap = i_lap + 1
                    maxspeed = min(maxspeed, 60)
                    P_correct = np.round(float(n_correct) / float(n_valid), 3)

                    if (average):
                        avespeed = nan_divide(avespeed, n_lap_bins, n_lap_bins > 0)
                        axs[row,0].step(self.Laps[lap].bincenters, avespeed, where='mid', c=speed_color)
                        axs[row,0].set_ylim([-1,1.2*maxspeed])
                    else:
                        im = axs[row,0].imshow(speed_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=maxspeed, extent=(0, self.corridor_length_roxel, 0, i_lap), origin='lower')
                        plt.colorbar(im, orientation='vertical',ax=axs[row, 0])

                    if (row == 0):
                        if (self.sessionID >= 0):
                            plot_title = 'session:' + str(self.sessionID) + ': ' + str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)
                        else:
                            plot_title = str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)
                    else:
                        plot_title = str(int(n_laps)) + ' (' + str(int(n_correct)) + ')' + ' laps in corridor ' + str(int(corridor_types[row])) + ', P-correct: ' + str(P_correct)

                    if self.behavior_score is not None and average == True:
                        if corridor_types[row] == corrA:
                            cells = [["P correct ("+str(corrA)+")", np.round(self.behavior_score_components[f'P correct ('+str(corrA)+")"],2),],
                                     ["Speed index ("+str(corrA)+")", np.round(self.behavior_score_components[f'Speed index ('+str(corrA)+")"],2),],
                                     ["Lick index ("+str(corrA)+")", np.round(self.behavior_score_components[f'Lick index ('+str(corrA)+")"],2)]]
                            cell_colors = [["white", "white"],
                                           ["white", "white"],
                                           ["white", "white"],]
                            table = axs[row,1].table(cellText=cells,
                                              colLabels=["component", "score"],
                                              cellColours=cell_colors,
                                              colColours=["beige", "beige"],
                                              loc="center")
                            table.auto_set_font_size(False)
                            table.auto_set_column_width([0,1])
                            axs[row,1].axis('off')

                            ymin, ymax = axs[row,0].get_ylim()
                            axs[row, 0].plot(self.preRZ[corrA], [ymax-ymax*0.05, ymax-ymax*0.05], linewidth=3, color="red")
                            axs[row, 0].plot(self.ctrlZ[corrA], [ymax - ymax * 0.05, ymax - ymax * 0.05], linewidth=3, color="red")
                            axs[row, 0].plot(self.crossZ, [ymax - ymax * 0.10, ymax - ymax * 0.10], linewidth=3, color="green")
                        elif corrB is not None and corridor_types[row] == corrB:
                            cells = [["P correct ("+str(corrB)+")", np.round(self.behavior_score_components[f'P correct ('+str(corrB)+")"],2),],
                                     ["Speed index ("+str(corrB)+")", np.round(self.behavior_score_components[f'Speed index ('+str(corrB)+")"],2),],
                                     ["Lick index ("+str(corrB)+")", np.round(self.behavior_score_components[f'Lick index ('+str(corrB)+")"],2)],
                                     ["Speed selectivity", np.round(self.behavior_score_components['Speed selectivity'],2)],
                                     ["Lick selectivity", np.round(self.behavior_score_components['Lick selectivity'],2)],
                                     ["BEHAVIOR SCORE", np.round(self.behavior_score,2)]]
                            cell_colors = [["white", "white"],
                                           ["white", "white"],
                                           ["white", "white"],
                                           ["white", "white"],
                                           ["white", "white"],
                                           ["aquamarine", "aquamarine"],]
                            table = axs[row,1].table(cellText=cells,
                                              colLabels=["component", "score"],
                                              cellColours=cell_colors,
                                              colColours=["beige", "beige"],
                                              loc="center")
                            table.auto_set_font_size(False)
                            table.auto_set_column_width([0,1])
                            axs[row,1].axis('off')

                            ymin, ymax = axs[row, 0].get_ylim()
                            axs[row, 0].plot(self.preRZ[corrB], [ymax-ymax*0.05, ymax-ymax*0.05], linewidth=3, color="red")
                            axs[row, 0].plot(self.ctrlZ[corrB], [ymax - ymax * 0.05, ymax - ymax * 0.05], linewidth=3, color="red")
                            axs[row, 0].plot(self.crossZ, [ymax - ymax * 0.10, ymax - ymax * 0.10], linewidth=3, color="green")
                        else:
                            axs[row,1].set_visible(False)

                    ########################################
                    ## reward zones

                    if (self.Laps[lap].zones.shape[1] > 0):
                        bottom, top = axs[row,0].get_ylim()
                        left = np.round(self.Laps[lap].zones[0,0] * self.corridor_length_roxel, -1) - 4.5 # threshold of the position rounded to 10s - this is what LabView does
                        right = np.round(self.Laps[lap].zones[1,0] * self.corridor_length_roxel, -1) - 4.5

                        if (average):
                            polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color=reward_zone_color, alpha=0.15)
                            axs[row,0].add_patch(polygon)
                        else :
                            axs[row,0].vlines((left, right), ymin=bottom, ymax=top, colors=reward_zone_color, lw=3)
                        n_zones = np.shape(self.Laps[lap].zones)[1]
                        if (n_zones > 1):
                            for i in range(1, n_zones):
                                left = np.round(self.Laps[lap].zones[0,i] * self.corridor_length_roxel, -1) - 4.5 # threshold of the position rounded to 10s - this is what LabView does
                                right = np.round(self.Laps[lap].zones[1,i] * self.corridor_length_roxel, -1) - 4.5
                                if (average):
                                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), closed = True, color=reward_zone_color, alpha=0.15)
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
                        if (self.Laps[lap].mode == 1): # only use the lap if it was a valid lap
                            if (average): 
                                ax2.step(self.Laps[lap].bincenters, self.Laps[lap].lick_rate, where='mid', c=lick_color_trial, linewidth=1)
                            else:
                                if (len(self.Laps[lap].reward_times) > 0):
                                    axs[row,0].plot(self.Laps[lap].reward_position, np.ones(len(self.Laps[lap].reward_position)) * i_lap + 0.5, 'o', ms=4, color='deepskyblue')
                                if (len(self.Laps[lap].lick_times) > 0):
                                    axs[row,0].plot(self.Laps[lap].lick_position, np.ones(len(self.Laps[lap].lick_position)) * i_lap + 0.5, 'oC1', ms=1)
                                    if (len(self.substage_change_laps) > i_sstage):
                                        if (lap >= self.substage_change_laps[i_sstage]):
                                            axs[row,0].hlines(i_lap, xmin=0, xmax=self.corridor_length_roxel, colors='crimson', lw=3)
                                            i_sstage = i_sstage + 1
                            lick_matrix[i_lap,:] =  np.round(self.Laps[lap].lick_rate, 2)
                            nans_lap = np.isnan(self.Laps[lap].lick_rate)
                            avelick = nan_add(avelick, self.Laps[lap].lick_rate)
                            n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                            if (np.nanmax(self.Laps[lap].lick_rate) > maxrate): maxrate = np.nanmax(self.Laps[lap].lick_rate)
                        i_lap = i_lap + 1
                    maxrate = min(maxrate, 20)

                    avelick = nan_divide(avelick, n_lap_bins, n_lap_bins > 0)
                    if (average):
                        ax2.step(self.Laps[lap].bincenters, avelick, where='mid', c=lick_color)
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
            plt.title('No data to show')
            plt.show(block=False)




# #load trigger log 
# datapath = '/Users/ubi/Projects/KOKI/VR/MiceData/'
# #datapath = 'C:\Users\LN-Treadmill\Desktop\MouseData\\'
# #datapath = 'C:\Users\Treadmill\Desktop\RitaNy_MouseData\\'

# # date_time = '2019-11-28_19-37-04' # this was OK!
# date_time = '2019-11-20_08-15-42' # this was not working!
# # date_time = '2019-11-28_19-01-06' # this was OK!
# # date_time = '2019-11-27_09-31-56' # this was OK!
# # date_time = '2019-11-22_13-51-39' # this was OK!
# name = 'th'
# task = 'TwoMazes'
# mm = Session(datapath, date_time, name, task)
# #
# #
# ## # mm.Laps[181].plot_tx()
# ## # mm.Laps[12].plot_xv()
# mm.Laps[25].plot_txv()
# mm.plot_session()


# mm.Laps[17].plot_tx()
# mm.Laps[17].plot_xv()
# mm.Laps[55].plot_tx()
# mm.Laps[55].plot_xv()


# for i in range(65):
#     mm.Laps[i].plot_tx()
#     raw_input("Press Enter to continue...")

