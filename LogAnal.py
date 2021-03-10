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
    N = len(a)
    x = np.zeros(N)
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

    def __init__(self, name, lap, laptime, position, lick_times, reward_times, corridor, mode, actions, corridor_list, dt=0.01, printout=False):
        self.name = name
        self.lap = lap

        self.raw_time = laptime
        self.raw_position = position
        self.lick_times = lick_times
        self.reward_times = reward_times
        self.corridor = corridor # the ID of the corridor in the given stage; This indexes the corridors in the vector called self.corridors
        self.corridor_list = corridor_list 
        self.mode = mode # 1 if all elements are recorded in 'Go' mode
        self.actions = actions
        self.speed_threshold = 5 ## cm / s 106 cm - 3500 roxels; roxel/s * 106.5/3500 = cm/s
        self.corridor_length_roxel = (self.corridor_list.corridors[self.corridor].length - 1024.0) / (7168.0 - 1024.0) * 3500

        self.speed_factor = 106.5 / 3500 ## constant to convert distance from pixel to cm
        self.corridor_length_cm = self.corridor_length_roxel * self.speed_factor # cm

        self.zones = np.vstack([np.array(self.corridor_list.corridors[self.corridor].reward_zone_starts), np.array(self.corridor_list.corridors[self.corridor].reward_zone_ends)])
        self.n_zones = np.shape(self.zones)[1]
        self.preZoneRate = [None, None] # only if 1 lick zone; Compare the 210 roxels just before the zone with the preceeding 210 

        self.dt = 0.01 # resampling frequency = 100 Hz

        ####################################################################
        ## resample time and position with a uniform 100 Hz
        nbins = int(round(self.corridor_length_roxel / 70))
        self.bincenters = np.arange(0, self.corridor_length_roxel, 70) + 70 / 2.0
        
        if (len(self.raw_time) > 200):
            F = interp1d(self.raw_time,self.raw_position) 
            start_time = np.ceil(self.raw_time.min()/self.dt)*self.dt
            end_time = np.floor(self.raw_time.max()/self.dt)*self.dt
            Ntimes = int(round((end_time - start_time) / self.dt)) + 1
            self.laptime = np.linspace(start_time, end_time, Ntimes)
            ppos = F(self.laptime)
            self.lick_position = F(self.lick_times)
            self.reward_position = F(self.reward_times)
    
            ## smooth the position data with a 50 ms Gaussian kernel
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
            speed = np.diff(self.smooth_position) * self.speed_factor  / self.dt # roxel [=rotational pixel] / s       
            speed_first = 2 * speed[0] - speed[1] # linear extrapolation: x1 - (x2 - x1)
            self.speed = np.hstack([speed_first, speed])
    
            ####################################################################
            ## calculate the lick-rate and the average speed versus location    
            bin_counts = np.zeros(nbins)
            for pos in self.smooth_position:
                bin_number = int(pos // 70)
                bin_counts[bin_number] += 1
            self.T_pos = bin_counts * self.dt
    
            lbin_counts = np.zeros(nbins)
            for lpos in self.lick_position:
                lbin_number = int(lpos // 70)
                lbin_counts[lbin_number] += 1
            self.N_licks = lbin_counts
            self.lick_rate = nan_divide(self.N_licks, self.T_pos, where=(self.T_pos > 0.025))
    
            total_speed = np.zeros(nbins)
            for i in range(len(self.smooth_position)):
                ii = int(self.smooth_position[i] // 70)
                total_speed[ii] = total_speed[ii] + self.speed[i]
            total_speed = total_speed * self.dt
            self.ave_speed = nan_divide(total_speed, self.T_pos, where=(self.T_pos > 0.025))
    
            ####################################################################
            ## Calculate the lick rate befor the reward zone - anticipatory licks 210 roxels before zone start
            ## only when the number of zones is 1!
    
            if (self.n_zones == 1):
    
                zone_start = int(self.zones[0][0] * self.corridor_length_roxel)
                lz_posbins = [0, zone_start-420, zone_start-210, zone_start, self.corridor_length_roxel]
    
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
            self.T_pos = np.zeros(nbins)
            self.N_licks = np.zeros(nbins)
            self.ave_speed = np.zeros(nbins)
            self.lick_rate = np.zeros(nbins)
            self.preZoneRate = np.zeros(2)
                

    def plot_tx(self):
        cmap = plt.cm.get_cmap('jet')   
        plt.figure(figsize=(6,4))
        plt.plot(self.laptime, self.smooth_position, c=cmap(50))
        plt.plot(self.raw_time, self.raw_position, c=cmap(90))

        plt.scatter(self.lick_times, np.repeat(self.smooth_position.min(), len(self.lick_times)), marker="|", s=100, c=cmap(180))
        plt.scatter(self.reward_times, np.repeat(self.smooth_position.min()+100, len(self.reward_times)), marker="|", s=100, c=cmap(230))
        plt.ylabel('position')
        plt.xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        plt.title(plot_title)
        plt.ylim(0, self.corridor_length_roxel)

        plt.show(block=False)
       
        # time = mm.Laps[55].time
        # smooth_position = mm.Laps[55].smooth_position
        # lick_times = mm.Laps[55].lick_times
        # reward_times = mm.Laps[55].reward_times
        # lap = mm.Laps[55].lap
        # corridor = mm.Laps[55].corridor
        # lick_rate = mm.Laps[55].lick_rate
        # bincenters = np.arange(0, 3500, 175) + 175 / 2.0

        # plt.figure(figsize=(6,4))
        # plt.plot(laptime, smooth_position, c='g')

        # plt.scatter(lick_times, np.repeat(smooth_position.min(), len(lick_times)), marker="|", s=100)
        # plt.scatter(reward_times, np.repeat(smooth_position.min()+100, len(reward_times)), marker="|", s=100, c='r')
        # plt.ylabel('position')
        # plt.xlabel('time (s)')
        # plot_title = 'Mouse: ' + name + ' position in lap ' + str(lap) + ' in corridor ' + str(corridor)
        # plt.title(plot_title)

        # plt.show(block=False)

    def plot_xv(self):
        cmap = plt.cm.get_cmap('jet')   

        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(self.smooth_position, self.speed, c=cmap(80))
        plt.step(self.bincenters, self.ave_speed, where='mid', c=cmap(30))
        plt.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, c=cmap(180))
        plt.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, c=cmap(230))
        plt.ylabel('speed (cm/s)')
        plt.ylim([min(0, self.speed.min()), max(self.speed.max(), 30)])
        plt.xlabel('position')
        plot_title = 'Mouse: ' + self.name + ' speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        plt.title(plot_title)


        bottom, top = plt.ylim()
        left = self.zones[0,0] * self.corridor_length_roxel
        right = self.zones[1,0] * self.corridor_length_roxel

        polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        ax.add_patch(polygon)
        if (self.n_zones > 1):
            for i in range(1, np.shape(self.zones)[1]):
                left = self.zones[0,i] * self.corridor_length_roxel
                right = self.zones[1,i] * self.corridor_length_roxel
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                ax.add_patch(polygon)

        ax2 = plt.twinx()
        ax2.step(self.bincenters, self.lick_rate, where='mid', c=cmap(200), linewidth=1)
        ax2.set_ylabel('lick rate (lick/s)', color=cmap(200))
        ax2.tick_params(axis='y', labelcolor=cmap(200))
        ax2.set_ylim([-1,max(2*np.nanmax(self.lick_rate), 20)])

        plt.show(block=False)       


        # cmap = plt.cm.get_cmap('jet')   
        # smooth_position = mm.Laps[55].smooth_position
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
        # ax.plot(smooth_position, speed, c=cmap(80))
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

        # polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
        # if (np.shape(zones)[1] > 1):
        #     for i in range(1, np.shape(zones)[1]):
        #         left = zones[0,i] * 3500
        #         right = zones[1,i] * 3500
        #         polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
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
        ax_top.plot(self.laptime, self.smooth_position, c=cmap(50))
        ax_top.plot(self.raw_time, self.raw_position, c=cmap(90))

        ax_top.scatter(self.lick_times, np.repeat(200, len(self.lick_times)), marker="|", s=100, c=cmap(180))
        ax_top.scatter(self.reward_times, np.repeat(400, len(self.reward_times)), marker="|", s=100, c=cmap(230))
        ax_top.set_ylabel('position')
        ax_top.set_xlabel('time (s)')
        plot_title = 'Mouse: ' + self.name + ' position and speed in lap ' + str(self.lap) + ' in corridor ' + str(self.corridor)
        ax_top.set_title(plot_title)
        ax_top.set_ylim(0, self.corridor_length_roxel + 100)


        ## next, plot speed versus position
        ax_bottom.plot(self.smooth_position, self.speed, c=cmap(80))
        ax_bottom.step(self.bincenters, self.ave_speed, where='mid', c=cmap(30))
        ax_bottom.scatter(self.lick_position, np.repeat(5, len(self.lick_position)), marker="|", s=100, c=cmap(180))
        ax_bottom.scatter(self.reward_position, np.repeat(10, len(self.reward_position)), marker="|", s=100, c=cmap(230))
        ax_bottom.set_ylabel('speed (cm/s)')
        ax_bottom.set_xlabel('position')
        ax_bottom.set_ylim([min(0, self.speed.min()), max(self.speed.max(), 30)])

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
        if ((self.m_base > 0) & (self.m_anti > 0)):
            self.test = scipy.stats.wilcoxon(self.baseline, self.anti_rate)
            if ((self.test[1] < 0.01 ) & (greater == True)):
                self.anti = True
        else:
            self.test = [0, 1]


class Session:
    'common base class for low level position and licksensor data in a given session'

    def __init__(self, datapath, date_time, name, task, sessionID=-1, printout=False):
        self.name = name
        self.stage = 0
        self.sessionID = sessionID
        self.stages = []

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

        self.Laps = []
        self.n_laps = 0

        self.get_stage(datapath, date_time, name, task)
        self.corridors = np.hstack([0, np.array(self.stage_list.stages[self.stage].corridors)])

        self.get_lapdata(datapath, date_time, name, task)
        self.test_anticipatory()

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

        for i_lap in np.unique(lap):
            y = lap == i_lap # index for the current lap

            mode_lap = np.prod(mode[y]) # 1 if all elements are recorded in 'Go' mode

            maze_lap = np.unique(maze[y])
            if (len(maze_lap) == 1):
                corridor = self.corridors[int(maze_lap)] # the maze_lap is the index of the available corridors in the given stage
            else:
                corridor = -1

            if (corridor > 0):
                t_lap = laptime[y]
                pos_lap = pos[y]
    
                lick_lap = lick[y]
                t_licks = t_lap[lick_lap]
    
                istart = np.where(y)[0][0]
                iend = np.where(y)[0][-1] + 1
                action_lap = action[istart:iend]
    
                reward_indices = [j for j, x in enumerate(action_lap) if x == "TrialReward"]
                t_reward = t_lap[reward_indices]
    
                actions = []
                for j in range(len(action_lap)):
                    if not((action_lap[j]) in ['No', 'TrialReward']):
                        actions.append([t_lap[j], action_lap[j]])
    
    
                # sessions.append(Lap_Data(name, i, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions))
#                print(self.n_laps, i_lap, len(pos_lap))
                self.Laps.append(Lap_Data(self.name, self.n_laps, t_lap, pos_lap, t_licks, t_reward, corridor, mode_lap, actions, self.corridor_list))
                self.n_laps = self.n_laps + 1
            else:
                N_0lap = N_0lap + 1 # grey zone (corridor == 0) or invalid lap (corridor = -1)

    def get_stage(self, datapath, date_time, name, task):
        action_log_file_string=datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_' + name + '_' + task + '_UserActionLog.txt'
        action_log_file=open(action_log_file_string)
        log_file_reader=csv.reader(action_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            if (line[1] == 'Stage'):
                self.stage = int(round(float(line[2])))

    def test_anticipatory(self):
        corridor_ids = np.zeros(self.n_laps)
        for i in range(self.n_laps):
            corridor_ids[i] = self.Laps[i].corridor # the true corridor ID
        corridor_types = np.unique(corridor_ids)
        nrow = len(corridor_types)
        self.anticipatory = []

        for row in range(nrow):
            ids = np.where(corridor_ids == corridor_types[row])
            n_laps = np.shape(ids)[1]
            n_zones = np.shape(self.Laps[ids[0][0]].zones)[1]
            if ((n_zones == 1) & (n_laps > 2)):
                lick_rates = np.zeros([2,n_laps])
                k = 0
                for lap in np.nditer(ids):
                    lick_rates[:,k] = self.Laps[lap].preZoneRate
                    k = k + 1
                self.anticipatory.append(anticipatory_Licks(lick_rates[0,:], lick_rates[1,:], corridor_types[row]))


    def plot_session(self):
        ## find the number of different corridors
        if (self.n_laps > 0):
            corridor_ids = np.zeros(self.n_laps)
            for i in range(self.n_laps):
                corridor_ids[i] = self.Laps[i].corridor
            corridor_types = np.unique(corridor_ids)
            nrow = len(corridor_types)
            nbins = len(self.Laps[0].bincenters)
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
                maxspeed = 10
                for lap in np.nditer(ids):
                    axs[row,0].step(self.Laps[lap].bincenters, self.Laps[lap].ave_speed, where='mid', c=speed_color_trial)
                    nans_lap = np.isnan(self.Laps[lap].ave_speed)
                    avespeed = nan_add(avespeed, self.Laps[lap].ave_speed)
                    n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                    if (max(self.Laps[lap].ave_speed) > maxspeed): maxspeed = max(self.Laps[lap].ave_speed)
                maxspeed = min(maxspeed, 60)
                
                avespeed = nan_divide(avespeed, n_lap_bins, n_lap_bins > 0)
                axs[row,0].step(self.Laps[lap].bincenters, avespeed, where='mid', c=speed_color)
                axs[row,0].set_ylim([-1,1.2*maxspeed])

                if (row == 0):
                    if (self.sessionID >= 0):
                        plot_title = 'session:' + str(self.sessionID) + ': ' + str(int(n_laps)) + ' laps in corridor ' + str(int(corridor_types[row]))
                    else:
                        plot_title = str(int(n_laps)) + ' laps in corridor ' + str(int(corridor_types[row]))                    
                else:
                    plot_title = str(int(n_laps)) + ' laps in corridor ' + str(int(corridor_types[row]))

                if (self.Laps[lap].zones.shape[1] > 0):
                    bottom, top = axs[row,0].get_ylim()
                    left = self.Laps[lap].zones[0,0] * self.Laps[lap].corridor_length_roxel
                    right = self.Laps[lap].zones[1,0] * self.Laps[lap].corridor_length_roxel

                    polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                    axs[row,0].add_patch(polygon)
                    n_zones = np.shape(self.Laps[lap].zones)[1]
                    if (n_zones > 1):
                        for i in range(1, np.shape(self.Laps[lap].zones)[1]):
                            left = self.Laps[lap].zones[0,i] * self.Laps[lap].corridor_length_roxel
                            right = self.Laps[lap].zones[1,i] * self.Laps[lap].corridor_length_roxel
                            polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True, color='green', alpha=0.15)
                            axs[row,0].add_patch(polygon)
                    # else: ## test for lick rate changes before the zone
                    #     self.anticipatory = np.zeros([2,n_laps])
                    #     k = 0
                    #     for lap in np.nditer(ids):
                    #         self.anticipatory[:,k] = self.Laps[lap].preZoneRate
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
                    ax2.step(self.Laps[lap].bincenters, self.Laps[lap].lick_rate, where='mid', c=lick_color_trial, linewidth=1)
                    nans_lap = np.isnan(self.Laps[lap].lick_rate)
                    avelick = nan_add(avelick, self.Laps[lap].lick_rate)
                    n_lap_bins = n_lap_bins +  np.logical_not(nans_lap)
                    if (max(self.Laps[lap].lick_rate) > maxrate): maxrate = max(self.Laps[lap].lick_rate)
                maxrate = min(maxrate, 20)

                avelick = nan_divide(avelick, n_lap_bins, n_lap_bins > 0)
                ax2.step(self.Laps[lap].bincenters, avelick, where='mid', c=lick_color)
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

