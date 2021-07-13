# -*- coding: utf-8 -*-
"""
Created in Mar 2018
@author: bbujfalussy - ubalazs317@gmail.com
A framework for managing behavioral data in in vivo mice experiments
We define a class - mouse - and the VR environment will interact with this class in each session and trial

"""

import numpy as np
from string import *
import datetime
import time
import os
import pickle
import scipy.stats
import matplotlib.pyplot as plt
import sys
from sys import version_info

from Stages import *
from Corridors import *


class Session_Data:
    'common base class for mice performance on individual sessions'

    def __init__(self, stage, stage_list, task, experimenter, datapath, session_Count, printout=False):
        # stages are defined in the class Mouse - class variable there
        # sessions are numbered consecutively

        # for each session we need the following data to be stored:
        # - start time and end time of the session
        # - stage 
        # - data related to the laps

        # for each trial we need the following data:
        # - context - defined by the maze in the VR
        # - choice
        # - correct
        # we may also store other parameters as
        # - lick left
        # - lick right

        now = datetime.datetime.now()
        self.experimenter = experimenter
        self.task = task
        self.datapath = datapath

        self.start_Time = now.strftime("%Y-%m-%d %H:%M") # human readable time stamp
        self.start_Time_secs = time.time() # current time in secs for internal computations
        self.end_Time = now.strftime("%Y-%m-%d %H:%M")

        self.stage_list = stage_list

        self.stage = int(stage)
        self.stage_Name = self.stage_list.stages[self.stage].name
        self.session = session_Count
        self.name = self.stage_Name + '_session' + str(session_Count)

        # to store trial data - it is a list with the following components for each trial:
        self.laps_data_names = ['VRname', 'start_Time', 'end_Time', 'valid', 'correct', 'choice (left or right)', 'reward (left or right)', 'lick count', 'lick_in_zone', 'reward delivered (num)', 'reward available (num)']
        self.laps = []
        self.n_laps = 0
        # self.laps.append(('VRname', now, 1_sec_later, valid, correct, choice)) - should be a tuple which is inmutable!

        self.sessionlog = []

        self.p_chance = 1
        self.performance = 0

        self.n_reward = 0 # reward collected
        self.n_laps = 0 # number of laps
        self.n_licks = 0 # number of licks
        self.lick_ratio = 0 # lick-in zone / total licks

        if printout: 
            print('New session created.')
            print('Stage: ' + self.stage_Name + ', session:' + str(self.session))

    def add_loginfo(self, VRname, log_Time, parameter, value, note=""):
        # add logging information to the session
        now = datetime.datetime.now()
        log_Time_h = now.strftime("%Y-%m-%d %H:%M") # human readable time stamp
        self.sessionlog.append((log_Time_h, log_Time, VRname, parameter, value, note))

    def test_perf(self, printout=True):
        # test the performance of the mouse if it is significantly different from chance
        # self.laps_data_names = ['VRname', 'start_Time', 'end_Time', 'valid', 'correct', 'choice (left or right)', 'reward (left or right)', 'lick count', 'lick_in_zone', 'reward delivered (num)', 'reward available (num)']
        num_laps = len(self.laps)
        self.n_laps = num_laps
        if (num_laps > 0):
            if printout: print('num of laps:' + str(num_laps))
            lick_count = 1
            lick_in_zone_count= 0
            reward_count = 0
            total_reward = 1
            for i in np.arange(num_laps):
                if (self.laps[i][3] > 0): # the trial is valid
                    lick_count = lick_count + self.laps[i][7]
                    lick_in_zone_count = lick_in_zone_count + self.laps[i][8]
                    reward_count =  reward_count+ self.laps[i][9]
                    total_reward = total_reward + self.laps[i][10]
            self.lick_ratio = float(lick_in_zone_count) / lick_count
            self.performance = float(reward_count) / total_reward
            self.n_reward = reward_count
            self.n_licks = lick_count
            if printout: print('session: ' + self.name + ', Lick_ratio =' + str(round(self.lick_ratio, 5)))

            if (self.stage_list.stages[self.stage].rule == 'correct'):

                is_correct = np.zeros(num_laps)
                is_valid = np.zeros(num_laps)
                for i in np.arange(num_laps):
                    is_valid[i] = self.laps[i][3]
                    if is_valid[i] > 0 : # only valid laps are counted
                        is_correct[i] = self.laps[i][4]
                n_correct = sum(is_correct)
                n_valid = sum(is_valid)
                # if (self.stage < 2):
                #   self.performance = float(n_correct) / num_laps
                #   self.p_chance = scipy.stats.binom_test(n_correct, num_laps, 1.0/2)
                # else:
                self.performance = float(n_correct) / n_valid
                self.p_chance = scipy.stats.binom_test(n_correct, n_valid, 1.0/2)
                if printout: print('session: ' + self.name + ', P(correct) =' + str(round(self.performance, 5)) + ', P(data|chance)= ' + str(round(self.p_chance, 5)))
        else:
                if printout: print('no laps, set todefault!')
                self.performance = 0
                self.p_chance = 1
                self.lick_ratio = 0
                  

    def add_trial(self, VRname, start_Time, valid, correct, choice, reward, lick_count, lick_in_zone, reward_delivered, reward_available, end_Time=None):
        # add a new trial to session
        if (end_Time == None):
            now = datetime.datetime.now()
            end_Time = now.strftime("%Y-%m-%d %H:%M:%S")
        # secsago_10 = now - datetime.timedelta(seconds=10)
        # start_Time = secsago_10.strftime("%Y-%m-%d %H:%M:%S")
        self.laps.append((VRname, start_Time, end_Time, valid, correct, choice, reward, lick_count, lick_in_zone, reward_delivered, reward_available))
        self.n_laps = self.n_laps + 1
        # change the end_Time field to the last trial's end
        self.end_Time = end_Time


    def plot(self):
        # add the following information:
        # 1. correct / error / invalid (y coordinate, to estimate the moving average of the correct)
        # 2. choice (left filled, right empty)
        # 3. trial type (color)

        self.test_perf(False)
        num_laps = len(self.laps)
        is_left = np.zeros(num_laps) # plotted top or bottom
        is_correct = np.zeros(num_laps) # red or green
        is_valid = np.zeros(num_laps) # filled or empty
        corridor = np.zeros(num_laps) # VR ids
        N_corridors = self.stage_list.stages[self.stage].N_corridors

        lick_ratio = np.zeros(num_laps) # plotted top or bottom
        reward_ratio = np.zeros(num_laps) # plotted top or bottom

        cmap = plt.cm.get_cmap('jet')   
        cmap2 = plt.cm.get_cmap('Greys')   
        cols_lick_rew = np.array([[cmap(100)], [cmap(160)]])

        cols_line = np.zeros((num_laps,4)) ## codes the VR
        cols_fill = np.zeros((num_laps,4)) ## codes the choice (left or right) 

        trial = np.arange(num_laps)

        for i in np.arange(num_laps):
            is_valid[i] = int(self.laps[i][3])
            is_correct[i] = self.laps[i][4]
            is_left[i] = self.laps[i][5]
            corridor[i] = self.laps[i][0]

            if (self.laps[i][7] > 0):
                lick_ratio[i] = self.laps[i][8] / float(self.laps[i][7])
            if (self.laps[i][10] > 0):
                reward_ratio[i] = self.laps[i][9] / float(self.laps[i][10])
            
            col_lap = cmap(int(corridor[i] * 250.0 / N_corridors))
            cols_line[i,] = col_lap
            cols_fill[i,] = col_lap
            if (is_left[i] == 0):
                cols_fill[i,] = cmap2(0) # right: white background
            if (is_valid[i] == 0):
                cols_fill[i,] = cmap2(50) # invallid: grey background
                is_correct[i] = 0.5

        plt.figure(figsize=(12,4))
        ax1 = plt.subplot()

        ylabels = ['error', 'invalid', 'correct']
        yy = [0, 0.5, 1]

        ax1.scatter(trial, is_correct, c=cols_fill, edgecolors=cols_line, s=200, linewidth=2, label='')
        ax1.set_ylim(-0.2,1.2)
        ax1.set_xlim(-1,num_laps+1)
        plt.yticks(yy, ylabels)

        for i in np.arange(N_corridors):
            col_lap = cmap(int((i+1) * 250.0 / N_corridors))
            l=ax1.scatter(-2-i, .2, c=np.array(col_lap, ndmin=2), edgecolors=col_lap, s=200, linewidth=2, label=str(self.stage_list.stages[self.stage].corridors[i]))

        l=ax1.scatter(-3-i, .2, c='b', edgecolors='b', s=200, linewidth=2, label='left choice')
        l=ax1.scatter(-4-i, .2, c='w', edgecolors='b', s=200, linewidth=2, label='right choice')

        legend = plt.legend(shadow=False, fontsize='medium', scatterpoints=1, loc=7, title='corridor')

        ax2 = ax1.twinx()
        ax2.plot(trial, lick_ratio, linewidth=2, c=cols_lick_rew[0,][0], label='lick_ratio')
        ax2.plot(trial, reward_ratio, linewidth=2, c=cols_lick_rew[1,][0], label='reward ratio')

        ax2.set_ylabel('lick/reward ratio', color=cols_lick_rew[0,][0])
        ax2.tick_params(axis='y', labelcolor=cols_lick_rew[0,][0])
        ax2.set_ylim([-0.2,1.2])
        ax2.set_xlim(-1,num_laps*1.25)

        plot_title = 'session: ' + self.name + ' performance: ' + str(round(self.performance, 3))
        plt.title(plot_title)
        # legend = plt.legend((l1, l2, l3), ('left', 'right', 'invalid'), shadow=False, fontsize='medium', scatterpoints=1, loc=6)
        plt.show(block=False)


    ## --------------------------------------------
    ## ONLY FOR TESTING - TO BE REMOVED FROM THE FINAL VERSION
    ## --------------------------------------------
    def add_mock_laps(self, n=50, laps_for_75correct=25, start_correct=0.5, filename=None):
        # fill the session with mock laps
        # the learning rate is the number of laps required to achieve 75% performance

        start_correct = start_correct
        level = self.stage_list.stages[self.stage].level
        corridors = self.stage_list.stages[self.stage].corridors
        N_corridors = len(corridors)

        self.corridorfilename = self.datapath + '/' + self.task + '_corridors.pkl'
        if (os.path.exists(self.corridorfilename)):
            input_file = open(self.corridorfilename, 'rb')
            if version_info.major == 2:
                corridors_list = pickle.load(input_file)
            elif version_info.major == 3:
                corridors_list = pickle.load(input_file, encoding='latin1')
            input_file.close()
        else:
            print ('corridorfile missing. Filename: ' + self.corridorfilename)

        if (filename != None):
            session_file = open(filename,"w")

        print ('adding Mock session - ' + self.stage_Name)
        for i in np.arange(n):
            now = datetime.datetime.now()
            start_time = now.strftime("%Y-%m-%d %H:%M:%S")
            percent_correct = 1 - (1-start_correct) * 1 / (float(1) + float(i)/laps_for_75correct)
            valid = int(np.random.binomial(1, percent_correct, 1))

            corridor = np.random.randint(N_corridors)
            corridor_chosen = corridors[corridor]

            n_zones = int(max(1, corridor_list.corridors[corridor_chosen].N_zones))

            correct = int(np.random.binomial(1, percent_correct, 1))

            if (corridor_list.corridors[corridor_chosen].reward == 'Right'):
                choice = correct
            elif (corridor_list.corridors[corridor_chosen].reward == 'Left'):
                choice = 1 - correct
            else:
                choice = int(np.random.binomial(1, 0.5, 1))

            reward_available = n_zones
            if (reward_available > 1):
                if (correct == 1):
                    reward_count = np.random.randint(n_zones) + 1
                else:
                    reward_count = 0
                reward = correct
            else:
                if (corridor_list.corridors[corridor_chosen].reward == 'Left'):    
                    reward = 1 - correct
                    reward_count = 1 - correct
                else:
                    reward = correct
                    reward_count = correct


            lick_count = max(reward_count, np.random.randint(n_zones * 2))
            lick_in_zone = max(reward_count, int(lick_count * percent_correct))
            if (filename == None):
                self.add_trial(corridor, start_time, valid, correct, choice, reward, lick_count, lick_in_zone, reward_count, reward_available)
            else:
                now = datetime.datetime.now()
                end_time = now.strftime("%Y-%m-%d %H:%M:%S")
                lapdata = 'Mock ' + start_time + '\t'  + end_time + '\t' + str(valid) + '\t' + str(correct) + '\t' + str(choice) + '\t' + str(reward) + '\t' + str(lick_count) + '\t' + str(lick_in_zone) + '\t' + str(reward_count) + '\t' + str(reward_available) + '\n'
                session_file.write(lapdata)

        if (filename != None):
            session_file.close()
        else :
            self.test_perf(printout=False)
    ## --------------------------------------------
    ## ONLY FOR TESTING - TO BE REMOVED FROM THE FINAL VERSION
    ## --------------------------------------------

    def update(self, filename):
        if os.path.exists(filename):
            session_file = open(filename,"r")
            session_data = session_file.readlines()
            session_file.close()

            n_laps = len(session_data)
            for i_lap in range(n_laps):
                lap_data = session_data[i_lap].rstrip()
#                sld = lap_data.split('\t') # split_lap_data
                sld = lap_data.split() # split_lap_data
                if not (sld[0] == '0'):
                    self.add_trial(sld[0], sld[1], int(sld[3]), int(sld[4]), int(sld[5]), int(sld[6]), int(sld[7]), int(sld[8]), int(sld[9]), int(sld[10]), end_Time=sld[2])
        
            self.n_laps = n_laps
            # change the end_Time field to the last trial's end
            if (n_laps > 0):
                self.end_Time = sld[2]
                self.test_perf(printout=False)


class Mouse():
    'Common base class for all mice'

    def __init__(self, name, task, datapath, left_color=None, right_color=None, printout=False):
        self.datapath = datapath
        self.name = name # Adrian
        self.task = task # 2maze
        self.left_color = left_color # Blue
        self.right_color = right_color # Green
        # these fields should be read interactively ...
        now = datetime.datetime.now()
        self.creation_date = now.strftime("%Y-%m-%d %H:%M")

        self.stagefilename = self.datapath + '/' + self.task + '_stages.pkl'
        input_file = open(self.stagefilename, 'rb')
        if version_info.major == 2:
            self.stage_list = pickle.load(input_file)
        elif version_info.major == 3:
            self.stage_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

        ## we define the stage list by reading it from file. Dould be the same for all instances!


        self.sessions = []
        self.stage = 0 # the stage on the next session
        self.proposed_stage = 0 # the stage on the next session

        self.performance = np.array([]) # performance of the past sessions (P(correct)) / # total reward COLLECTED / reward AVAILABLE in the different sessions
        self.p_chance = np.array([]) # P(n_correct | chance)
        self.stages = np.array([]) # the stages of the past sessions
        
        self.n_reward = np.array([]) # reward collected in a given session
        self.n_laps = np.array([]) # number of laps in a given session
        self.n_licks = np.array([]) # number of licks in a given session
        self.lick_ratio = np.array([]) # ratio of licks within reward zone in the past sessions (licks_in_zone / totel_lick_count)

        if (printout==True):
            print('New mouse created with name ' + self.name + ', task:' + self.task)


    def add_session(self, stage, experimenter):
        # create a new behavioral session
        session_Count = len(self.sessions)
        self.stage = int(stage) # update the stage variable with the new session's stage
        self.stages = np.append(self.stages, stage) # update the stages with the new session's stage
        self.sessions.append(Session_Data(stage, self.stage_list, self.task, experimenter, self.datapath, session_Count))

    # def testattr(self):
    #   # create a new behavioral session
    #   y = getattr(self, "stage", None)
    #   if y is None:
    #       print('nincs - 1')
    #   else:
    #       print("van neki - 1")

    #   y = getattr(self, "stage2", None)
    #   if y is None:
    #       print('nincs - 2')
    #   else:
    #       print("van neki - 2")

    def update(self):
        # update session data with data read from file
        new_session_found = False
        data_dir = self.datapath + '/data/' + self.name + '_' + self.task + '/behaviour_data/'

        if not os.path.exists(data_dir):
            print(data_dir)
            print ('no data has been found')     
            return 

        for i_session in range(len(self.sessions)):
            nlaps = self.sessions[i_session].n_laps
            if ((nlaps == 0) & (self.sessions[i_session].stage > 0)):
                filename = data_dir + self.name + '_' + self.task + '_session' + str(i_session) + '.txt'
                self.sessions[i_session].update(filename)
                print(data_dir)
                print('updating mouse for session ', i_session)
                new_session_found = True

        print('mouse updated')
        self.test_perf(printout=False)
        return new_session_found

    ## --------------------------------------------
    ## ONLY FOR TESTING - TO BE REMOVED FROM THE FINAL VERSION
    ## --------------------------------------------
    def add_mock_sessions(self, experimenter, stage_max=None, n_new_session=None, write_to_file=False, add_new_session=True):
        # create a new behavioral session
        n75=25 # laps to 75% performance
        stc=0.5 # start correct
        stage = self.stage
        n = 0
        more_stages = True
        if ((stage_max == None) & (n_new_session == None)):
            return
        if (write_to_file):
            n_new_session = 1

        while (more_stages == True):
            self.proposed_stage = int(stage)
            self.stage = self.proposed_stage
            if (add_new_session):
                self.add_session(stage, experimenter)
            i_session = len(self.sessions) - 1
            stc = min(0.5 + sum(self.stages == stage)/10.0, 0.75)
            print('stc='+str(stc))
            n_laps = np.random.randint(50) + 25

            if (write_to_file == True):
                data_dir = './data/' + self.name + '_' + self.task + '/'
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                filename = data_dir + self.name + '_' + self.task + '_session' + str(i_session) + '.dat'
                self.sessions[i_session].add_mock_laps(n=n_laps, laps_for_75correct=n75, start_correct=stc, filename=filename)
            else :
                filename = None
                self.sessions[i_session].add_mock_laps(n=n_laps, laps_for_75correct=n75, start_correct=stc)
                if (self.stage_list.stages[self.stage].rule == 'correct'):
                    if (self.sessions[i_session].p_chance < 0.01):
                        if (self.sessions[i_session].performance > 0.6):
                            stage = np.random.choice(self.stage_list.stages[self.stage].next_stage)
                            ss = 'stage changed, next stage:' + str(stage)
                            n75 = 50
                        else :
                            n75 = max(n75-10, 10)
                    else :
                        n75 = max(n75-10, 20)
                else :
                    if (self.sessions[i_session].lick_ratio > 0.6):
                        stage = np.random.choice(self.stage_list.stages[self.stage].next_stage)
                        ss = 'stage changed, next stage:' + str(stage)

            n = n + 1
            if (stage_max == None):
                if (n >= n_new_session):
                    more_stages = False
            else:
                if (stage > stage_max):
                    more_stages = False

        print('\n')

    ## --------------------------------------------
    ## ONLY FOR TESTING - TO BE REMOVED FROM THE FINAL VERSION
    ## --------------------------------------------

    def write(self, test=False):
        # write mouse data to file
        # before writing it to file we test the mouse's performance
        if test==True: self.test_perf(printout=False)
        # create the directory 
        data_dir = self.datapath + '/data/' + self.name # + '_' + self.task
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # the filename
        fname = data_dir + '/' + self.name + '_' + self.task + '.pkl'
        f = open(fname, 'wb')
        pickle.dump(self, f)
        f.close()


    def plot(self):
        # plot the performance of the mouse in the different session
        self.test_perf(printout=False)

        num_sessions = len(self.stages)
        session = np.arange(num_sessions)

        cmap = plt.cm.get_cmap('jet')  
        cmap2 = plt.cm.get_cmap('Greys')  
        
        num_stages = max(11,int(self.stage_list.num_stages))
        cols_levels = np.zeros((num_stages,4))
        for i in np.arange(num_stages):
            cols_levels[i,] = cmap(int(i * 256 / num_stages))

        # cols_levels = np.array([[cmap(0)], [cmap(12)], [cmap(24)], [cmap(36)], [cmap(48)], [cmap(200)], [cmap(240)], [cmap(85)], [cmap(210)], [cmap(130)], [cmap(170)]])
        cols_perf = np.zeros((num_sessions,4))
        cols_lick = np.zeros((num_sessions,4))
        cols_lick2 = cols_levels[10]
        cols_laps = cols_levels[1]
        cols_rew = cols_levels[5]
        cols_P = np.zeros((num_sessions,4))
        for i in np.arange(num_sessions):
            ind_col = int(self.stages[i])
            cols_perf[i,] = cols_levels[ind_col]
            cols_lick[i,] = cmap2(100)
            cols_P[i,] = cols_levels[8]


        plt.figure(figsize=(10,8))
        ax = plt.subplot(3, 1, 1)
        ax.set_ylabel('performance')

        plt.scatter(session, self.performance, c=cols_perf, s=200, marker='o')
        plt.scatter(session, self.lick_ratio, c=cols_lick, s=100, marker='*')
        plot_title = 'Mouse: ' + self.name + ' ' + self.task + ' performance and lick count'
        plt.title(plot_title)
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([-1,num_sessions*1.3])

        for i in np.sort(np.unique(self.stages)):
            l=plt.scatter(-4-int(i), 0.5, c=np.array(cols_levels[int(i)], ndmin=2), s=100, marker='o', label=self.stage_list.stages[int(i)].name)
        l=plt.scatter(-5-int(i), 0.5, c=np.array(cols_levels[int(i)], ndmin=2), s=100, marker='o', label='performance')
        l=plt.scatter(-6-int(i), 0.5, c=np.array(cmap2(100), ndmin=2), s=100, marker='*', label='lick ratio')
        legend = plt.legend(shadow=False, fontsize='small', scatterpoints=1, loc=7, title='stage')

        ax2 = ax.twinx()
        ax2.plot(session, self.n_licks, c=cols_lick2, linewidth=3)
        ax2.set_ylabel('number of licks', color=cols_lick2)
        ax2.tick_params(axis='y', labelcolor=cols_lick2)
        ax2.set_xlim([-1,num_sessions*1.3])
        ax2.set_ylim([0,1.1*max(self.n_licks)])

        ax3 = plt.subplot(3, 1, 2)
        plt.plot(session, self.n_laps, c=cols_laps, linewidth=3)
        plt.title('')
        ax3.set_ylabel('number of laps', color=cols_laps)
        ax3.tick_params(axis='y', labelcolor=cols_laps)
        ax3.set_xlim([-1,num_sessions*1.3])
        ax3.set_ylim([0,1.1*max(self.n_laps)])

        ax4 = ax3.twinx()
        ax4.plot(session, self.n_reward, c=cols_rew, linewidth=3)
        ax4.set_ylabel('reward consumed', color=cols_rew)
        ax4.tick_params(axis='y', labelcolor=cols_rew)
        ax4.set_xlim([-1,num_sessions*1.3])
        ax4.set_ylim([0,1.1*max(self.n_reward)])

        ax5 = plt.subplot(3, 1, 3)
        ax5.scatter(session, self.p_chance, c=cols_P, s=100, marker='^')
        ax5.set_xlim([-1,num_sessions*1.3])
        ax5.set_xlabel('session number')
        ax5.set_yscale('log')
        ax5.set_ylim([0.1 * min(self.p_chance),2])
        ax5.set_ylabel('P(data|chance)', color=cols_P[1])
        ax5.hlines(y=[0.01, 0.05, 1], xmin=-1, xmax = num_sessions+1, linestyles='dotted', colors='0.75')
        plt.show(block=False)


    def test_perf(self, printout=True):
        # test the performance of the mouse in each session if it is significantly different from chance
        # update the stage variable to be the last session that was significantly different from chance
        n_sessions = int(len(self.sessions))
        last_stage = int(self.stages[n_sessions-1])
        if (n_sessions > 0):
            performance = np.zeros(n_sessions)
            p_chance = np.zeros(n_sessions)
            lick_ratio = np.zeros(n_sessions)
            n_reward = np.zeros(n_sessions)
            n_laps = np.zeros(n_sessions)
            n_licks = np.zeros(n_sessions)

            # n_done = len(self.performance)
            # performance[0:n_done] = self.performance
            # p_chance[0:n_done] = self.p_chance
            # lick_ratio[0:n_done] = self.lick_ratio
            # n_reward[0:n_done] = self.n_reward
            # n_laps[0:n_done] = self.n_laps
            # n_licks[0:n_done] = self.n_licks

            for i in range(n_sessions):
                self.sessions[i].test_perf(printout=printout)
                performance[i] = self.sessions[i].performance   
                p_chance[i] = self.sessions[i].p_chance
                lick_ratio[i] = self.sessions[i].lick_ratio
                n_reward[i] = self.sessions[i].n_reward
                n_laps[i] = self.sessions[i].n_laps
                n_licks[i] = self.sessions[i].n_licks

            self.performance = performance
            self.p_chance = p_chance
            self.lick_ratio = lick_ratio
            self.n_reward = n_reward
            self.n_laps = n_laps
            self.n_licks = n_licks

            # change stage if performance is good enough
            if (self.sessions[n_sessions-1].n_laps >= 50):
                if (self.stage_list.stages[last_stage].rule == 'correct'):
                    if (self.sessions[n_sessions-1].p_chance < 0.01):
                        if (self.sessions[n_sessions-1].performance > 0.6):
                            self.proposed_stage = np.random.choice(self.stage_list.stages[last_stage].next_stage)
                else:
                    if (self.sessions[n_sessions-1].lick_ratio > 0.6):
                        self.proposed_stage = np.random.choice(self.stage_list.stages[last_stage].next_stage)


            if (printout==True):
                print('\n mouse '+self.name+'`s performance tested:')
                for i in range(n_sessions):
                    if (self.stage_list.stages[int(self.stages[i])].rule == 'correct'):
                        ss = 'session: ' + self.sessions[i].name + ', P(correct) =' + str(round(self.performance[i],5)) + ', P(data|chance)= ' + str(round(self.p_chance[i],5))
                    else :
                        ss = 'session: ' + self.sessions[i].name + ', lick_ratio =' + str(round(self.lick_ratio[i],5))
                    print(ss)

                print('proposed next session: ' + self.stage_list.stages[self.proposed_stage].name)
        else :
            if (printout==True):
                print('new mouse, next session: ' + self.stage_list.stages[self.stage].name)

class Read_Mouse:
    def __init__(self, name, task, data_dir, printout=False):
        self.name = name
        self.task = task

        input_path = data_dir + '/data/' + self.name + '/' + self.name + '_' + self.task + '.pkl'
        print (input_path)
        
        if (os.path.exists(input_path)):
            input_file = open(input_path, 'rb')
            if version_info.major == 2:        
                self.mm = pickle.load(input_file)
            elif version_info.major == 3:
                self.mm = pickle.load(input_file, encoding='latin1')
            input_file.close()
            n_sessions = len(self.mm.sessions)-1
            self.mm.test_perf(printout=printout)
            stage_Name = self.mm.sessions[n_sessions].stage_Name
            performance = self.mm.sessions[n_sessions].performance
            p_val = self.mm.sessions[n_sessions].p_chance
            if (printout == True):
                print('Mouse read from file with name ' + self.name + ' and task: ' + self.task)
                print('\t had ' +  str(n_sessions + 1) + ' sessions') 
                print('\t and the last session was in the ' + stage_Name + ' stage with ' + str(round(performance, 5)) + ' performance (P='+ str(round(p_val, 5)) + ')')

        else:
            self.mm = Mouse(name, task, data_dir, printout=printout)


# datapath = os.getcwd()
# name = 'rn018'
# task = 'contingency_learning'
# data_dir = datapath
# input_path = data_dir + '/data/' + name + '/' + name + '_' + task + '.pkl'
# input_file = open(input_path, 'rb')
# mm = pickle.load(input_file, encoding='latin1')
# input_file.close()
# mm.plot()
# mm.sessions[19].plot()

# m1 = Read_Mouse('rn018', 'contingency_learning', datapath, False).mm
# m1.plot()
# m1.sessions[4].plot()


