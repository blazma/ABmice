# -*- coding: utf-8 -*-
"""
Created in Mar 2018
@author: bbujfalussy - ubalazs317@gmail.com
A framework for managing behavioral data in in vivo mice experiments
We define a class - mouse - and the VR environment will interact with this class in each session and trial

"""
from sys import version_info
if version_info.major == 2:
    # We are using Python 2.x
    # import Tkinter as tk
    from Tkinter import *
elif version_info.major == 3:
    # We are using Python 3.x
    # import tkinter as tk
    from tkinter import *
import datetime
import sys
import numpy as np
import os
import sys
from Mice import *
from LogAnal import *

class Close_Mouse:
    
    def __init__(self, mouse):
        self.mouse=mouse
        self.loaded_session = np.nan
        self.newsession = False
        ## if exists, read data from the last session, add it to the mouse and save it.
        self.new_data_found = self.mouse.update()

        self.root = Tk()

        ####################################################
        # first column
        Label(self.root, text="Mouse name").grid(row=0)
        Label(self.root, text="task").grid(row=1)
        Label(self.root, text="Last session").grid(row=2)
        Label(self.root, text="Comment for last session: ").grid(row=3)

        Label(self.root, text="performance: ").grid(row=4)
        Label(self.root, text="P(performance|chance): ").grid(row=5)
        Label(self.root, text="number of laps:").grid(row=6)
        Label(self.root, text="N licks: ").grid(row=7)
        Label(self.root, text="Lick ratio: ").grid(row=8)
        Label(self.root, text="N reward: ").grid(row=9)
        Label(self.root, text="Session to plot: ").grid(row=10)

        Label(self.root, text="Analyse session/trial: ").grid(row=12)
        Button(self.root, text='Apply', command=self.apply_mouse_data).grid(row=13, column=0, sticky=W, pady=4)

        ####################################################
        # 2nd column
        Label(self.root, text=self.mouse.name).grid(row=0, column=1)
        Label(self.root, text=self.mouse.task).grid(row=1, column=1)
        i_last_session = len(self.mouse.sessions) - 1
        Label(self.root, text=str(i_last_session)).grid(row=2, column=1)

        self.e3 = Entry(self.root)
        self.e3.grid(row=3, column=1)
        self.e3.insert(3,'add your comment here')

        Label(self.root, text=str(np.round(self.mouse.sessions[i_last_session].performance, 5))).grid(row=4, column=1)
        Label(self.root, text=str(np.round(self.mouse.sessions[i_last_session].p_chance, 5))).grid(row=5, column=1)
        Label(self.root, text=str(self.mouse.sessions[i_last_session].n_laps)).grid(row=6, column=1)
        Label(self.root, text=str(self.mouse.sessions[i_last_session].n_licks)).grid(row=7, column=1)
        Label(self.root, text=str(np.round(self.mouse.sessions[i_last_session].lick_ratio, 5))).grid(row=8, column=1)
        Label(self.root, text=str(self.mouse.sessions[i_last_session].n_reward)).grid(row=9, column=1)

        self.e10 = Entry(self.root)
        self.e10.grid(row=10, column=1)
        self.e10.insert(10,i_last_session)

        Button(self.root, text='Plot selected session', command=self.plot_session).grid(row=11, column=1, sticky=W, pady=4)
        Button(self.root, text='Plot mouse data', command=self.plot).grid(row=11, column=2, sticky=W, pady=4)

        self.e12a = Entry(self.root)
        self.e12a.grid(row=12, column=1)
        self.e12a.insert(12,i_last_session)

        self.e12b = Entry(self.root)
        self.e12b.grid(row=12, column=2)
        self.e12b.insert(12,0)

        Button(self.root, text='Analyse selected session', command=self.analyse_session).grid(row=13, column=1, sticky=W, pady=4)
        Button(self.root, text='Analyse lap', command=self.analyse_lap).grid(row=13, column=2, sticky=W, pady=4)

        mainloop( )

    def plot(self):
        self.mouse.plot()

    def load_session(self, i_session_load):
# datapath = '/Users/ubi/Dropbox/Bazsi/MiceData/'
# date_time = '2019-11-13_08-40-14'
# name = 'th'
# task = 'TwoMazes'
# sessiondata = Session(datapath, date_time, name, task)

        self.newsession = False
        datadir = './data/' + self.mouse.name + '_' + self.mouse.task
        sessiondirs = sorted(os.listdir(datadir), )

        save_date = datetime.datetime.strptime(self.mouse.sessions[i_session_load].start_Time, '%Y-%m-%d %H:%M')
        stime = datetime.timedelta(seconds=5)
        ref_date = save_date - stime # we start the search from 5s before the initial creation of the session

        folder_index = -1
        for ii in range(len(sessiondirs)):
            # we select the first folder after the ref_date within 63s creation ...
            try:
                i_date = datetime.datetime.strptime(sessiondirs[ii], '%Y-%m-%d_%H-%M-%S')
                if (i_date >= ref_date):
                    delta_t = i_date - ref_date
                    delta_t_sec = delta_t.total_seconds()
                    if (delta_t_sec < 120):
                        folder_index = ii
            except:
                i_date = np.nan
        if (folder_index >= 0):
            date_time = sessiondirs[folder_index] # the first directory is 
            # sessiondir = datadir + '/' + date_time
            self.sessiondata = Session('./', date_time, self.mouse.name, self.mouse.task, i_session_load)
            self.loaded_session = i_session_load
            self.newsession = True
        else:
            print('no data found for the session selected')
           

    def analyse_session(self):
        i_session_anal = int(self.e12a.get())
        ## we load the selected session if no session is loaded or ... 
        if (np.isnan(self.loaded_session)):
            if (i_session_anal > len(self.mouse.sessions)):
                print('session selected is out of range')
                return
            else :
                self.load_session(i_session_anal)
                print('session loaded successfully')
        ## we load the selected session if ... a different session was loaded
        if (self.loaded_session != i_session_anal):
            if (i_session_anal > len(self.mouse.sessions)):
                print('session selected is out of range')
                return
            else :
                self.load_session(i_session_anal)
                print('session loaded successfully')
        
        if (self.newsession):
            self.sessiondata.plot_session()
        print('\n')

    def analyse_lap(self):
        i_session_anal = int(self.e12a.get())
        ## we load the selected session if no session is loaded or ... 
        if (np.isnan(self.loaded_session)):
            if (i_session_anal > len(self.mouse.sessions)):
                print('selected session is out of range')
                return
            else :
                self.load_session(i_session_anal)
        ## we load the selected session if ... a different session was loaded
        if (self.loaded_session != i_session_anal):
            if (i_session_anal > len(self.mouse.sessions)):
                print('selected session is out of range')
                return
            else :
                self.load_session(i_session_anal)
        i_lap_anal = int(self.e12b.get())# - 1
        if (i_lap_anal <= self.sessiondata.n_laps):
            self.sessiondata.Laps[i_lap_anal].plot_txv()
        print('\n')

    def plot_session(self):
        i_session_plot = int(self.e10.get())
        if (i_session_plot <= len(self.mouse.sessions)):
            self.mouse.sessions[i_session_plot].plot()
        else :
            print('selected session is out of range')
        print('\n')

    def apply_mouse_data(self):
        e3_data = self.e3.get()
        if (e3_data != 'add your comment here'):
            self.new_data_found = True
            i_last_session = len(self.mouse.sessions) - 1
            self.mouse.sessions[i_last_session].add_loginfo(VRname='-', log_Time='-', parameter='-', value='-', note=e3_data)

        if (self.new_data_found):
            self.mouse.write()

        self.root.destroy()
        self.root.quit()


if __name__ == '__main__':

    name = sys.argv[1]
    task = sys.argv[2]
    experimenter = sys.argv[3]
#     name = 'test'
#     left_color = 'No'
#     right_color = 'No'
    datapath = os.getcwd() #current working directory - look for data and strings here!
    m1 = Read_Mouse(name, task, datapath, False).mm
    cm = Close_Mouse(m1)

    #ss = 'saving data of mouse ' + name + ' finished, file closed.'
    #print (ss)



