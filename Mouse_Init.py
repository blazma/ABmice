# -*- coding: utf-8 -*-
"""
Created in Mar 2019
@author: bbujfalussy - ubalazs317@gmail.com
A framework for managing behavioral data in in vivo mice experiments
We define a class - mouse - and the VR environment will interact with this class in each session and trial

initializing mouse parameters for the VR setup
expects three string inputs: Mouse name, left color and right color
provides a dialog for plotting the past behavior and selecting the next stage
returns the next stage and the corridor properties
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
import pickle
from Mice import *
from Stages import *
from Corridors import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class InitMouse:
    
    def __init__(self, mouse, experimenter):
        self.mouse=mouse
        if (len(m1.sessions) > 0):
            old_mouse = True
        else :
            old_mouse = False

        self.root = Tk()
        Label(self.root, text="Mouse name").grid(row=0)
        Label(self.root, text="task").grid(row=1)
        Label(self.root, text="experimenter").grid(row=2)
        Label(self.root, text="proposed stage").grid(row=3)
        Label(self.root, text="next stage:").grid(row=4)

        Label(self.root, text=self.mouse.name).grid(row=0, column=1)
        Label(self.root, text=self.mouse.task).grid(row=1, column=1)
        Label(self.root, text=experimenter).grid(row=2, column=1)
        Label(self.root, text=self.mouse.proposed_stage).grid(row=3, column=1)

        self.e4 = Entry(self.root)
        self.e4.grid(row=4, column=1)
        self.e4.insert(10,self.mouse.stage)

        Button(self.root, text='Apply', command=self.apply_mouse_data).grid(row=5, column=0, sticky=W, pady=4)
        Button(self.root, text='Show stages', command=self.show).grid(row=5, column=1, sticky=W, pady=4)
        if (old_mouse):
            Button(self.root, text='Plot mouse data', command=self.plot).grid(row=5, column=2, sticky=W, pady=4)
        else:
            Label(self.root, text='New mouse, no plotting').grid(row=5, column=2)

        mainloop( )

    def plot(self):
        self.mouse.plot()

    def show(self):
        stageimagefile = datapath + '/' + self.mouse.task + '_stages.png'
        img = mpimg.imread(stageimagefile)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(img)
        ax.set_axis_off()
        plt.show(block=False)   

    def apply_mouse_data(self):
        e4_data = self.e4.get()
        self.stage_selected = int(e4_data)
        if ((self.stage_selected < self.mouse.stage_list.num_stages) and (self.stage_selected >= 0)):
            self.root.destroy()
            self.root.quit()
        else:
            self.e4.delete(0, 'end')



if __name__ == '__main__':
    name = sys.argv[1]
    task = sys.argv[2]
    experimenter = sys.argv[3]
    # name = 'test'
    # task = 'morphing'
    # experimenter = 'Kata'
    datapath = os.getcwd() #current working directory - look for data and strings here!

    corridorfilename = datapath + '/'  + task + '_corridors.pkl'
    if (os.path.exists(corridorfilename)):
        input_file = open(corridorfilename, 'rb')
        if version_info.major == 2:
            corridor_list = pickle.load(input_file)
        elif version_info.major == 3:
            corridor_list = pickle.load(input_file, encoding='latin1')
        input_file.close()

    m1 = Read_Mouse(name, task, datapath).mm

    im = InitMouse(m1, experimenter)

    m1.add_session(im.stage_selected, experimenter)
    m1.write()

    VR_ids = m1.stage_list.stages[im.stage_selected].corridors
    substages = m1.stage_list.stages[im.stage_selected].substages
    n_corridors = len(VR_ids)
    total_rows = 0
    corridor_lengths = np.zeros(n_corridors) ## no grey zone ...
    for i in range(n_corridors):
        i_corridor = VR_ids[i]
        n_zones = len(corridor_list.corridors[i_corridor].reward_zone_starts)
        total_rows = total_rows + n_zones
        corridor_lengths[i] = corridor_list.corridors[i_corridor].length

    if (len(np.unique(corridor_lengths)) > 1):
        print('corridor length is not unique!')
    corridorLength = int(round(corridor_lengths[0]))

    grey_zone_active = 0
    P_reward = 1
    if (m1.stage_list.stages[im.stage_selected].rule == 'correct'):
        grey_zone_active = 1
    if (isinstance(m1.stage_list.stages[im.stage_selected].rule, float)):
        grey_zone_active = 1
        if ((m1.stage_list.stages[im.stage_selected].rule <= 1) & (m1.stage_list.stages[im.stage_selected].rule > 0)):
            P_reward = float(m1.stage_list.stages[im.stage_selected].rule)

    ### stage;      session number; number of rows; grey_zone_Active;   corridor length (roxel);    Reward probability 
    ### i_corridor; i_substage;     zone start;     zone end;           reward side;                corridor probability 
    return_value = str(im.stage_selected) + '_' + str(int(len(m1.sessions))-1) + '_' + str(total_rows) + '_' + str(grey_zone_active) + '_' + str(corridorLength) + '_' + str(P_reward) + '\n'
    for i in range(n_corridors):
        i_corridor = VR_ids[i]
        i_substage = str(substages[i])
        n_zones = len(corridor_list.corridors[i_corridor].reward_zone_starts)
        P_corridor = 1
        if (isinstance(m1.stage_list.stages[im.stage_selected].random, list)):
            P_corridor = m1.stage_list.stages[im.stage_selected].random[i]
        for ii in range(n_zones):
            zone_start = round(corridor_list.corridors[i_corridor].reward_zone_starts[ii], 5)
            zone_end = round(corridor_list.corridors[i_corridor].reward_zone_ends[ii], 5)
            reward_side_num = 0
            if (corridor_list.corridors[i_corridor].reward == 'Left'): reward_side_num = -1
            if (corridor_list.corridors[i_corridor].reward == 'Right'): reward_side_num = 1
            return_value = return_value+str(i+1)+'_'+i_substage+'_'+str(zone_start)+'_'+str(zone_end)+'_'+str(reward_side_num)+'_'+str(P_corridor)+'\n'
    
    return_value = return_value+'\n'
    print (return_value)    
    
