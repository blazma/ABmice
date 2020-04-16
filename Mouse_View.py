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
from Mouse_Close import *

class View_Mouse:
    
    def __init__(self):
        self.root = Tk()
        self.name = 'rn011'
        self.task = 'contingency_learning'

        ####################################################
        # first column
        Label(self.root, text="Mouse name").grid(row=0)
        Label(self.root, text="task").grid(row=1)

        Button(self.root, text='Apply', command=self.apply_view_mouse).grid(row=2, column=0, sticky=W, pady=4)

        ####################################################
        # 2nd column
        self.e0 = Entry(self.root)
        self.e0.grid(row=0, column=1)
        self.e0.insert(0,self.name)

        self.e1 = Entry(self.root)
        self.e1.grid(row=1, column=1)
        self.e1.insert(1,self.task)

        mainloop( )

    def apply_view_mouse(self):
        name = self.e0.get()
        task = self.e1.get()
        name_task_added = True
        if (name == ''):
            print('Enter a valid name string!')
            name_task_added = False
        if (task == ''):
            print('Enter a valid task string!')
            name_task_added = False
        if (name_task_added):
            self.name = name
            self.task = task
            self.root.destroy()
            self.root.quit()


if __name__ == '__main__':

    name_task = View_Mouse()

    datapath = os.getcwd() #current working directory - look for data and strings here!
    # datapath = '/Users/ubi/Projects/KOKI/VR/MiceData'
    m1 = Read_Mouse(name_task.name, name_task.task, datapath, False).mm
    m1.datapath = datapath
    cm = Close_Mouse(m1)

    #ss = 'saving data of mouse ' + name + ' finished, file closed.'
    #print (ss)



