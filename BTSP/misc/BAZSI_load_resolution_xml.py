# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:34:54 2022

@author: luko.balazs
"""
from xml.dom import minidom
from matplotlib import pyplot as plt
import numpy as np
import time

def get_resolution(imaging_logfile_name):
    
    import xml.etree.ElementTree as ET
    
    
    tree = ET.parse(imaging_logfile_name)
    root = tree.getroot()
    
    PV = root[1]
    for child in PV:
        if child.attrib['key'] == 'micronsPerPixel':
            # print(child.attrib)
            microns_x = child[0].attrib['value']
            microns_y = child[1].attrib['value']
            microns_z = child[2].attrib['value']
    
    if microns_x != microns_y:
        print('Warning! X and Y resolution of recording is not the same')
    
    return microns_x, microns_y,microns_z

def load_frame_rate(imaging_logfile_name):
    prew_time = time.time()
    # function that reads the action_log_file and finds the current stage
    # minidom is an xml file interpreter for python
    # hope it works for python 3.7...
    imaging_logfile = minidom.parse(imaging_logfile_name)
    # voltage_rec = imaging_logfile.getElementsByTagName('VoltageRecording')
    # voltage_delay = float(voltage_rec[0].attributes['absoluteTime'].value)
    ## the offset is the time of the first voltage signal in Labview time
    ## the signal's 0 has a slight delay compared to the time 0 of the imaging recording 
    ## we substract this delay from the offset to get the LabView time of the time 0 of the imaging recording
    
    #find out whether it's a multiplane recording
    sequence = imaging_logfile.getElementsByTagName('Sequence')
    # print('sequence',len(sequence))
    frames = imaging_logfile.getElementsByTagName('Frame')
    # print('frames',len(frames))
    if sequence[0].attributes['type'].value == 'TSeries ZSeries Element':
        print('multi-plane')

        frame_times = np.zeros(len(sequence)-1)
        for i in range(len(frame_times)):
            # frame_times[i] = (float(frames[2*i].attributes['relativeTime'].value) + float(frames[2*i+1].attributes['relativeTime'].value))/2
            frame_times[i] = (float(frames[2*i].attributes['absoluteTime'].value) + float(frames[2*i+1].attributes['absoluteTime'].value))/2
        
    else:
        print('single-plane')

        frame_times = np.zeros(len(frames)) # this is already in labview time
        # im_reftime = float(frames[1].attributes['relativeTime'].value) - float(frames[1].attributes['absoluteTime'].value)
        for i in range(len(frames)):
            frame_times[i] = float(frames[i].attributes['relativeTime'].value)
            
    
    # plt.figure()
    # plt.plot(frame_times)
    # plt.show()
    
    # print(np.average(np.diff(frame_times)), frame_times[-1]/len(frames))
    # print('time used:',round(time.time()-prew_time,2), 'seconds')
    return np.median(np.diff(frame_times))
    
def load_frame_rate_fast(imaging_logfile_name):
    prew_time = time.time()
    imaging_logfile = minidom.parse(imaging_logfile_name)
    
    #find out whether it's a multiplane recording
    sequence = imaging_logfile.getElementsByTagName('Sequence')
    frames = imaging_logfile.getElementsByTagName('Frame')
    if sequence[0].attributes['type'].value == 'TSeries ZSeries Element':
        print('multi-plane')
        # last_time = (float(frames[-4].attributes['relativeTime'].value) + float(frames[-4+1].attributes['relativeTime'].value))/2
        last_time = (float(frames[-4].attributes['absoluteTime'].value) + float(frames[-4+1].attributes['absoluteTime'].value))/2
        average_rate = last_time/(len(sequence)-1)
        print(average_rate)
        
    else:
        print('single-plane')
        last_time = float(frames[-1].attributes['relativeTime'].value)
        average_rate = last_time/len(frames)
        print(average_rate)
    
    print('time used:',round(time.time()-prew_time,2), 'seconds')
    
    return average_rate

# imaging_logfile_name = imaging_logfile_name = r'E:\data\Snezana\srb210_xmls\srb210_220624-006.xml'

# # x,y,z = get_resolution(imaging_logfile_name)
# # print(x,y,z)

# average_frame_rate = load_frame_rate(imaging_logfile_name)
# print(average_frame_rate)
# average_frame_rate_fast = load_frame_rate_fast(imaging_logfile_name)
# print(average_frame_rate_fast)
