# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:30:49 2020

@author: luko.balazs
"""
import sys
import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import minimize_scalar
sys.path.append(r'C:\Users\luko.balazs\AppData\Local\Continuum\anaconda3\Lib\site-packages')
import pandas as pd
from numba import njit, prange
from scipy.ndimage import filters
from scipy.signal import argrelextrema, find_peaks
import math
import openpyxl
from xml.dom import minidom
from scipy.interpolate import interp1d
import time

from ImageAnal import LocateImaging

start_time = time.time()

plt.close('all')

class ProcessManualRoiData:
    def __init__ (self, datapath, suite2p_folder, imaging_logfile_name, excel_file_path, name, task, date_time, data_source='manual', ids=-1):
        #OOP parameters from inputs
        self.data_source=data_source
        self.excel_file_path=excel_file_path
        self.suite2p_folder=suite2p_folder
        self.save_path=suite2p_folder[0:-1]+'manual_ROI\\'
        self.man_image=self.LoadImage()#Todo: létezik-e a file
        self.ids=ids
        self.manual_data_excel_file = self.suite2p_folder+'manual_roi_data.xlsx'
        self.imaging_logfile_name = imaging_logfile_name
        
        #parameters of analysis
        self.show_tau_indexes = [1,2]
        self.tau_init = 'self'
        self.tau_length = 100
        
        #matching behavior with imaging
        print('Matching behavior with imaging...')
        prew_time=time.time()
        trigger_log_file_string = datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_'+name+'_'+task+'_TriggerLog.txt'
        exp_log_file_string = datapath + 'data/' + name + '_' + task + '/' + date_time + '/' + date_time + '_'+name+'_'+task+'_ExpStateMashineLog.txt'
        self.time_shift=LocateImaging(trigger_log_file_string, excel_file_path)
        print('   behavior matching done in',round(time.time()-prew_time,2), 'seconds')
        
        if np.isnan(self.time_shift):
            print('Behavior data could not be matched, terminating analysis!')
        else:            
            #load data (from s2p or manual drawn ROI excel sheet)
            self.Load_Data()
            
            #load frametimes from .xml logfile
            self.LoadImaging_times()
            
            #load beh
            self.Load_Beh(exp_log_file_string, plot=False)
            
            #check motion correction, threshold suspicious parts
            self.SetMotionCorrectionThreshold(threshold=25)
    
            #Preprocess, calculate properties
            self.Preprocess_and_CalculateProperties()
            
            #Z stuff
            self.Zmotion()
            
            #save spks for place-cell analysis
            self.save_F_spks(plot=False)
            
            # process trace
            self.F_s_c_z=self.ProcessTrace()#change it to: apply Z?
            self.F_s_c_z=preprocess(self.F_s_c_z,self.ops)#but then not use?
            
#            self.spks_test()
            #should recalculate properties here??
            
            #initialize peak comparison array here
            self.Paired_Data=[]#we will append all the peak data of a given trace-pair here
            
            #spike detection
            self.DetectSpikes(traces=self.F_s_c,baselines=self.baselines_corr,threshold_value=10, plot=False)
            
            #peak correlation
            self.CorrelatePeaks_test(toplot1=1, toplot2=2, plot=True)#check indexing type!
            #TODO
            
            self.PlotCorrelations(what='p')
            self.PlotCorrelations(what='a')
            
            #save results
            self.Save_to_Excel()
            
            #plot a given pair of suspicios tarces
            self.PlotPair(id_in1=1, id_in2=2)#indexing starts with 1 - ToDo: wrong pair given - catch error
                      
        

    def spks_test(self):
        fig, ax=plt.subplots(self.N, 2, sharex=True)
        for i in range(self.N):
            self.ops['tau']=self.Taus2[i]
            spks_temp=oasis(self.F_c, self.ops)[i,:]
            spks_smooth=oasis(self.F_s_c, self.ops)[i,:]
            ax[i,0].plot(self.x, self.F_c[i])
            ax[i,0].plot(self.x, spks_temp)
            ax[i,1].plot(self.x, self.F_s_c[i])
            ax[i,1].plot(self.x, spks_smooth)
        plt.show()
    
    def Load_Data(self):
        prew_time=time.time()
        print('Loading', self.data_source, 'data...')
        #we load the ops here as it contains information useful later
        F_string=self.suite2p_folder+'/F.npy'
        iscell_string=self.suite2p_folder+'/iscell.npy'
        ops_string=self.suite2p_folder+'/ops.npy'
        if os.path.isfile(ops_string)==True:
            ops =  np.load(ops_string, allow_pickle=True)
            ops = ops.item()
            self.ops=ops
        else:
            #this SetOps function makes an ops dictionary with the neccessary items with default values...
            print('ops file not found, using default parameters')
            self.ops=SetDefaultOpsParameters
        #MANUAL ROI
        if self.data_source=='manual':
            training_data_x = pd.read_excel(self.manual_data_excel_file)
            array = training_data_x.values
            Fim = np.transpose(array)
            
            HD = list(training_data_x.columns)
            hd_np=np.array(HD)
            
            #now we subselect the ROI here
            F_indexes=[]
            if self.ids!=-1:
                for i in range(hd_np.size):
                    try:
                        item=int(hd_np[i])
                        if item in self.ids:
                            F_indexes.append(i)
                            print('      ROI',item,'added')
                        else:
                            print('      ROI',item,'omitted')
                    except ValueError:
                        pass
            else:
                F_indexes=np.arange(1,Fim.shape[0]-1,2)
                
            self.header=[int(hd_np[i]) for i in F_indexes]
            self.F=Fim[F_indexes,:]
#            self.Fneu=Fim[Fneu_indexes,:]
            self.N=self.F.shape[0]
            self.M=self.F.shape[1]
            print('   Manual ROI data loaded, found ',self.N,'traces in the shape:',self.F.shape)
            
        # S2P ROI    
        if self.data_source=='s2p':
            #load
            F = np.load(F_string)
            iscell = np.load(iscell_string)
            iscell=iscell[:,0]
            #select the "cell" ROI-s only
            F_s2p =F[np.nonzero(iscell)[0]]
            #finalise
            self.F=F_s2p
            self.N=F_s2p.shape[0]
            self.N=F_s2p.shape[1]
            self.header=np.nonzero(iscell)[0]
            print('   suite2p data loaded',self.N, 'traces loaded found')
        
        print('   Traces loaded in',round(time.time()-prew_time,2), 'seconds')
        
    def LoadImaging_times(self):
        print('Loading imaging frames time...')
        prew_time=time.time() 
        offset = self.time_shift
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
#        print('   corrected offset:', corrected_offset, 'voltage_delay:', voltage_delay)        

        frames = imaging_logfile.getElementsByTagName('Frame')
        self.frame_times = np.zeros(len(frames)) # this is already in labview time
        for i in range(len(frames)):
            self.frame_times[i] = float(frames[i].attributes['relativeTime'].value) + corrected_offset
#        print('times from xml:', self.frame_times[0], self.frame_times[-1], self.frame_times[-1]-self.frame_times[0])
        #checking from here on
        if self.frame_times.size != self.M:
            print('number of frames not same in .xml file')
        self.x=self.frame_times-self.frame_times[0]
                 
        print('   LoadImaging_times time:',round(time.time()-prew_time,2), 'seconds')
    
    def LoadImage(self):
        image_path=self.suite2p_folder+'manual_roi_big.png'
        
        img2 = mpimg.imread(image_path)
        plt.figure('manual ROI')
        plt.imshow(img2)
        plt.title('ROI-s with ID-s')
        return img2
    
    def Load_Beh(self, exp_log_file_string, plot=False):
        prew_time=time.time()
        print('Loading and resampling relevant behavior...')
        position=[]
        exp_loop_timestamps=[]
        mazeID=[]
        lap=[]
        lick=[]
        reward=[]
        exp_log_file=open(exp_log_file_string)
        log_file_reader=csv.reader(exp_log_file, delimiter=',')
        next(log_file_reader, None)#skip the headers
        for line in log_file_reader:
            position.append(int(line[3]))
            exp_loop_timestamps.append(float(line[0]))
            lap.append(int(line[1]))
            mazeID.append(int(line[2]))
            if line[9]=='TRUE':
                lick.append(float(line[0]))
            if line[14]=='TrialReward':
                reward.append(float(line[0]))
        exp_look_timestamps=np.array(exp_loop_timestamps)
        position=np.array(position)
        mazeID=np.array(mazeID)
        lap=np.array(lap)
        lick=np.array(lick)
        reward=np.array(reward)
        
        resampled_time = self.frame_times
        
        #interpolation        
        F = interp1d(exp_look_timestamps, position) 
        self.pos_res = F(resampled_time)
        F = interp1d(exp_look_timestamps, mazeID) 
        self.maze_res = F(resampled_time)
        F = interp1d(exp_look_timestamps, lap) 
        self.lap_res = F(resampled_time)
        #select licks, rewards
        self.licks=lick[np.nonzero((lick>self.frame_times[0]) & (lick<self.frame_times[-1]))[0]]-self.frame_times[0]
        self.rewards=reward[np.nonzero((reward>self.frame_times[0]) & (reward<self.frame_times[-1]))[0]]-self.frame_times[0]
        
        
        self.speed_factor = 106.5 / 3500 ## constant to convert distance from pixel to cm
        self.dt_imaging = self.frame_times[1]-self.frame_times[0]
        
        speed = np.diff(self.pos_res) * self.speed_factor / self.dt_imaging # cm / s       
        speed_first = 2 * speed[0] - speed[1] # linear extrapolation: x1 - (x2 - x1)
        self.speed = np.hstack([speed_first, speed])
        resets = np.nonzero(self.speed<-500)[0]
        for i in resets:
            if (i != 0) & (i != self.M):
                self.speed[i]=(self.speed[i-2]+self.speed[i+2])/2
                
        print('   Speed calculated')
        
        if plot==True:
            plt.figure()
            plt.plot(resampled_time, self.pos_res)
            plt.plot(resampled_time, self.maze_res)
            plt.plot(resampled_time, self.lap_res)
            plt.scatter(self.licks, np.ones_like(self.licks))
            plt.scatter(self.rewards, np.ones_like(self.rewards)*100, c='r')
            plt.show()

        print('   Behavior loaded in',round(time.time()-prew_time,2), 'seconds')
    
    
    def SetMotionCorrectionThreshold(self, threshold):
        prew_time=time.time()
        print('Showing s2p motion corection with potential artefacts...')
        self.motion_threshold=threshold
        
        smove=np.power((np.power(self.ops['xoff'],2)+np.power(self.ops['yoff'],2)),1/2)
        
        self.motion_spikes=np.nonzero(smove>threshold)[0]

        fig, ax=plt.subplots(3, sharex=True)
        ax[0].set_title('suite2p x-y motion correction')
        ax[0].plot(self.x, self.ops['xoff'], c='r')
        ax[0].plot(self.x, self.ops['yoff'], c='g')
        ax[1].set_title('vectorial sum ')
        ax[1].plot(self.x, smove)
        ax[1].hlines(threshold, self.x[0],self.x[-1], linestyles='dashed')
        ax[1].scatter(self.motion_spikes*self.dt_imaging, np.ones(self.motion_spikes.shape)*threshold, c='r')
        ax[2].set_title('behavior')
        ax[2].set_ylabel('speed (cm/sec)', color='r')
        ax[2].set_xlabel('imaging frame')
        ax[2].plot(self.x, self.speed, c='r')            
        ax[2].tick_params(axis='y', labelcolor='r')
        axy=ax[2].twinx()
        axy.yaxis.set_label_position("right")
        axy.set_ylabel('position')
        axy.plot(self.x, self.pos_res, c='k')
        axy.vlines(self.licks,0,3000, colors='k', linewidth=0.5)
        axy.scatter(self.rewards,np.ones_like(self.rewards)*3600, c='b' )
        plt.show()
              
        print('   x-y motion done in',round(time.time()-prew_time,2), 'seconds')
        
    def Preprocess_and_CalculateProperties(self):
        prew_time=time.time()  

        print('Preprocessing traces...')
        baseline_percentile=10

        self.F_s_sharp=SmoothTraces(self.F,1)
        self.F_s=SmoothTraces(self.F)
        self.F_s_c, self.flow=preprocess(self.F_s, self.ops)#baseline corrected
        self.F_c=self.F-self.flow
#        fig, ax=plt.subplots(self.N,2, sharex=True)
#        flow_raw=preprocess(self.F, self.ops)[1]#baseline corrected
        
        
        self.baselines=CalculateBaselines(self.F_s, plot=False)
        self.baselines_corr=CalculateBaselines(self.F_s_c, plot=False)
        self.SDmin=CalculateBaselineSTDs(self.F_s_c, baseline_percentile, plot=False)
        self.SDmin_raw=CalculateBaselineSTDs(self.F, baseline_percentile, plot=False)
#        for i in range(self.N):
#            ax[i,0].plot(self.F_s_c[i,:], c='g')
#            ax[i,0].plot(self.F_c[i,:], linewidth=0.5)
#            ax[i,1].plot(self.F_s[i,:], c='g')
#            ax[i,1].plot(self.F[i,:], linewidth=0.5)
#            ax[i,1].plot(self.flow[i,:],c='k')
#            ax[i,1].plot(flow_raw[i,:],c='r')
#        plt.show()
        print('   Smoothing done')
        self.saturation=TestSaturation(self.F_s)
                
        print('   Preprocessing finished in',round(time.time()-prew_time,2), 'seconds')
    
    def Zmotion(self, show_decays_fitted=True):
        prew_time=time.time()
        ###################
        #1)under the baseline
        ###################
        print('Calculating Z-motion...')
        z=np.array([-1,-1])
        std_threshold=10#user interaction??
        
        fig, ax=plt.subplots(self.N, sharex=True, sharey=False)
        for i in range(self.N):
            
            threshold=self.baselines_corr[i]-std_threshold*self.SDmin[i]
            below=np.nonzero(self.F_c[i,:]<threshold)[0]
            
            if self.saturation[i]==0:    
                z=np.concatenate((z, below))
            ax[i].set_title('trace going under the baseline')
            ax[i].plot(self.F_c[i,:])
            ax[i].scatter(below, np.ones_like(below)*threshold, c='r')
            ax[i].hlines(threshold, 0, self.M, color='k')
            ax[i].hlines(self.baselines_corr[i], 0, self.M, color='y')
            
        plt.show()
        z=z[np.nonzero(z>-1)[0]]
        #hanyszor van?
        z=np.unique(z)
        self.UnderBaseline=z
        print('   under the baseline Z done')
        
        ################
        #2)fit Tau to decays, detect Z-motion...
        ################
        default_Tau=1.5#if we cannot fit spikes
        #these will be filled with the decays we want to fit, and the trace_id
        self.decays_to_fit=[]
        self.decay_id=[]
        self.decay_start=[]
        self.decay_smooth_peak=[]
        self.decay_ampl_before=[]
        self.decay_ampl_after=[]
        #this needs to be made optimal
        self.SaveDecays(traces=self.F_s_c, baselines=self.baselines_corr,threshold_value=10)
        
        #fit Tau for organised, normalised data
        self.organise_decays()
        self.hidden_index=0
        self.Taus=np.ones(self.N)
        for i in range(self.N):
            if self.decay_traces[self.hidden_index].size==1:
                self.Taus[i]=default_Tau
                print('for the ', i,'th trace default Tau is used' )
                self.hidden_index+=1
            else:
                Tau=minimize_scalar(self.fit_tau2, bounds=(0.1, 10), method='bounded')
                self.hidden_index+=1
                self.Taus[i]=Tau.x


        
        ###
        #fit Tau to all traces at once
        ###
        self.Taus2=np.ones(self.N)
        self.hidden_index=0
        self.plot_flag=True
        for i in range(self.N):
            if self.decay_traces[self.hidden_index].size==1:
                self.Taus2[i]=default_Tau
                self.hidden_index+=1
                print('for the ', i,'th trace default Tau is used still...' )
            else:
                Tau=minimize_scalar(self.fit_tau_at_once, bounds=(0.1, 10), method='bounded')
                self.hidden_index+=1
                self.Taus2[i]=Tau.x
        self.show_tau()
        self.FitS2p(sd_times=16)
    
    
    
    
   
#        ############################################################
#        #fit Tau for individual traces
#        self.hid_ind_index=0
#        for i in range(self.N):
#            #print(self.hid_ind_index, len(self.decay_id))
#            if self.hid_ind_index == len(self.decay_id):
#                break
#            plt.figure()
#            ax = plt.gca()
#            taus=[]
#            while self.decay_id[self.hid_ind_index]==i:
#                Tau=minimize_scalar(self.fit_tau_individual, bounds=(0.1, 10), method='bounded').x
#                taus.append(Tau)
#                
#                #scaled individual trace
#                trace= self.decays_to_fit[self.hid_ind_index]
#                if len(trace)>self.tau_length:
#                    trace=trace[:self.tau_length]
#                smooth_max=self.decay_smooth_peak[self.hid_ind_index]
#                ind=self.decay_id[self.hid_ind_index]
#                baseline=self.baselines[ind]
#                range_=smooth_max-baseline
#                trace=(trace-baseline)/range_
#                
#                #the fitted exponential
#                lengthening_factor=3            
#                alpha=1/Tau
#                if self.tau_init == 'one':
#                    y0=1
#                else:
#                    y0=trace[0]
#                expo=[]
#                time=np.linspace(0,(self.decays_to_fit[self.hid_ind_index].size*self.dt_imaging)*lengthening_factor, self.decays_to_fit[self.hid_ind_index].size*lengthening_factor)
#                for t in time:
#                    value=y0*math.exp(-t*alpha)
#                    expo.append(value)
#                np_expo=np.array(expo)
#
#                #plot
#                color = next(ax._get_lines.prop_cycler)['color']
#                plt.plot(trace, c=color)
#                plt.plot(np_expo, linewidth=1, c=color)
#                #exit if needed
#                self.hid_ind_index+=1
#                if self.hid_ind_index == len(self.decay_id):
#                    break
#            
#            title=str(round(np.average(taus),2))
#            plt.title(title)
#        ###################################################################################### 
        
        ###
        #showing all saved decays
        ###
        if show_decays_fitted==True:
            fig, ax=plt.subplots(self.N, sharex=True)
            unique, ms = np.unique(self.decay_id, return_counts=True)
            index=0
            decay_ampl_after = np.copy(self.decay_ampl_after)
            ids = np.unique(self.decay_id)
            for i in range(self.N):
                m=ms[i]
                ampl_after = decay_ampl_after[np.nonzero(self.decay_id==ids[i])[0]]
                threshold=np.median(ampl_after)
                used_for_fit=np.nonzero(ampl_after>=threshold)[0]            
                
                ax[i].plot(self.x, self.F_s_sharp[i,:], c='g')
                ax[i].plot(self.x, self.F_s[i,:], c='k')
                ax[i].plot(self.x, self.F[i,:], linewidth=0.5)
                ax[i].hlines(self.baselines[i]+10*self.SDmin[i], 0, self.M*self.dt_imaging)
                index2=0
                for j in range(m):
                    if type(self.decays_to_fit[index])==int:
                        index+=1
                    else:
                        length=len(self.decays_to_fit[index])
                        start=self.decay_start[index]
        
                        if index2 in used_for_fit:        
                            ax[i].plot(np.linspace(start,start+length-1, length)*self.dt_imaging, self.decays_to_fit[index], c='r')
                        else:
                            ax[i].plot(np.linspace(start,start+length-1, length)*self.dt_imaging, self.decays_to_fit[index], c='brown', linestyle='-.')
                        ax[i].scatter(self.decay_start[index]*self.dt_imaging, self.decay_smooth_peak[index], c='k')
                        ax[i].vlines(start*self.dt_imaging, self.decay_smooth_peak[index]-self.decay_ampl_before[index],self.decay_smooth_peak[index], colors='pink' )
                        ax[i].vlines((start+1)*self.dt_imaging, self.decay_smooth_peak[index]-self.decay_ampl_after[index],self.decay_smooth_peak[index], colors='brown' )
                        index+=1
                        index2+=1
                        

        print('   suite2p-fit based Z done')
                  
        print('   Z motion detection finished in',round(time.time()-prew_time,2), 'seconds')
##################################################################################
        #From here these are all called by other functions of the main class
################################################################################## 
            
    def SaveDecays(self, traces, baselines, threshold_value=10):
        print('      Saving decays to fit Tau')
        length_threshold=10

        for i in range(self.N):
            decays_saved=0
            threshold=baselines[i]+threshold_value*self.SDmin[i]
            secondary_peak_threshold=self.SDmin[i]*3
            F_s_c=traces[i,:]
            F=self.F[i,:]
            F_s=self.F_s_sharp[i,:]#it is only used to get a not so noise-prone max value
            
            #find peaks
            rise_index=np.nonzero((F_s_c[0:-1] < threshold)&(F_s_c[1:]>= threshold))[0]+1
            fall_index=np.nonzero((F_s_c[0:-1] > threshold)&(F_s_c[1:]<= threshold))[0]+1
            if rise_index.size>0 and fall_index.size>0:
                if rise_index.size> fall_index.size:
                    rise_index=rise_index[:-1].copy()
                if rise_index.size< fall_index.size:
                    fall_index=fall_index[1:].copy()
                if rise_index[0]>fall_index[0]:
                    rise_index=rise_index[:-1].copy()
                    fall_index=fall_index[1:].copy()
                    
                for j in range(rise_index.size):
                    trace_raw=F[rise_index[j]:fall_index[j]]
                    trace=F_s_c[rise_index[j]:fall_index[j]]
                    if trace.size>2:
                        #find local max and min
                        loc_max_p=argrelextrema(trace, np.greater)[0]
                        loc_max_p+=np.ones_like(loc_max_p)*rise_index[j]
                        loc_min_p=argrelextrema(trace, np.less)[0]
                        loc_min_p+=np.ones_like(loc_min_p)*rise_index[j]
                        loc_max_v=F_s_c[loc_max_p]
                        loc_min_v=F_s_c[loc_min_p]
                        ampl_before = np.zeros(loc_max_p.shape)
                        ampl_after = np.zeros(loc_max_p.shape)
                        for k in range(loc_max_p.size):
                            if k == 0:
#                                print(k, 'first')
                                ampl_before[0] = loc_max_v[0]-threshold 
                                if loc_min_p.size>0:
                                    ampl_after[0] = loc_max_v[0]-loc_min_v[0]
                                else:
                                    ampl_after[0] = loc_max_v[0]-threshold
                            elif k ==loc_max_p.size-1:
#                                print(k, 'last')
                                ampl_before[-1] = loc_max_v[-1]-loc_min_v[-1]
                                ampl_after[-1] = loc_max_v[-1]-threshold
                            else:
#                                print(k)
                                ampl_before[k] = loc_max_v[k]- loc_min_v[k-1]
                                ampl_after[k] = loc_max_v[k]-loc_min_v[k]
                                
                            
                        #delete potential local min at begining
                        if loc_min_p.size>0 and loc_min_p[0]<loc_max_p[0]:
                            print('Deleting first min')
                            loc_min_p=np.delete(loc_min_p, 0)
                            loc_min_v=np.delete(loc_min_v, 0)
                            
                        
                        if loc_max_p.size!=ampl_after.size:
                            print('kecske')
                        ampl_after2=ampl_after[np.nonzero(ampl_after>=0)[0]]
                        if ampl_after2.size!=ampl_after.size:
                            print('!!!!!minuszegy')
                        #remove  aftervalley fluctuation
                        for k in range(loc_min_v.size):
                            if loc_max_v.size>loc_min_v.size:#! van hogy 1-1
                                #TODO
                                # Todo: mi van ha az első peak picike épphogy threshold fölé megy, a következő is kicsi, emiatt negatív ampl_after jön létre
                                if loc_max_v[k+1]-loc_min_v[k]<secondary_peak_threshold:
#                                    last peak
                                    if k==loc_min_v.size-1:
                                        l=0
                                        while loc_max_p[k-l] == -1:
                                            l+=1
                                        ampl_after[k-l] = loc_max_v[k-l]-threshold  
                                        #at the end we can delete for sure
                                        loc_max_p[k+1]=-1
                                        loc_min_p[k]=-1
                                        ampl_after[k+1] = -1
                                        ampl_before[k+1] = -1
#                                    not last peak
                                    else:
                                        #get index of preceding peak (we will change the amplitude f this)
                                        l=0
                                        while loc_max_p[k-l] == -1:
                                            l+=1
                                        if (loc_max_v[k-l]-loc_min_v[k+1])>0:
                                            ampl_after[k-l] = loc_max_v[k-l]-loc_min_v[k+1]
                                            
                                            loc_max_p[k+1]=-1
                                            loc_min_p[k]=-1
                                            ampl_after[k+1] = -1
                                            ampl_before[k+1] = -1
                                        else:
                                            print('      slow climb - deleting peak',k)
                                            if k!=0:
                                                ampl_before[k] = loc_max_v[k]-loc_min_v[k-1] 
                                            else:
                                                ampl_before[k] = loc_max_v[k]-threshold
                                            
                                            ampl_after[k-l] = -1
                                            ampl_before[k-l] = -1
                                            loc_max_p[k-l] = -1
                                            loc_min_p[k-l] = -1
#                                        #plot if there is a negative amplitude
#                                        if (loc_max_v[k-l]-loc_min_v[k+1])<0:
#                                            print(k-l, k+1)
#                                            plt.figure()
#                                            plt.plot(trace)
#                                            plt.plot(np.ones_like(trace)*threshold)
#                                            plt.plot(np.ones_like(trace)*loc_max_v[k-l], c='r')
#                                            plt.plot(np.ones_like(trace)*loc_min_v[k+1], c='k')
#                                            plt.show()
      
                        #this may not be cost effective
                        loc_max_p=loc_max_p[np.nonzero(loc_max_p>=0)[0]]
                        loc_min_p=loc_min_p[np.nonzero(loc_min_p>=0)[0]]
                        ampl_after=ampl_after[np.nonzero(ampl_after>=0)[0]]
                        ampl_before=ampl_before[np.nonzero(ampl_before>=0)[0]]
                        #we use the only smoothed but not baselinecorrected trace for saving the smoothed peak value
                        ##this is a more "sharp" smoothing here                      
                        loc_max_v=F_s[loc_max_p]
                        
                        #save decays to fit Tau 
                        if loc_max_p.size!=ampl_after.size:
                            print(i, 'KECSKE')
                        for k in range(loc_max_p.size):
                            if k==loc_max_p.size-1:
                                raw=trace_raw[int(loc_max_p[k]-rise_index[j]):]
                            else:
                                raw=trace_raw[int(loc_max_p[k]-rise_index[j]):int(loc_min_p[k]-rise_index[j])]
                            if raw.size>length_threshold:
                                self.decays_to_fit.append(raw)
                                self.decay_id.append(i)
                                self.decay_smooth_peak.append(loc_max_v[k])
                                self.decay_start.append(loc_max_p[k])#this is saved only to be able to look at the peaks saved with the whole trace together
                                self.decay_ampl_after.append(ampl_after[k])
                                self.decay_ampl_before.append(ampl_before[k])
                                decays_saved+=1
            else:
                if decays_saved==0:
                    print('   for ',i,'th trace no decay found')
                    self.decays_to_fit.append(-1)
                    self.decay_id.append(i)
                    self.decay_smooth_peak.append(-1)
                    self.decay_start.append(-1)
                    self.decay_ampl_after.append(-1)
                    self.decay_ampl_before.append(-1)
            
    def organise_decays(self):
        print('      organising decays...')
        plot = False
        ids = np.unique(self.decay_id)
        self.decay_traces = []
        self.decay_traces_ind = []
        decays_to_fit = np.copy(self.decays_to_fit)#numpy arrays are better when it comes to indexing them
        decay_smooth_peak = np.copy(self.decay_smooth_peak)
        decay_ampl_after = np.copy(self.decay_ampl_after)
        for i in ids:
            if (i in self.show_tau_indexes):
                plot = True
            decays_used = decays_to_fit[np.nonzero(self.decay_id==i)[0]]
            smooth_maxes = decay_smooth_peak[np.nonzero(self.decay_id==i)[0]]
            ampl_after = decay_ampl_after[np.nonzero(self.decay_id==i)[0]]
            if len(decays_used) == 1:
                if decays_used == -1:
                    self.decay_traces.append(np.array([-1]))
                    self.decay_traces_ind.append(np.array([-1]))
            else:
                #we want to fit the 'big' traces
                threshold=np.median(ampl_after)
                
                #cut the noisy end
                for j in range(len(decays_used)):
                    decays_used[j] = decays_used[j][:self.tau_length]
                
                #select the bigger half of the traces
                remaining_indexes=np.nonzero(ampl_after>=threshold)[0]
                remain=decays_used[remaining_indexes]
                smooth_maxes=smooth_maxes[remaining_indexes]
                    
                #find longest trace
                max_length=0
                for j in range(len(remain)):
                    if remain[j].size> max_length:
                        max_length=remain[j].size
                decays_to_average=np.zeros((len(remain), max_length))
                
                #scale
                if plot==True:
                    plt.figure('showtau'+str(i))
                for j in range(len(remain)):
                    mini=self.baselines[i]
                    range_=smooth_maxes[j]-mini
                    remain[j]=(remain[j]-mini)/range_
                    
                    if plot==True:
                        plt.plot(remain[j])
                    #pad for same length with nans
                    if remain[j].size<max_length:
                        nans = np.ones(max_length-remain[j].size)
                        nans[:] = np.nan
                        remain[j]=np.append((remain[j]),(nans))
                    decays_to_average[j,:]=remain[j]
                
                self.decay_traces_ind.append(decays_to_average)
                trace=nan_average(decays_to_average)#nan_average is custom function able to average arrays containing nans 
                
                if plot== True:
                    plt.plot(trace, linewidth=2, c='r')
                    plt.show()
                
                self.decay_traces.append(trace)
            plot = False
                
    def fit_tau2(self, Tau):
        #calculate error with tau. 
        trace=self.decay_traces[self.hidden_index]
        time=np.linspace(0,(trace.size-1)*self.dt_imaging, trace.size)
        alpha=1/Tau
        if self.tau_init == 'one':
            y0=1
        else:
            y0=trace[0]
        expo=[]
        for t in time:
            value=y0*math.exp(-t*alpha)
            expo.append(value)
        np_expo=np.array(expo)
        
        return np.sum(np.abs(np_expo-trace))
    
    def fit_tau_individual(self, Tau):
        trace= self.decays_to_fit[self.hid_ind_index]
        if len(trace)>self.tau_length:
            trace=trace[:self.tau_length]
        time=np.linspace(0,(trace.size-1)*self.dt_imaging, trace.size)
        smooth_max=self.decay_smooth_peak[self.hid_ind_index]
        ind=self.decay_id[self.hid_ind_index]
        baseline=self.baselines[ind]
        range_=smooth_max-baseline
        trace=(trace-baseline)/range_
        #fitting
        alpha=1/Tau
        if self.tau_init == 'one':
            y0=1
        else:
            y0=trace[0]
        expo=[]
        for t in time:
            value=y0*math.exp(-t*alpha)
            expo.append(value)
        np_expo=np.array(expo)
        
        return np.sum(np.abs(np_expo-trace))
    
    def fit_tau_at_once(self, Tau):
        traces=self.decay_traces_ind[self.hidden_index]
        n,m=traces.shape
        time=np.linspace(0,(m-1)*self.dt_imaging, m)
        
        alpha=1/Tau
        if self.tau_init == 'one':
            y0=1
        else:
            y0=np.average(traces[:,0])
        expo=[]
        for t in time:
            value=y0*math.exp(-t*alpha)
            expo.append(value)
        np_expo=np.array(expo)
        diference=0
        length=0
        for i in range(n):
            for j in range(m):
                if np.isnan(traces[i,j]):
                    length=j
                    break
            diference=diference+np.sum(np.abs(np_expo[:length-1]-traces[i,:length-1]))
        
        return diference
    
    def FitS2p(self, sd_times=15, plot=True):
        print('      Fitting s2p output with calculated Tau, detecting Z artefacts...') 
#        n=4
        self.z_s2p_up=np.array(-1)
        self.z_s2p_down=np.array(-1)
        self.spks=np.zeros((self.N, self.M))
        if plot==True:
            fig, ax=plt.subplots(self.N, sharex=True)
        for i in range(self.N):
            min_std=self.SDmin[i]
            min_std_r=self.SDmin_raw[i]
            self.ops['tau']=self.Taus2[i]
            spks_temp=oasis(self.F_c, self.ops)
            self.spks[i,:]=spks_temp[i,:]
            decay=exp_decay(self.spks[i,:],self.ops['tau'])
            z=np.nonzero(self.F_c[i,:]<(decay-sd_times*min_std))[0]
            z_plus=np.nonzero(self.F_c[i,:]>(decay+(sd_times)*min_std))[0]
            
            if plot==True:    
                ax[i].plot(self.x,self.F_c[i,:], c='b')
                ax[i].plot(self.x,decay, c='g', linewidth=0.5)
                ax[i].plot(self.x,decay, c='magenta', linewidth=0.5)
                ax[i].plot(self.x,decay-sd_times*min_std, c='r', linewidth=0.5)
                ax[i].plot(self.x,decay+(sd_times)*min_std, c='r', linewidth=0.5)  
                ax[i].plot(self.x,decay-3*min_std_r, c='k', linewidth=0.5)
                ax[i].plot(self.x,decay+3*min_std_r, c='k', linewidth=0.5)  
                
                
                ax[i].scatter(z*self.dt_imaging,self.F_c[i,z], c='k')
                ax[i].scatter(z_plus*self.dt_imaging,self.F_c[i,z_plus], c='b')
                ax[i].plot(self.x,self.spks[i,:],c='orange')
                title='Tau ='+str(round(self.Taus2[i], 2))
                ax[i].set_title(title)
                if i==0:
                    ax[i].set_title('Z artefacts based on suite2p fit with \n'+title)
                    
            self.z_s2p_down=np.append(self.z_s2p_down, z)
            self.z_s2p_up=np.append(self.z_s2p_up, z_plus)
        plt.show()      
        self.z_s2p_down=np.unique(self.z_s2p_down)
        self.z_s2p_up=np.unique(self.z_s2p_up) 
    
    def show_tau(self):
        #this function is for visually checking the 
        #    fitted exponential with optimal Tau
        lengthening_factor=2
        ids=np.unique(self.decay_id)
        for i in ids:
            if (i in self.show_tau_indexes):
                plt.figure('showtau'+str(i))
                if (self.decay_traces[i].size!=1) & (self.decay_traces_ind[i].size!=1):
                    #average fit
                    Tau=self.Taus2[i]
                    alpha=1/Tau
                    if self.tau_init == 'one':
                        y0=1
                    else:
                        y0=self.decay_traces[i][0]
                    expo=[]
                    time=np.linspace(0,(self.decay_traces[i].size*self.dt_imaging)*lengthening_factor, self.decay_traces[i].size*lengthening_factor)
                    for t in time:
                        value=y0*math.exp(-t*alpha)
                        expo.append(value)
                    np_expo=np.array(expo)
                    
                    #fit all
                    Tau=self.Taus2[i]
                    alpha=1/Tau
                    if self.tau_init == 'one':
                        y0=1
                    else:
                        y0=np.average(self.decay_traces_ind[i][:,0])
                    expo2=[]
                    time=np.linspace(0,(self.decay_traces_ind[i].shape[1]*self.dt_imaging)*lengthening_factor, self.decay_traces_ind[i].shape[1]*lengthening_factor)
                    for t in time:
                        value=y0*math.exp(-t*alpha)
                        expo2.append(value)
                    np_expo2=np.array(expo2)
                
                
                
                    plt.plot(np_expo, linewidth=2, c='k')
                    plt.plot(np_expo2, linewidth=2, c='b')
                else:
                    plt.text(0,0.5, 'No good decays - default Tau is used ', size=20)
                title=str(round(self.Taus2[i],2))+','+ str(round(self.Taus2[i],2))
                plt.title(title)
        plt.show()
    
    def save_F_spks(self,plot=False):
        print('Saving "s2p-like" output here: ',self.save_path)
        prew_time=time.time()
        #saving
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.save(self.save_path+'spks.npy',self.spks)
        np.save(self.save_path+'F.npy',self.F)
        np.save(self.save_path + 'iscell.npy', np.ones((self.N,2)))
        if plot==True:
            fig, ax=plt.subplots(self.N)
            for i in range(self.N):
                ax[i].plot(self.F[i,:])
                ax[i].plot(self.spks[i,:], c='orange')
            plt.show()        
        print('   Saving finished in',round(time.time()-prew_time,2), 'seconds')
            
##################################################################################
        #old code
################################################################################## 
           
    def ProcessTrace(self):
        prew_time=time.time()
        print('Processing traces...')
#       #return traces (interpolated, cut)

        processed_data=np.zeros_like(self.F)
        #kill corrupted frames here
        print('   Z_under_baseine',self.UnderBaseline.size)
        print('   motion_spikes',self.motion_spikes.size)
        print('   Z_s2p_down',self.z_s2p_down.size)
        print('   Z_s2p_up',self.z_s2p_up.size)
        #we delete these, and not show in the figures
        self.frames_to_thrash=np.unique(np.concatenate((self.UnderBaseline, self.z_s2p_down)))
        #we just mark these
        self.suspicious_frames=np.unique(np.concatenate((self.motion_spikes, self.z_s2p_up)))
        self.frames_to_kill=np.unique(np.concatenate((self.frames_to_thrash,self.frames_to_thrash+1,self.frames_to_thrash-1,self.frames_to_thrash+2)))
        self.frames_sus=np.unique(np.concatenate((self.suspicious_frames,self.suspicious_frames+1,self.frames_to_thrash-1,self.frames_to_thrash+2)))
        self.frames_to_kill=self.frames_to_kill[np.nonzero(self.frames_to_kill<self.F.shape[1])[0]]
        self.frames_sus=self.frames_sus[np.nonzero(self.frames_sus<self.F.shape[1])[0]]

        fig, ax=plt.subplots(self.N, sharex=True)
        ax[0].set_title('Z artefacts cut out interpolation shown with red')
        for i in range(self.N):
            ROI=np.copy(self.F[i,:])
            #raw smoothed trace
            
            ROI[self.frames_to_kill]=np.NaN
            gaus_trace=NaN_Gaussian_filter(ROI,3)#gaus filtered trace                      
            
            temp=np.copy(gaus_trace)
            temp[self.frames_to_kill]=np.NaN
            #delete small islands of not corrupted frames here
            temp3=delete_small_islands(temp)
            int_gaus_trace=LinearInterpolate(temp3)# linearly interpolate NaNs in gaus filtered trace
            processed_data[i,:]=int_gaus_trace
            
            ax[i].plot(self.x, self.F[i,:], linewidth=0.3)
            ax[i].plot(self.x, gaus_trace, c='k')
            ax[i].plot(self.x, int_gaus_trace, c='r')
            ax[i].plot(self.x, temp, c='g')
            
        plt.show()
        #TODO
        #we do not want to kill the island nans probably
        frames_to_kill2=np.argwhere(np.isnan(gaus_trace)).flatten()
        self.frames_to_kill=np.unique(np.concatenate((self.frames_to_kill,frames_to_kill2)))
                  
        print('   Processing traces finished in',round(time.time()-prew_time,2), 'seconds')
        return processed_data
    
    def DetectSpikes(self, traces, baselines,threshold_value=10, plot=True):
        # return the max value, place
        prew_time=time.time()  

        print('Detecting spikes...')
        
        fig, ax=plt.subplots(self.N+1, sharex=True)
        ax[0].set_title('detected spikes')
        ax[self.N].set_title('behavior')
        ax[self.N].set_ylabel('speed (cm/sec)', color='r')
        ax[self.N].set_xlabel('imaging frame')
        ax[self.N].plot(self.x, self.speed, c='r')            
        ax[self.N].tick_params(axis='y', labelcolor='r')
        axy=ax[self.N].twinx()
        axy.yaxis.set_label_position("right")
        axy.set_ylabel('position')
        axy.plot(self.x, self.pos_res, c='k')
        axy.vlines(self.licks,0,3000, colors='k', linewidth=0.5)
        axy.scatter(self.rewards,np.ones_like(self.rewards)*3600, c='b' )
        
        self.loc_max_p=[]
        self.loc_max_v=[]
        self.loc_max_ampl=[]           
       
        for i in range(self.N):
          
            decays_saved=0
            max_places=np.zeros((0))
            max_values=np.zeros((0))
            amplitudes=np.zeros((0))

            F=np.copy(self.F[i,:])#just for plotting
            F_s=np.copy(traces[i,:])
            F_s=np.concatenate(([baselines[i]],F_s,[baselines[i]]))
            x=np.hstack(([-self.dt_imaging],self.x, [self.x[-1]+self.dt_imaging]))
            threshold=baselines[i]+threshold_value*self.SDmin[i]
            secondary_peak_threshold=self.SDmin[i]*3
            
            ax[i].plot(self.x, F-np.ones_like(F)*np.min(F), linewidth=0.3)
            ax[i].plot(x, F_s, c='g')
            ax[i].hlines(baselines[i], self.x[0], self.x[-1],colors='k', linewidth=0.7)
            ax[i].hlines(threshold, self.x[0], self.x[-1],colors='r', linewidth=0.7)
            
            #find peaks
            rise_index=np.nonzero((F_s[0:-1] < threshold)&(F_s[1:]>= threshold))[0]+1
            fall_index=np.nonzero((F_s[0:-1] > threshold)&(F_s[1:]<= threshold))[0]+1
            
            if rise_index.size>0 and fall_index.size>0:
                if rise_index.size> fall_index.size:
                    rise_index=rise_index[:-1].copy()
                if rise_index.size< fall_index.size:
                    fall_index=fall_index[1:].copy()
                if rise_index[0]>fall_index[0]:
                    rise_index=rise_index[:-1].copy()
                    fall_index=fall_index[1:].copy()
                for j in range(rise_index.size):
                    trace=F_s[rise_index[j]:fall_index[j]]
                    trace_raw=F[rise_index[j]:fall_index[j]]
                    #some baseline before the peak (to get real amplitudes for the first)
                    if rise_index[j]-10<0:
                        baseline_start=0
                    else:
                        baseline_start=rise_index[j]-10
                    baseline=np.min(F_s[baseline_start:rise_index[j]])
                    if baseline< baselines[i]:
                        baseline=baselines[i]
                    if trace.size>2:    
                        #find local max and min
                        loc_max_p=argrelextrema(trace, np.greater)[0]
                        loc_max_p+=np.ones_like(loc_max_p)*rise_index[j]
                        loc_min_p=argrelextrema(trace, np.less)[0]
                        loc_min_p+=np.ones_like(loc_min_p)*rise_index[j]
                        loc_max_v=F_s[loc_max_p]
                        loc_min_v=F_s[loc_min_p]
                        #delete potential local min at begining
                        if loc_min_p.size>0 and loc_min_p[0]<loc_max_p[0]:
                            loc_min_p=np.delete(loc_min_p, 0)
                            loc_min_v=np.delete(loc_min_v, 0)
                        #remove  aftervalley fluctuation
                        for k in range(loc_min_v.size):
                            if loc_max_v.size>loc_min_v.size:#! van hogy 1-1
                                if loc_max_v[k+1]-loc_min_v[k]<secondary_peak_threshold:
                                    loc_max_p[k+1]=-1
                                    loc_min_p[k]=-1
                        loc_max_p=loc_max_p[np.nonzero(loc_max_p>=0)[0]]
                        loc_max_v=F_s[loc_max_p]
                        loc_min_p=loc_min_p[np.nonzero(loc_min_p>=0)[0]]
                        loc_min_v=F_s[loc_min_p]

                        #calculate amplitudes                       
                        ampl=np.zeros_like(loc_max_p, dtype=float)
                        for k in range(loc_max_p.size):
                            if k==0:
                                ampl[k]=loc_max_v[k]-baseline

                                ax[i].vlines(loc_max_p[k]*self.dt_imaging, loc_max_v[k]-ampl[k],loc_max_v[k], colors='b')
                            else:
                                ampl[k]=loc_max_v[k]-loc_min_v[k-1]

                                ax[i].vlines(loc_max_p[k]*self.dt_imaging, loc_max_v[k]-ampl[k],loc_max_v[k], colors='b')
                                if ampl[k]<0:
                                    ax[i,0].scatter(loc_max_p[k]*self.dt_imaging,baseline, c='r', s=100)
                                if ampl[k]==0:
                                    ax[i,0].scatter(loc_max_p[k]*self.dt_imaging,baseline, c='r', s=100)
                        
                        ax[i].scatter(loc_max_p*self.dt_imaging, loc_max_v, c='k')
                        ax[i].scatter(loc_min_p*self.dt_imaging, loc_min_v, c='b', s=5)
                        
                        max_places=np.concatenate((max_places,loc_max_p))
                        max_values=np.concatenate((max_values,loc_max_v))
                        amplitudes=np.concatenate((amplitudes,ampl))
                        
                self.loc_max_p.append(max_places-1)
                self.loc_max_v.append(max_values)
                self.loc_max_ampl.append(amplitudes)
                if decays_saved==0:
                    for k in range(loc_max_p.size):
                        if k==loc_max_p.size-1:
                            raw=trace_raw[int(loc_max_p[k]-rise_index[j]):]
                        else:
                            raw=trace_raw[int(loc_max_p[k]-rise_index[j]):int(loc_min_p[k]-rise_index[j])]
                        self.decays_to_fit.append(raw)
                        self.decay_id.append(i)
                    
            else:
                print('      No peaks detected for',i, '. trace')
                self.loc_max_p.append(np.array([1, 2]))
                self.loc_max_v.append(np.array([1, 2]))
                self.loc_max_ampl.append(np.array([1, 2]))
        plt.show()        
                
        print('   Spikedetection finished in',round(time.time()-prew_time,2), 'seconds')
        
    def CorrelatePeaks_test(self, toplot1=-1, toplot2=-2, plot=True, same_peak_thr=15):
        prew_time=time.time()        
        print('Correlating peaks...')
        
        wind_size = 12
        volley_kern = 4
        if toplot1==-1 or toplot2==-1:
            specified_pair_plots=False
        else:
            specified_pair_plots=True
            fig, ax_sampl=plt.subplots(2, sharex=True)
        
        for m in range(self.N):
            for n in range(self.N-(m+1)):
#                print(m,n+m+1)
                index1=m
                index2=n+m+1
                
                trace1=self.F_s_c[index1,:]
                trace2=self.F_s_c[index2,:]
#                raw_trace1=self.F[index1,:]
#                raw_trace2=self.F[index2,:]
                loc_max_p1=np.copy(self.loc_max_p[index1])
                loc_max_v1=np.copy(self.loc_max_v[index1])
                loc_max_p2=np.copy(self.loc_max_p[index2])
                loc_max_v2=np.copy(self.loc_max_v[index2])
                amplitudes1=np.copy(self.loc_max_ampl[index1])
                amplitudes2=np.copy(self.loc_max_ampl[index2])
                
                pair_quality=[]#0=default, 1= only in first trace, 2= only in second trace, 3 = in both
                P1=[]
                P2=[]
                T1=[]
                T2=[]
                A1=[]
                A2=[]
                TV1=[]
                TV2=[]
                
                for i in range(loc_max_p1.size):
                    if loc_max_p2.size!=0:
                        nearest_peak=find_nearest(loc_max_p2,loc_max_p1[i])
                        nearest_peak_value=loc_max_v2[np.where(loc_max_p2==nearest_peak)[0]]
                        nearest_ampl=amplitudes2[np.where(loc_max_p2==nearest_peak)[0]]
                        #adatokkal való feltöltés ha nincs talált pár
                        if np.abs(nearest_peak- loc_max_p1[i])>same_peak_thr:
                            #define subtrace:
                            if (loc_max_p1[i]-wind_size)<0:
                                sub_start=0
                                print('subtrace at start')
                            else:
                                sub_start=int(loc_max_p1[i]-wind_size)
                            if (loc_max_p1[i]+wind_size)>trace2.size:
                                sub_fin=trace2.size-1
                                print('subtrace at end')
                            else:
                                sub_fin=int(loc_max_p1[i]+wind_size+1)
                            subtrace=trace2[sub_start:sub_fin]
                            sub_peaks=find_peaks(subtrace)[0]
                            #ha 0, akkor?? -> pontos érték adott pontban, ampl=0
                            #ha 1, egyértelmű
                            #ha több, mint 1 -> max magasságú, (vagy közelebbi)
                            if sub_peaks.size==0:
                                P1.append(loc_max_v1[i])
                                P2.append(trace2[int(loc_max_p1[i])])
                                T1.append(loc_max_p1[i])
                                T2.append(loc_max_p1[i])
                                A1.append(amplitudes1[i])
                                A2.append(0.01)
                                TV1.append(-1)
                                TV2.append(-1)
                                pair_quality.append(1)
                            else:
#                            define neg subtrace start
                                if (loc_max_p1[i]-volley_kern*wind_size)<0:
                                    sub_neg_start=0
                                else:
                                    sub_neg_start=int(loc_max_p1[i]-volley_kern*wind_size)
                                    
                                if sub_peaks.size==1:
                                    neg_subtrace=trace2[sub_neg_start:sub_fin]*-1# invert to be able to use find_peaks
                                    sub_volleys=find_peaks(neg_subtrace)[0]
                                    sub_volleys=sub_volleys[np.nonzero(sub_volleys<sub_peaks[0]+(volley_kern-1)*wind_size)[0]]#only volleys before are looked
                                    volley=max(sub_volleys)
                                    
                                    t2=int(sub_start+sub_peaks[0])
                                    tv2=int(sub_neg_start+volley)
                                    
                                    P1.append(loc_max_v1[i])
                                    P2.append(subtrace[sub_peaks[0]])
                                    T1.append(loc_max_p1[i])
                                    T2.append(t2)
                                    A1.append(amplitudes1[i])
                                    A2.append(trace2[t2]-trace2[tv2])
                                    TV1.append(-1)
                                    TV2.append(tv2)
                                    pair_quality.append(1)
                                    
#                                    plt.figure(str(n))
#                                    plt.plot(subtrace)
#                                    plt.scatter(sub_peaks, subtrace[sub_peaks])
#                                    plt.plot(np.linspace(0,len(neg_subtrace)-1,len(neg_subtrace))-3*wind_size, neg_subtrace*-1, linestyle='--', c='k')
#                                    plt.scatter(sub_volleys-3*wind_size, -1*(neg_subtrace[sub_volleys]), c='k')
#                                    plt.scatter(volley-3*wind_size,-1*(neg_subtrace[volley]), s=200)
#                                    plt.show()
                                    
                                if sub_peaks.size>1: #in this case the biggest peak is used
                                    sub_peak_index=np.argmax(subtrace[sub_peaks])
#                                    print(subtrace[sub_peaks], sub_peak_index)
                                    
                                    neg_subtrace=trace2[sub_neg_start:sub_fin]*-1# invert to be able to use find_peaks
                                    sub_volleys=find_peaks(neg_subtrace)[0]
                                    sub_volleys=sub_volleys[np.nonzero(sub_volleys<sub_peaks[sub_peak_index]+(volley_kern-1)*wind_size)[0]]#only volleys before the biggest peak are looked
                                    volley=max(sub_volleys)
                                    
                                    t2=int(sub_start+sub_peaks[sub_peak_index])
                                    tv2=int(sub_neg_start+volley)
                                    
                                    P1.append(loc_max_v1[i])
                                    P2.append(subtrace[sub_peaks[sub_peak_index]])
                                    T1.append(loc_max_p1[i])
                                    T2.append(t2)
                                    A1.append(amplitudes1[i])
                                    A2.append(trace2[t2]-trace2[tv2])
                                    TV1.append(-1)
                                    TV2.append(tv2)
                                    pair_quality.append(1)                                
                                
#                                if len(sub_peaks)>1:
#                                    plt.figure(str(n))
#                                    plt.plot(subtrace)
#                                    plt.scatter(sub_peaks, subtrace[sub_peaks])
#                                    plt.plot(np.linspace(0,len(neg_subtrace)-1,len(neg_subtrace))-wind_size, neg_subtrace*-1, linestyle='--', c='k')
#                                    plt.scatter(sub_volleys-wind_size, -1*(neg_subtrace[sub_volleys]), c='k')
#                                    plt.show()

                        else:
                            P1.append(loc_max_v1[i])
                            P2.append(nearest_peak_value)
                            T1.append(loc_max_p1[i])
                            T2.append(nearest_peak)
                            A1.append(amplitudes1[i])
                            A2.append(nearest_ampl)
                            TV1.append(-1)
                            TV2.append(-2)
                            pair_quality.append(0)
                #dealing with peaks in trace 2 not found yet
                for i in range(loc_max_p2.size):
                    if loc_max_p2[i] in T2:
                        pass
                    else:
                        if loc_max_p1.size!=0:
                            nearest_peak=find_nearest(loc_max_p1,loc_max_p2[i])
                            nearest_peak_value=loc_max_v1[np.where(loc_max_p1==nearest_peak)[0]]
                            nearest_ampl=amplitudes1[np.where(loc_max_p1==nearest_peak)[0]]
                            if np.abs(nearest_peak- loc_max_p2[i])>same_peak_thr:
                                #define subtrace:
                                if (loc_max_p2[i]-wind_size)<0:
                                    sub_start=0
                                    print('subtrace at start')
                                else:
                                    sub_start=int(loc_max_p2[i]-wind_size)
                                if (loc_max_p2[i]+wind_size)>trace1.size:
                                    sub_fin=trace1.size-1
                                    print('subtrace at end')
                                else:
                                    sub_fin=int(loc_max_p2[i]+wind_size+1)
                                subtrace=trace1[sub_start:sub_fin]
                                sub_peaks=find_peaks(subtrace)[0]
                                if sub_peaks.size==0:
                                    P1.append(trace1[int(loc_max_p2[i])])
                                    P2.append(loc_max_v2[i])
                                    T1.append(loc_max_p2[i])
                                    T2.append(loc_max_p2[i])
                                    A1.append(0.01)
                                    A2.append(amplitudes2[i])
                                    TV1.append(-1)
                                    TV2.append(-1)
                                    pair_quality.append(2)
                                else:
    #                            define neg subtrace start
                                    if (loc_max_p2[i]-volley_kern*wind_size)<0:
                                        sub_neg_start=0
                                        print('volley_at start')
                                    else:
                                        sub_neg_start=int(loc_max_p2[i]-volley_kern*wind_size)
                                        
                                    if sub_peaks.size==1:
                                        neg_subtrace=trace1[sub_neg_start:sub_fin]*-1# invert to be able to use find_peaks
                                        sub_volleys=find_peaks(neg_subtrace)[0]
                                        if sub_volleys.size==0:
                                            print('kecske', sub_neg_start, sub_fin)
                                            plt.figure()
                                            plt.plot(neg_subtrace*-1)
                                            plt.show()
                                        sub_volleys=sub_volleys[np.nonzero(sub_volleys<sub_peaks[0]+(volley_kern-1)*wind_size)[0]]#only volleys before are looked
                                        volley=max(sub_volleys)
                                        
                                        t1=int(sub_start+sub_peaks[0])
                                        tv1=int(sub_neg_start+volley)
                                        
                                        P1.append(subtrace[sub_peaks[0]])
                                        P2.append(loc_max_v2[i])
                                        T1.append(t1)
                                        T2.append(loc_max_p2[i])
                                        A1.append(trace1[t1]-trace1[tv1])
                                        A2.append(amplitudes2[i])
                                        TV1.append(tv1)
                                        TV2.append(-1)
                                        pair_quality.append(2)
                                    if sub_peaks.size>1: #in this case the biggest peak is used
                                        sub_peak_index=np.argmax(subtrace[sub_peaks])                                        
                                        neg_subtrace=trace1[sub_neg_start:sub_fin]*-1# invert to be able to use find_peaks
                                        sub_volleys=find_peaks(neg_subtrace)[0]
                                        sub_volleys=sub_volleys[np.nonzero(sub_volleys<sub_peaks[sub_peak_index]+(volley_kern-1)*wind_size)[0]]#only volleys before the biggest peak are looked
                                        volley=max(sub_volleys)
                                        
                                        t1=int(sub_start+sub_peaks[sub_peak_index])
                                        tv1=int(sub_neg_start+volley)
                                        
                                        P1.append(subtrace[sub_peaks[sub_peak_index]])
                                        P2.append(loc_max_v2[i])
                                        T1.append(t1)
                                        T2.append(loc_max_p2[i])
                                        A1.append(trace1[t1]-trace1[tv1])
                                        A2.append(amplitudes2[i])
                                        TV1.append(tv1)
                                        TV2.append(-1)
                                        pair_quality.append(2)

                                
                            else:
#                                print('found new valid pair')
                                P1.append(nearest_peak_value)
                                P2.append(loc_max_v2[i])
                                T1.append(nearest_peak)
                                T2.append(loc_max_p2[i])
                                A1.append(nearest_ampl)
                                A2.append(amplitudes2[i])
                                TV1.append(-1)
                                TV2.append(-1)
                                pair_quality.append(0)
                            if (toplot1==index1 and toplot2== index2) or (toplot1==index2 and toplot2== index1):
                                pass

                #convert to numpy
                P1=np.array(P1)
                P2=np.array(P2)
                T1=np.array(T1)
                T2=np.array(T2)
                A1=np.array(A1)
                A2=np.array(A2)
                TV1=np.array(TV1)
                TV2=np.array(TV2)
                pair_quality=np.array(pair_quality)
                if specified_pair_plots==True:
                    if (toplot1==index1 and toplot2== index2) or (toplot1==index2 and toplot2== index1):
                        difis=np.nonzero(np.abs(T1-T2)>100)[0]
                        ax_sampl[0].plot(trace1, c='g')
                        ax_sampl[0].scatter(T1, P1, c=pair_quality)
                        ax_sampl[0].vlines(T1, P1-A1, P1, colors='b')
                        ax_sampl[0].scatter(T1[difis], P1[difis]+1, c='r')
                        ax_sampl[1].plot(trace2, c='g')
                        ax_sampl[1].scatter(T2,P2, c=pair_quality)
                        ax_sampl[1].vlines(T2, P2-A2, P2, colors='b')
                        ax_sampl[1].scatter(T2[difis], P2[difis]+1, c='r')
                #pump into pairwise dataframe
                self.Paired_Data.append([index1, index2, P1, P2, A1, A2, T1, T2, pair_quality])


        print('   correlation calcuated in',round(time.time()-prew_time,2), 'seconds')
                    
    
    def PlotCorrelations(self, what='b'):
        prew_time=time.time()
        fig, ax=plt.subplots(self.N+1,self.N+1)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        ax[0,0].imshow(self.man_image)
                        
        for i in range(self.N):
            title=self.header[i]
                
            ax[0,i+1].plot(self.F_s_c[i,:], c='gray')
            ax[0,i+1].scatter(self.loc_max_p[i], self.loc_max_v[i], c='k', s=5)
            ax[0,i+1].set_title(title)
            ax[i+1,0].plot(self.F_s_c[i,:], c='gray')
            ax[i+1,0].scatter(self.loc_max_p[i], self.loc_max_v[i], c='k', s=5)
            ax[i+1,0].set_title(title)

        for i in range(len(self.Paired_Data)):
            index1=self.Paired_Data[i][0]
            index2=self.Paired_Data[i][1]
            P1=self.Paired_Data[i][2]
            P2=self.Paired_Data[i][3]
            A1=self.Paired_Data[i][4]
            A2=self.Paired_Data[i][5]
            T1=self.Paired_Data[i][6]
            T2=self.Paired_Data[i][7]
            pair_quality=self.Paired_Data[i][8]
            P1=normalise(P1, np.amin(self.F_s_c[index1,:]))
            P2=normalise(P2, np.amin(self.F_s_c[index2,:]))
            A1=normalise(A1, np.amin(self.F_s_c[index1,:]))
            A2=normalise(A2, np.amin(self.F_s_c[index2,:]))
            
            corrupt_peaks1=np.intersect1d(T1, self.frames_to_kill, return_indices=True)[1]
            corrupt_peaks2=np.intersect1d(T2, self.frames_to_kill, return_indices=True)[1]
            shaky_peaks1 =np.intersect1d(T1, self.frames_sus, return_indices=True)[1]
            shaky_peaks2 =np.intersect1d(T2, self.frames_sus, return_indices=True)[1]
            corrupt_peaks=np.unique(np.concatenate((corrupt_peaks1, corrupt_peaks2)))
            shaky_peaks=np.unique(np.concatenate((shaky_peaks1, shaky_peaks2)))
            
            zero=np.nonzero(pair_quality==0)[0]
            one=np.nonzero(pair_quality==1)[0]
            two=np.nonzero(pair_quality==2)[0]
            
            e_zero=np.intersect1d(shaky_peaks, zero)
            e_one=np.intersect1d(shaky_peaks, one)
            e_two=np.intersect1d(shaky_peaks, two)
            
            
            x_zero=np.intersect1d(corrupt_peaks, zero)
            x_one=np.intersect1d(corrupt_peaks, one)
            x_two=np.intersect1d(corrupt_peaks, two)
            
            g_zero=np.setdiff1d(one, np.concatenate((e_zero, x_zero)))
            g_one=np.setdiff1d(one, np.concatenate((e_one, x_one)))
            g_two=np.setdiff1d(two, np.concatenate((e_two, x_two)))
            
#            print('zero')
#            print(np.intersect1d(g_zero, x_zero))
#            print('one')
#            print(np.intersect1d(g_one, x_one))
#            print('two')
#            print(np.intersect1d(g_two, x_two))
#            print(zero.size, x_zero.size, g_zero.size, e_zero.size)
#            print(zero)
            
            if what=='a':
                W1=A1
                W2=A2
                fig = plt.gcf()
                fig.suptitle("Relative 'local' Ampl.", fontsize=14)
            if what=='p':
                W1=P1
                W2=P2
                fig = plt.gcf()
                fig.suptitle("Absolute Ampl. value", fontsize=14)
                
            ax[index1+1,index2+1].scatter(W1[g_zero],W2[g_zero], c='gray', s=5)#
            ax[index1+1,index2+1].scatter(W1[e_zero],W2[e_zero], edgecolors='gray', s=5, facecolors='none')#
            ax[index1+1,index2+1].plot(W1[x_zero],W2[x_zero], "x", c='gray')
            
            ax[index1+1,index2+1].scatter(W1[g_one],W2[g_one], c='blue', s=5)#
            ax[index1+1,index2+1].scatter(W1[e_one],W2[e_one], edgecolors='blue', s=5, facecolors='none')#
            ax[index1+1,index2+1].plot(W1[x_one],W2[x_one], "x",c='blue')
            
            ax[index1+1,index2+1].scatter(W1[g_two],W2[g_two], c='red', s=5)#
            ax[index1+1,index2+1].scatter(W1[e_two],W2[e_two], edgecolors='red', s=5, facecolors='none')
            ax[index1+1,index2+1].plot(W1[x_two],W2[x_two], "x", c='red')
            
        print('   correlation plotted in ',round(time.time()-prew_time,2), 'seconds')
        
    def PlotPair(self,id_in1, id_in2):
        id1=self.header.index(id_in1)
        id2=self.header.index(id_in2)
        for i in range(len(self.Paired_Data)):
            trace1_id=self.Paired_Data[i][0]
            trace2_id=self.Paired_Data[i][1]
            #look for desired pair        
            if trace1_id==id1 and trace2_id==id2 or trace1_id==id2 and trace2_id==id1:
                trace1=self.F_s_c[trace1_id,:]
                trace2=self.F_s_c[trace2_id,:]
                raw_trace1=self.F[trace1_id,:]
                raw_trace2=self.F[trace2_id,:]
                raw_trace1=raw_trace1-np.average(raw_trace1)
                raw_trace2=raw_trace2-np.average(raw_trace2)
                P1=self.Paired_Data[i][2]
                P2=self.Paired_Data[i][3]
                A1=self.Paired_Data[i][4]
                A2=self.Paired_Data[i][5]
                T1=self.Paired_Data[i][6]
                T2=self.Paired_Data[i][7]
                
                fig, ax=plt.subplots(3, sharex=True)
                fig = plt.gcf()
                fig.suptitle("red: absolute, green: relative Ampl.", fontsize=14)
                
                ax[0].plot(self.x, trace1, c='g')
                ax[0].plot(self.x, raw_trace1, linewidth=0.5, c='b')
                ax[0].set_title(str(id1))#na ez nem biztos, hogy jó így!
                ax[0].scatter(T1*self.dt_imaging, P1, c='k')
                ax[0].vlines(T1*self.dt_imaging, P1-A1, P1, color='k' )
                ax[1].plot(self.x, trace2, c='g')
                ax[1].plot(self.x, raw_trace2, linewidth=0.5, c='b')
                ax[1].set_title(str(id2))
                ax[1].scatter(T2*self.dt_imaging, P2, c='k')
                ax[1].vlines(T2*self.dt_imaging, P2-A2, P2, color='k' )
                
                p1=normalise(P1, np.amin(self.F_s_c[trace1_id,:]))
                p2=normalise(P2, np.amin(self.F_s_c[trace2_id,:]))
                a1=normalise(A1, np.amin(self.F_s_c[trace1_id,:]))
                a2=normalise(A2, np.amin(self.F_s_c[trace2_id,:]))
                
                ax[2].vlines(T1*self.dt_imaging, 0, np.abs(p1-p2), color='r')
                ax[2].vlines((T1+3)*self.dt_imaging, 0, np.abs(a1-a2), color='g')
                plt.show()

    def Save_to_Excel(self):
        print('saving results to excel...')
        name=self.suite2p_folder.split('/')[-2]
        print('   filename: ',name,'.xlsx')
        filepath=self.save_path+'/'+name+'traces.xlsx'
        frame_info=np.zeros_like(self.F[0,:])
        frame_info[self.frames_sus]=1
        frame_info[self.frames_to_kill]=2
        
        dfa = pd.DataFrame (np.transpose(self.F))
        dfb=pd.DataFrame (np.transpose(self.F_s_c))
        dfc=pd.DataFrame(frame_info)
        head=self.header
        wb = openpyxl.Workbook()
        wb.save(filepath)
        
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:          
            #traces
            d = {'raw traces': [self.N, self.N]}
            name1=pd.DataFrame(data=d)
            name1.to_excel(writer,sheet_name="traces",index=False, startrow=0, startcol=0)
            dfa.to_excel(writer,sheet_name="traces",index=False, startrow=0, startcol=1, header=head)
            d = {'processed traces': [self.N, self.N]}
            name1=pd.DataFrame(data=d)
            name1.to_excel(writer,sheet_name="traces",index=False, startrow=0, startcol=self.N+1)
            dfb.to_excel(writer,sheet_name="traces",index=False, startrow=0, startcol=self.N+2, header=head)
            d = {'frames': ['0=good','1=suspicious', '2=corrupted']}
            name1=pd.DataFrame(data=d)
            name1.to_excel(writer,sheet_name="traces",index=False, startrow=0, startcol=self.N*2+2)
            dfc.to_excel(writer,sheet_name="traces",index=False, startrow=0, startcol=self.N*2+3, header=' ')
            #spikes
            startcol=0
            for i in range(len(self.Paired_Data)):
                ROI1_id=self.header[self.Paired_Data[i][0]]
                ROI2_id=self.header[self.Paired_Data[i][1]]                                
                d = {'ROI '+str(ROI1_id)+' vs '+str(ROI2_id): [ROI1_id,ROI2_id]}
                name=pd.DataFrame(data=d)
                name.to_excel(writer,sheet_name="spikes",index=False, startrow=0, startcol=startcol)
                startcol+=1
                
                head=[str(ROI1_id)+' abs',str(ROI2_id)+' abs',str(ROI1_id)+' rel',str(ROI2_id)+' rel', str(ROI1_id)+' time',str(ROI2_id)+' time', 'Pair quality']
                temp=np.zeros((self.Paired_Data[i][2].size,7))
                temp[:,0]=self.Paired_Data[i][2]
                temp[:,1]=self.Paired_Data[i][3]
                temp[:,2]=self.Paired_Data[i][4]
                temp[:,3]=self.Paired_Data[i][5]
                temp[:,4]=self.Paired_Data[i][6]
                temp[:,5]=self.Paired_Data[i][7]
                temp[:,6]=self.Paired_Data[i][8]
                dfp=pd.DataFrame(temp)
                dfp.to_excel(writer,sheet_name="spikes",index=False, startrow=0, startcol=startcol, header=head)    
                startcol+=7
        print('   saving done')
                              
            
##########################################
#custom defined functions:                 
##########################################

def exp_decay(spks, tau):
    a=1/tau
    delta_T=1/30
    output=np.zeros_like(spks,dtype='float')
    sp_ind=np.nonzero(spks>0)[0]
    for i in sp_ind:
        decay=[]
        t=0
        y0=spks[i]
        value=1
        while int(value)!=0:
            value=y0*math.exp(-t*a)
            decay.append(value)
            t+=delta_T
        decay=np.array(decay)
        if i+decay.size>output.size:
            decay=decay[0:output.size-i]
        output[i:i+decay.size]+=decay
    return output

def delete_small_islands(trace):
    output=np.copy(trace)
    bool_trace=np.isnan(trace)
    nan_isl_length=0
    nan_isl_start=-1
    prev=False
    i=0
    for i in range(bool_trace.size) :
        if bool_trace[i]==True and prev==False and nan_isl_start!=-1:
            if nan_isl_length<30: 
                output[nan_isl_start:i]=np.nan
            nan_isl_length=0
        if bool_trace[i]==False and prev==True:
            nan_isl_start=i
            nan_isl_length+=1
        if bool_trace[i]==False and prev==False: 
            nan_isl_length+=1
        prev=bool_trace[i]
        i+=1
    return output

def DistanceToLine(m, b, x0, y0):#unused, but may come in handy!
    #calculate a points distance to astraight line defined by parameters m and b
    x=(x0+m*y0-m*b)/(1+m*m)
    y=m*x+b
    d=np.sqrt((x0-x)**2+(y0-y)**2)
    output=[x,y,d]
    return output

def normalise(array, min_value=np.NaN):#numpy array needed
    #normalise all elements of a given array between 0 and 1
    #you can give a manual min
    if np.isnan(min_value):
        ext_array=array
    else:
        ext_array=np.append(array, min_value) 
    maxi=np.max(array)
    mini=np.min(ext_array)
    the_range=maxi-mini
    output=(array-mini)/the_range
    return output
    
def find_nearest(array, value):
    #gives back the closest element of an array to a given value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def gaussian_filter(trace, sdfilt):
    #gaussian filter method, copied from UBI's previous scripts
    N=10
    sampling_time=1
    xfilt=np.arange(-N*sdfilt, N*sdfilt+sampling_time, sampling_time)
    filt=np.exp(-(xfilt**2)/(2*(sdfilt**2)))
    filt=filt/sum(filt)

    temp=np.hstack([np.repeat(trace[0],N*sdfilt),trace, np.repeat(trace[-1], N*sdfilt)])
    result=np.convolve(temp,filt,mode='valid')
    
    return result
#TODO
def NaN_Gaussian_filter(trace, sdfilt):
    #same as gaussian_filter but can handle NaNs to an extent
    N=10
    sampling_time=1
    xfilt=np.arange(-N*sdfilt, N*sdfilt+sampling_time, sampling_time)
    filt=np.exp(-(xfilt**2)/(2*(sdfilt**2)))
    filt=filt/sum(filt)
    
    temp=np.hstack([np.repeat(trace[0],N*sdfilt),trace, np.repeat(trace[-1], N*sdfilt)])
    out=np.zeros_like(trace)
    for i in range(trace.size):
        snip=temp[i:i+2*(N*sdfilt)+1]
        
        nans=np.argwhere(np.isnan(snip))
        nonnan_snip=np.delete(snip,nans)

        filt_nan=np.delete(filt,nans)
        filt_nan=filt_nan/sum(filt_nan)
        if nans.size<N*sdfilt:
            out[i]=np.sum(nonnan_snip*filt_nan)
        else:
            out[i]=np.nan
#            print('too many NaNs!')
    return out

def LinearInterpolate(A):
    #linearly interpolate NaN values in a given numpy array (borrowed from other sorce)
    
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]

    A[np.isnan(A)] = np.interp(x, xp, fp)
    
    return A

def CalculateBaselineSTDs(traces, baseline_percentile, plot=False):
    kernel_size=101
    
    if plot==True:
        fig, ax=plt.subplots(traces.shape[0])
        
    SSDmin = np.zeros((traces.shape[0]))  
    for i in range(traces.shape[0]):
        SSD=np.zeros(traces[i,:].size-kernel_size)
        for j in range(traces[i,:].size-kernel_size):
            SSD[j]=np.std(traces[i,j:j+kernel_size])
        SSDmin[i]=np.percentile(SSD,baseline_percentile)
        
        if plot==True:
            x=np.linspace(kernel_size, traces.shape[1], traces.shape[1]-kernel_size) -50
            ax[i].plot(traces[i,:], zorder=0)
            ax[i].plot(x,SSD, c='k', zorder=5)
            ax[i].hlines(SSDmin[i], 0,traces.shape[1], zorder=10)
            plt.show()
    return SSDmin
            
def CalculateBaselines(traces, plot=False):
    baselines = np.zeros((traces.shape[0])) 
    for i in range(traces.shape[0]):
        hist=np.histogram(traces[i,:], bins=100)            
        max_value=max(hist[0])
        max_index=list(hist[0]).index(max_value)
        baselines[i]=hist[1][max_index]
    if plot==True:
        fig, ax=plt.subplots(traces.shape[0])
        for i in range(traces.shape[0]):
            ax[i].plot(traces[i,:], zorder=0)
            ax[i].hlines(baselines[i], 0, traces.shape[1], zorder=5)
        plt.show()
    return baselines

def SmoothTraces(traces, sd=3):  
    out=np.zeros_like(traces)
    for i in range(traces.shape[0]):
        ROI = np.copy(traces[i,:])
        out[i,:] = gaussian_filter(ROI, sd)
    return out

def TestSaturation(traces):
    print('   Testing saturation')
    saturation=np.zeros(traces.shape[0])
    for i in range(traces.shape[0]):
        trace = traces[i]
        hist=np.histogram(trace, bins=10)
        last_bins_ratio=(hist[0][-1]+hist[0][-2])/np.sum(hist[0])
        first_bin_ratio=hist[0][0]/np.sum(hist[0])
        if last_bins_ratio>0.05 and last_bins_ratio<0.1:
            print(i+1, '. trace likely saturated (or no meaningful spikes)')
            saturation[i]=1
            
        if last_bins_ratio>0.1:
            print(i+1, '. trace saturated (or no meaningful spikes)')
            saturation[i]=2
        
        if first_bin_ratio<0.1 and saturation[i]==0:
            print(i+1,'. trace too active or has heavy artefacts or inactive')
            saturation[i]=3
    
    print('      saturation checked')        
    return saturation

def SetDefaultOpsParameters():
    ##################
    #we should not really run without the original ops file, this is just a getaround
    ##################
    tau = 1.4 # timescale of indicator
    fs = 30.0 # sampling rate in Hz
    neucoeff = 0.7 # neuropil coefficient
    # baseline correction
    baseline = 'maximin' # take the running max of the running min after smoothing with gaussian
    sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
    win_baseline = 60.0 # in seconds, window in which to compute max/min filters default: 60
    batch_size = 500
    
    ops = {'tau': tau, 'fs': fs, 'neucoeff': neucoeff,
           'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline,
           'batch_size': batch_size}
    return ops

def nan_average(data):
    ii, jj=data.shape
    x_av=np.zeros(jj)
    x_counter=np.zeros(jj)
    for i in range(ii):
        for j in range(jj):
            if np.isnan(data[i,j])==False:
                x_av[j]+=data[i,j]
                x_counter[j]+=1
    ave=x_av/x_counter
    
    return ave


############
#copies of some of suite2p-s inner functions
#date of copy: ~2020.08-09 month
############

@njit(['float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32'], cache=True)
def oasis_trace(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on a single neuron """
    NT = F.shape[0]
    g = -1./(tau * fs)

    it = 0
    ip = 0

    while it<NT:
        v[ip], w[ip],t[ip],l[ip] = F[it],1,it,1
        while ip>0:
            if v[ip-1] * np.exp(g * l[ip-1]) > v[ip]:
                # violation of the constraint means merging pools
                f1 = np.exp(g * l[ip-1])
                f2 = np.exp(2 * g * l[ip-1])
                wnew = w[ip-1] + w[ip] * f2
                v[ip-1] = (v[ip-1] * w[ip-1] + v[ip] * w[ip]* f1) / wnew
                w[ip-1] = wnew
                l[ip-1] = l[ip-1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1
        
    s[t[1:ip]] = v[1:ip] - v[:ip-1] * np.exp(g * l[:ip-1])


@njit(['float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32'], parallel=True, cache=True)
def oasis_matrix(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on many neurons parallelized with prange  """
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], tau, fs)

def oasis(F, ops):
    """ computes non-negative deconvolution
    no sparsity constraints
    
    Parameters
    ----------------
    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence
    ops : dictionary
        'batch_size', 'tau', 'fs'
    Returns
    ----------------
    S : float, 2D array
        size [neurons x time], deconvolved fluorescence
    """
    NN,NT = F.shape
    F = F.astype(np.float32)
    batch_size = ops['batch_size']
    S = np.zeros((NN,NT), dtype=np.float32)
#    V = np.zeros((NN,NT), dtype=np.float32)
#    W = np.zeros((NN,NT), dtype=np.float32)
#    T = np.zeros((NN,NT), dtype=np.float32)
#    L = np.zeros((NN,NT), dtype=np.float32)            
    for i in range(0, NN, batch_size):
        f = F[i:i+batch_size]
        v = np.zeros((f.shape[0],NT), dtype=np.float32)
        w = np.zeros((f.shape[0],NT), dtype=np.float32)
        t = np.zeros((f.shape[0],NT), dtype=np.int64)
        l = np.zeros((f.shape[0],NT), dtype=np.float32)
        s = np.zeros((f.shape[0],NT), dtype=np.float32)
        oasis_matrix(f, v, w, t, l, s, ops['tau'], ops['fs'])
        S[i:i+batch_size] = s
    
    return S

def preprocess(F,ops):
    """ preprocesses fluorescence traces for spike deconvolution
    baseline-subtraction with window 'win_baseline'
    
    Parameters
    ----------------
    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence
    ops : dictionary
        'baseline', 'win_baseline', 'sig_baseline', 'fs',
        (optional 'prctile_baseline' needed if ops['baseline']=='constant_prctile')
    
    Returns
    ----------------
    F : float, 2D array
        size [neurons x time], baseline-corrected fluorescence
    """
    sig = ops['sig_baseline']
    win = int(ops['win_baseline']*ops['fs'])
    if ops['baseline']=='maximin':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = filters.minimum_filter1d(Flow,    win)
        Flow = filters.maximum_filter1d(Flow,    win)
    elif ops['baseline']=='constant':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = np.amin(Flow)
    elif ops['baseline']=='constant_prctile':
        Flow = np.percentile(F, ops['prctile_baseline'], axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    F = F - Flow

    return F, Flow

######################
#call main class
######################
#issues with this but incomplete files
datapath = os.getcwd() + '/'

####
#no files currently
####
#date_time = '2021-07-27_08-18-40'# date and time of the beh session
#name = 'srb121' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb121_TSeries-07272021-0758-001.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb121_TSeries-07272021-0758-001_Cycle00001_VoltageRecording_001.csv'

####
#2 roi
####
#date_time = '2021-03-24_09-15-23' # date and time of the beh session
#name = 'srb081' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb081_210324_006/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb081_TSeries-03242021-0752-006.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb081_TSeries-03242021-0752-006_Cycle00001_VoltageRecording_001.csv'

####
# nagyon sok ROI, egyik inaktív
####
#error
#date_time = '2021-03-12_13-25-52' # date and time of the beh session
#name = 'rs077' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/rs077_210312/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'rs077_TSeries-03122021-0822-001.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'rs077_TSeries-03122021-0822-001_Cycle00001_VoltageRecording_001.csv'

####
# nagyon lassúnak tűnő jelek, 2 fölötti Tau
####
#date_time = '2021-02-05_11-32-15' # date and time of the beh session
#name = 'srb067' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb210205/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb057_TSeries-02052021-0813-002.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb057_TSeries-02052021-0813-002_Cycle00001_VoltageRecording_001.csv'

####
# hatalmas base-zaj
#####
#date_time = '2021-02-11_08-58-02' # date and time of the beh session
#name = 'srb067' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb210211/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb067_210211-002.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb067_210211-002_Cycle00001_VoltageRecording_001.csv'

####
# sok ROI van kinullázott is közte - correlations ezen nézve
####
#date_time = '2021-02-12_08-48-36' # date and time of the beh session
#name = 'srb067' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb210212/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb067_210212-003.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb067_210212-003_Cycle00001_VoltageRecording_001.csv'

####
# sok ROI, a fele inaktív
####
#date_time = '2021-02-15_09-13-31' # date and time of the beh session
#name = 'srb067' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb210215/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb068_TSeries-02152021-0821-003.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb068_TSeries-02152021-0821-003_Cycle00001_VoltageRecording_001.csv'

####
# elég rossz jel/zaj arány
####
#date_time = '2021-03-25_08-55-48' # date and time of the beh session
#name = 'srb081' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb081_210325/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb081_TSeries-03252021-0835-003.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb081_TSeries-03252021-0835-003_Cycle00001_VoltageRecording_001.csv'

####
# érdekes jelek, de amúgy nem rossz!
####
date_time = '2021-06-23_08-14-01' # date and time of the beh session
name = 'srb106' # mouse name
task = 'NearFarLong' # task name
suite2p_folder = datapath + 'data/' + name + '_imaging/srb106_210623_t2/' # suite2p folder
imaging_logfile_name = suite2p_folder + 'srb106__210623-002.xml'
TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb106__210623-002_Cycle00001_VoltageRecording_001.csv'

####
# lassú Tau-k itt is eléggé
####
#date_time = '2021-06-23_08-33-52' # date and time of the beh session
#name = 'srb106' # mouse name
#task = 'NearFarLong' # task name
#suite2p_folder = datapath + 'data/' + name + '_imaging/srb106_210623_t3/' # suite2p folder
#imaging_logfile_name = suite2p_folder + 'srb106__210623-003.xml'
#TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'srb106__210623-003_Cycle00001_VoltageRecording_001.csv'


#ManD = ProcessManualRoiData(datapath, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, name, task, date_time, ids=[1,2,3,7])
#ManD.PlotPair(2,3)
def vcorrcoef(X,y): # numpy's corrcoeff is just stupid
        Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
        ym = np.mean(y)
        r_num = np.sum((X-Xm)*(y-ym),axis=1)
        r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
        #sometimes there are completely flat ROI-s found by suite2p, wich would result in division by zero, we counter that here
        #   for these 'invalid' ROI-s r = 0 is returned
        val_ind=np.nonzero(r_den)[0]
        r = np.zeros(np.shape(r_num))
        
        r[val_ind] = r_num[val_ind]/r_den[val_ind]
        
        return r

def SelectRois(suite2p_folder, ids=-1):
    prew_time=time.time()
    manual_data_excel_file = suite2p_folder+'manual_roi_data.xlsx'

    #MANUAL ROI
    training_data_x = pd.read_excel(manual_data_excel_file)
    print('inside', round(time.time()-prew_time,2))
    
    array = training_data_x.values
    
    Fim = np.transpose(array)
    
    HD = list(training_data_x.columns)
    hd_np=np.array(HD)
    
    #now we subselect the ROI here
    F_indexes=[]
    if ids!=-1:
        for i in range(hd_np.size):
            try:
                item=int(hd_np[i])
                if item in ids:
                    F_indexes.append(i)
                    print('      ROI',item,'added')
                else:
                    print('      ROI',item,'omitted')
            except ValueError:
                pass
    else:
        F_indexes=np.arange(1,Fim.shape[0]-1,2)
        
    header=[int(hd_np[i]) for i in F_indexes]
    F=Fim[F_indexes,:]
    F_s=SmoothTraces(F)
#    F_s=np.copy(F)
    
    N=F.shape[0]
    plt.figure(figsize=(10, 10))
    plt.scatter(-1,0, s=0.01)
    plt.scatter(N,N+1, s=0.01)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    exp_factor=4
    for i in range(N):
        F_= np.delete(F_s, (i), axis=0)
        line = vcorrcoef(F_, F_s[i,:])
        line = np.insert(line,i,0.001)
        plt.annotate(str(header[i]),(-1-0.15, N-i-0.1))
        plt.annotate(str(header[i]),(i-0.15,N+1))
        for j in range(N):
            plt.scatter(j,N-i, s=(line[j]**exp_factor)*1000, c='b')
            if line[j]>0.4:
                plt.annotate(str(round(line[j],2)),(j-0.25, N-i))
                
    fig, ax=plt.subplots(N, sharex=True)
    for i in range(N):
        ax[i].plot(F[i,:])
        ax[i].text(.5,.5,header[i],horizontalalignment='center',transform=ax[i].transAxes)
#            title=str(header[i])
#            ax[i].set_title(title)
    plt.show()

prew_time=time.time()            
print('Standup time:',round(prew_time-start_time,2), 'seconds')
 
#SelectRois(suite2p_folder)

#SelectRois(suite2p_folder, ids=[1,2,3,7])

#print('SelectRois time:',round(time.time()-prew_time,2), 'seconds')

ManD = ProcessManualRoiData(datapath, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, name, task, date_time, ids=[1,2,3,7])

#ManD = ProcessManualRoiData(datapath, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, name, task, date_time)
print('Total runtime:',round(time.time()-start_time,2), 'seconds')