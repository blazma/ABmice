# -*- coding: utf-8 -*-

"""
@author: djustus

04/2016	        		include digital out for lick sensor & reward
					port0/line7 for lick sensor
					use port0/line6 for reward TTL

03/2017                         switch to incremental rotation sensor

03/2017                         include up to 10 switchable corridor designs

10/2017                         include different reward locations for corridor designs

10/2017                         made one standard file for different monitor designs                              

10/2017                         Monitor settings in this file

10/2017                         Multiple reward sections per VR

@author: Balazs B Ujfalussy - balazs.ujfalussy@gmail.com

03/2018                         Left and Right reward ports

03/2018                         calibration removed - no tactile cues on belt

03/2018                         Different corridor lengths 

03/2018                         Handling automatically mice data and training stages stage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

05/2018							Major change: script only generating scene from input position

05/2019                         Major change: corridor definitions moved to a python Corridor_list class from the excel file


QUESTIONS: 

"""

import ogre.renderer.OGRE as ogre
import SampleFramework as sf
import ogre.io.OIS as OIS
import math

import time
import datetime

now = datetime.datetime.now()
print('script started', now.strftime("%Y-%m-%d %H:%M:%S"))

import random
import ctypes
import numpy as np
import pickle

from subprocess import Popen, PIPE, STDOUT
import subprocess

import Tkinter, tkFileDialog
import sys
import os


sys.path.append('C:\LuigsNeumann_Treadmill\PyOgre\packages_2.7\ogre\renderer\OGRE')
from sf_OIS import *

#from openpyxl import Workbook
#from openpyxl import load_workbook
sys.path.append('C:\Users\LN-Treadmill\Desktop\MouseData')
from Corridors import *
from Stages import *

import socket #Mate

global wDir
wDir = os.getcwd()

#import wx
import re

# sys.path.append(wDir+'\..\..\..\IO\ReadWriteNI')
# sys.path.append(wDir+'\..\..\..\IO\ReadWriteXLS')

TCP_IP = '127.0.0.1' #Mate
TCP_PORT = 5005  #Mate
BUFFER_SIZE = 15000 #Mate: Maximum 3000 message is read out from the buffer at once

now = datetime.datetime.now()
print('packages loaded', now.strftime("%Y-%m-%d %H:%M:%S"))


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Mate
s.bind((TCP_IP, TCP_PORT))  #Mate
print('Listening for connection')  #Mate
s.listen(1)  #Mate

global conn
global addr


conn, addr = s.accept()  #Mate
print('Connection established')
# 
# global file_path
# file_path = wDir+'\..\..\..\Configurations\parameters_5stage_2AFC.xlsx'

#from PyDAQmx import Task
#from PyDAQmx.DAQmxTypes import *
#from PyDAQmx.DAQmxConstants import *

# from PyDAQmxWriteDig import _write2NIDev
# from PyDAQmxReadDig import _readfromNIDev
# from PyDAQmxWriteAnalog import _write2NIDev_analog
# from PyDAQmxReadAnalog import _readFromNIDev_analog

# from Write2xls import _stopRecording_rew2_VR, _writeRecording_rew2_VR, _record_rew2_VR, _initRecording
# from readFromXLS import _readParameters


now = datetime.datetime.now()
print('connection established', now.strftime("%Y-%m-%d %H:%M:%S"))


def readVR_Position_from_LabView():
    global conn
    global addr
    global msg
    VR_Pos = 'M99P9999'
    try:
        msg = conn.recv(BUFFER_SIZE)
        VR_Pos = msg.decode().split(',')[-2] # -2 because the last product of the spilt is ","
    except (IndexError):
        pass
    except (socket.error):
        conn.close()
        print("Connection closed")
        s.listen(1)
        print('Listening for new connection')
        conn, addr = s.accept()
        try:
            msg = conn.recv(BUFFER_SIZE)          
            VR_Pos = msg.decode().split(',')[-2]
        except (IndexError):
            pass
    return VR_Pos

# add images in the textures folder to the materials 

def addTextures(imgname):
    with open('C:\LuigsNeumann_Treadmill\PyOgre\VR\media\materials\scripts\Wall.material',"r+") as textures:
        existing = textures.read()

        if existing.find('material %s' %imgname)==-1:               # not yet existing
            textures.seek(0)
            newblock='material %s\n{\n\ttechnique\n\t{\n\t\tpass\n\t\t{\n\n\t\t\ttexture_unit\n\t\t\t{\n\t\t\t\ttexture %s\n\t\t\t}\n\t\t}\n\t}\n}\n\n\n' %(imgname,imgname)
            
            textures.write(existing+newblock)


for file in os.listdir('C:\LuigsNeumann_Treadmill\PyOgre\VR\media\materials\\textures'):
    if file.endswith('.png'):
        addTextures(file)
    if file.endswith('.jpg'):
        addTextures(file)

###################################################s


class FrameListener(sf.FrameListener, OIS.MouseListener):
    # 
    def _updateStatistics(self):
        self._setGuiCaption('Core/DebugText', Application.debugText)
        
    def _chooseSceneManager (self):
        self.sceneManager = self.root.createSceneManager (ogre.ST_EXTERIOR_CLOSE, 'TerrainSM')

    def __init__(self, renderWindow, camera, sceneManager, current_stage, task, bufferedMouse = True):
        sf.FrameListener.__init__(self, renderWindow, camera)
        self.current_stage = current_stage
        self.task = task
        OIS.MouseListener.__init__(self)

        now = datetime.datetime.now()
        print('Framelistener started', now.strftime("%Y-%m-%d %H:%M:%S"))

#==============================================================================
# Define Experiment Settings here:
#==============================================================================

        self.MoveMode = "Line"
        self.ExpMode = "Corridor"
        
#==============================================================================
        
        self.sceneManager = sceneManager        
        
        App=Application(self.current_stage, self.task)
 #       App=Application()
            
        self.toggle = 0
        self.mouseDown = False

        # wb = load_workbook(file_path)
        # ws = wb.get_sheet_by_name("VR0")

        infilename = 'C:\\Users\\LN-Treadmill\\Desktop\\MouseData\\images\\' + self.task + '_stages.pkl'
        input_file = open(infilename, 'rb')
        stage_list = pickle.load(input_file)
        input_file.close()

        infilename = 'C:\\Users\\LN-Treadmill\\Desktop\\MouseData\\images\\' + self.task + '_corridors.pkl'
        input_file = open(infilename, 'rb')
        corridor_list = pickle.load(input_file)
        input_file.close()

#        corridor_list.print_images()                

        ########### Corridor Properties                
        self.numMonitors = 3 #int(ws.cell('C46').value)          # number of monitors
        # self.num_of_VR = ws.cell('C44').value
        self.num_of_VR = stage_list.stages[self.current_stage].N_corridors + 1 # number of VRs - corridors

        corridor_lengths = np.zeros(self.num_of_VR-1) ## no grey zone ...
        for i in range(self.num_of_VR-1):
            i_corridor = stage_list.stages[self.current_stage].corridors[i]
            corridor_lengths[i] = int(round((corridor_list.corridors[i_corridor].length - corridor_list.corridors[i_corridor].width) / 6144.0 * 240))

        if (len(np.unique(corridor_lengths)) > 1):
            print('corridor lenth is not unique!')

        ########### mouse movement
        # self.corridorLength = 240 # 6 x 40 (width)
        self.corridorLength = corridor_lengths[0]
        self.SpeedFactor = self.corridorLength / 3499.0
#        self.SpeedFactor = self.corridorLength / 34990.0

        self.num_WallTexture = []
        self.num_FloorTexture = []
        self.num_CeilTexture = []

        self.wallLeft = []
        self.wallRight = []
        self.wallEnd = []
        self.floorPattern = []
        self.ceilPattern = []
        
        self.wall_height = []
        self.width = []
        
        self.corridorStart = []
        self.corridorEnd = []
     
        self.nodes = []

        self.startingPos = []             # Start of corridor
        self.endingPos = []               # Position of Jump back

        for i in range(self.num_of_VR): # we add the grey zone as well...
            if (i==0):
                i_corridor = 0
            else:
                i_corridor = stage_list.stages[self.current_stage].corridors[i-1]

            self.wallLeft.append(corridor_list.corridors[i_corridor].left_image)
            self.wallRight.append(corridor_list.corridors[i_corridor].right_image)
            self.wallEnd.append(corridor_list.corridors[i_corridor].end_image)
            self.floorPattern.append(corridor_list.corridors[i_corridor].floor_image)
            self.ceilPattern.append(corridor_list.corridors[i_corridor].ceiling_image)

            if not self.floorPattern[i]:
                self.floorPattern[i] = '';
            if not self.ceilPattern[i]:
                self.ceilPattern[i] = '';

            self.num_WallTexture.append(1)
            self.num_FloorTexture.append(1)
            self.num_CeilTexture.append(1)

            self.wall_height.append(30)
            self.width.append(40)

            self.corridorStart.append(0)
            self.corridorEnd.append(self.corridorLength)
         
            nodes = []
            nodes.append((0,self.corridorStart[i]))          #x, z coordinates, start position, index 0
            nodes.append((0,self.corridorEnd[i]))            #x, z coordinates, start position, index 1
            self.nodes.append(nodes)

            self.startingPos.append(0)             
            self.endingPos.append(self.corridorLength)               
                    
        self.bufferedMouse = bufferedMouse
    
        self.translateVector = ogre.Vector3(0.0,0.0,0.0)  # translation vector - will be filled by the rotation encoder - MoveMouse

        self.signNode = sceneManager.getRootSceneNode().createChildSceneNode((0,20,1024))
        
        self.camNode = camera.parentSceneNode.parentSceneNode
        self.camYawNode = camera.parentSceneNode
        self.camera = camera
        self.height = 10;    #Sichthöhe - in the previous version

        self.readInterval = 0.001                                # ! default: 0.04 s
        self.currentTime = time.time()
        self.lastRead = self.currentTime

        self.initVR = True
        self.test_VRs = False #Mate:True # we will cycle through all VRs
        self.VRs = range(self.num_of_VR)
        self.VR_Pos = 'M00P0000'
        self.Position_From_Labview = int(self.VR_Pos[4:8]) * self.SpeedFactor
        self.VR_From_Labview = int(self.VR_Pos[1:3])
        self.currentVR = self.VR_From_Labview

        # # subprocess to read the rotation encoder and write its values to a pipe (stdout) at every 40 ms (hardcoded, unrelated to self.recInterval)
        # self.p1 = Popen(['python', 'C:\LuigsNeumann_Treadmill\IO\ReadWriteNI\sub_count_Incr2.py',str(self.RotationDev),'0'], stdout = PIPE, stderr = STDOUT,bufsize = -1)
        self.output1O = 0

        now = datetime.datetime.now()
        print('framelistener initialised', now.strftime("%Y-%m-%d %H:%M:%S"))


## I don't know whether we need this block...                                  
    def _setupInput(self):
        # Initialize OIS.
        windowHnd = self.renderWindow.getCustomAttributeInt("WINDOW")
        self.InputManager = OIS.createPythonInputSystem([("WINDOW", str(windowHnd))])
 
        self.Keyboard = self.InputManager.createInputObjectKeyboard(OIS.OISKeyboard, self.bufferedKeys)
        self.Mouse = self.InputManager.createInputObjectMouse(OIS.OISMouse, self.bufferedMouse)
        self.Joy = False

            
    def frameStarted(self, frameEvent):
        if self.initVR:
            now = datetime.datetime.now()
            print('initialising VRs', now.strftime("%Y-%m-%d %H:%M:%S"))
            for i in range (self.num_of_VR-1,-1,-1):
                self._destroyVR()
                self.currentVR = i
                print "create %d" %(i)
                self._createVR(i)
        
        if self.initVR:
            self._destroyVR()
            self._createVR(self.currentVR)
            self.camNode.setPosition(0,self.height,0.05)        # reset location
            print ('VR created:' +  str(self.currentVR))
            self.output1O = self.startingPos[self.currentVR]
            self.initVR = False
            now = datetime.datetime.now()
            print('VRs ready', now.strftime("%Y-%m-%d %H:%M:%S"))

        self.currentTime = time.time()
        camPos = self.camNode.getPosition() # current camera position - 

        # Procedures for closing if the renderWindow is closed...
        if(self.renderWindow.isClosed()):
            return False
                
        #######################################################
    	## set the position and VR

        if (self.test_VRs):
    	        #######################################################
    	    	## demo position and VR - cycle through all VRs
             if ((self.currentTime - self.lastRead) > self.readInterval):
                 self.Position_From_Labview = self.Position_From_Labview + 1
#                 print("new position", self.Position_From_Labview)
    
        else :
            #######################################################
            ## reading the external VR and position input
            self.VR_Pos = readVR_Position_from_LabView()
            next_maze = int(self.VR_Pos[1:3])            
            if (next_maze != 99):
                self.Position_From_Labview = int(self.VR_Pos[4:8]) * self.SpeedFactor
                #print(self.Position_From_Labview)
                self.VR_From_Labview = int(self.VR_Pos[1:3])
                
        if (self.VR_From_Labview not in self.VRs): # we will close for any invalid character
            if(self.VR_From_Labview==98):
                exit()            
            return False

        ## start a new VR
        if (self.VR_From_Labview != self.currentVR):
            self._treadmillEndTrial()

        #######################################################
        ## checking the displacement of the camera
#        self._MouseMoved(frameEvent) # line 717 
#        self._Line_Move(self.signNode)
        self.camNode.setPosition(0,self.height,self.Position_From_Labview)        # reset location

        #######################################################
        ## position reset to the corridor ...
        ## 			... at the beginning
        if camPos.z < self.startingPos[self.currentVR]:
            self.camNode.setPosition(0,self.height,self.startingPos[self.currentVR])        # reset location

        ## 			...  at the end
        if camPos.z > self.endingPos[self.currentVR]:
            self.camNode.setPosition(0,self.height,self.endingPos[self.currentVR])        # reset location
            if (self.test_VRs):
                self.VR_From_Labview = self.VR_From_Labview + 1
                self.Position_From_Labview = self.startingPos[self.VR_From_Labview]
 
        return True
        
    def _treadmillEndTrial(self):
        self._destroyVR()
        #print (self.currentVR, 'VR destroyed')
        
        self.currentVR = self.VR_From_Labview
        #print (self.currentVR, 'creating VR')
        self._createVR(self.currentVR)
        self.camNode.setPosition(0,self.height,self.Position_From_Labview)        # reset location
        self.output10 = self.Position_From_Labview
        #print ('VR created:' +  str(self.currentVR))
           
#    def _MouseMoved(self, frameEvent):
#    # reads input to create the translateVector 
#        orientation = self.camYawNode.getOrientation()*ogre.Vector3().UNIT_Z
#
#        output1 = self.Position_From_Labview
#        # output1 = int(self.p1.stdout.readline())
#        # print "t = ", str(time.time()), " ch1 = ", str(output1)#, " ch2 = ", str(output2)
#
#        dist = (output1-self.output1O) # current - past
#        print("new pos:"+str(output1))
#        print("old pos:"+str(self.output1O))
#        self.output1O = output1
#        self.translateVector = (0,0,dist * self.SpeedFactor * orientation[2])
##        print("new distance", dist)
#
#
#    def _Line_Move(self,signNode):
#        camPos = self.camNode.getPosition()
#        camPos2D = (10*round(camPos.x/10),10*round(camPos.z/10))
#        self.orient = self.camYawNode.getOrientation()*ogre.Vector3().UNIT_X # this block seems to be never used...
#        o = self.orient
#
#		# move forward 5 pixels if we are beyond the start point        
##        if camPos2D[1]<self.corridorStart[self.currentVR]+5:
##            self.camNode.setPosition(0,self.height,self.corridorStart[self.currentVR]+5)
#        
#        # moves the camera width a translateVector that come from the function MouseMoved - line 720 - called in FrameStarted around line 482
#        self._moveCamera()
##        print("new position", self.Position_From_Labview)
#                             
#        
#    def _moveCamera(self):
#        try:
#            self.camNode.translate(self.translateVector)
##            print("new position", self.translateVector)
#        except AttributeError:
#            self.camNode.moveRelative(self.translateVector)
            

    #################################################
    ##########    Building the corridor    ##########
    #################################################

    def _buildCeil(self, sceneManager, width, wall_height, texture, num_texture):
        name = "ceil"
        name_floor = "fl_%s_%d" %(name,self.currentVR)
        name_plane = "pl_%s_%d" %(name,self.currentVR)
        name_Ent = "Ent_%s_%d" %(name,self.currentVR)

        plane = ogre.Plane()
        plane.normal = -ogre.Vector3().UNIT_Y
        plane.d = 0
        
        ogre.MeshManager.getSingleton().createPlane(name_floor,
                                      ogre.ResourceGroupManager.DEFAULT_RESOURCE_GROUP_NAME, plane,
                                      self.corridorEnd[self.currentVR]-self.corridorStart[self.currentVR]+width, width,
                                      10, 10,
                                      False,
                                      1, num_texture, 1,
                                      ogre.Vector3().UNIT_X)
        name_Ent = sceneManager.createEntity(name_plane, name_floor)
        name_Ent.setMaterialName(texture)        
        sceneManager.getRootSceneNode().createChildSceneNode((0,wall_height-1,(self.corridorStart[self.currentVR]+self.corridorEnd[self.currentVR])/2)).attachObject(name_Ent)
        return;


    def _buildFloor(self, sceneManager, width, texture, num_texture):
        name = "floor"
        name_floor = "fl_%s_%d" %(name,self.currentVR)
        name_plane = "pl_%s_%d" %(name,self.currentVR)
        name_Ent = "Ent_%s_%d" %(name,self.currentVR)

        plane = ogre.Plane()
        plane.normal = ogre.Vector3().UNIT_Y
        plane.d = 0
        
        ogre.MeshManager.getSingleton().createPlane(name_floor,
                                      ogre.ResourceGroupManager.DEFAULT_RESOURCE_GROUP_NAME, plane,
                                      self.corridorEnd[self.currentVR]-self.corridorStart[self.currentVR]+width, width,
                                      10, 10,
                                      False,
                                      1, num_texture, 1,
                                      ogre.Vector3().UNIT_X)
        name_Ent = sceneManager.createEntity(name_plane, name_floor)
        name_Ent.setMaterialName(texture)        
        sceneManager.getRootSceneNode().createChildSceneNode((0,1,(self.corridorStart[self.currentVR]+self.corridorEnd[self.currentVR])/2)).attachObject(name_Ent)
        return;

 
    def _buildWall(self, sceneManager, name, richtung, start, stop, wall_height, texture, num_texture): #richtung: direction - position
        name_floor = "fl_%s_%d" %(name,self.currentVR)
        name_plane = "pl_%s_%d" %(name,self.currentVR)
        name_Ent = "Ent_%s_%d" %(name,self.currentVR)
        
        if (richtung==1):
            plane = ogre.Plane()
            plane.normal = ogre.Vector3().UNIT_X
            plane.d = 0
        elif richtung==2:
            plane = ogre.Plane()
            plane.normal = -ogre.Vector3().UNIT_X
            plane.d = 0
        elif richtung==3:
            plane = ogre.Plane()
            plane.normal = ogre.Vector3().UNIT_Z
            plane.d = 0
        elif richtung==4:
            plane = ogre.Plane()
            plane.normal = -ogre.Vector3().UNIT_Z
            plane.d = 0
        
        vec = [start[0]-stop[0],start[1]-stop[1]]
        vecLen = math.sqrt(vec[0]*vec[0]+vec[1]*vec[1])
        
        ogre.MeshManager.getSingleton().createPlane(name_floor,
                                      ogre.ResourceGroupManager.DEFAULT_RESOURCE_GROUP_NAME, plane,
                                      vecLen, wall_height,
                                      10, 10,
                                      False,
                                      1, num_texture, 1,
                                      ogre.Vector3().UNIT_Y)
        name_Ent = sceneManager.createEntity(name_plane, name_floor)
        name_Ent.setMaterialName(texture)        
        sceneManager.getRootSceneNode().createChildSceneNode(((start[0]+stop[0])/2,wall_height/2,(start[1]+stop[1])/2)).attachObject(name_Ent)
        return;
        
    
    def _buildCorridor (self, sceneManager, name, start, stop, end_start, end_stop, front, width, wall_height, texture_left, texture_right, texture_end, num_texture):
   # self._buildCorridor(self.sceneManager,"corridor_wall",self.nodes[0],self.nodes[1],0,0,1,self.width[VR_to_create],self.wall_height[VR_to_create],self.wallLeft[VR_to_create],self.wallRight[VR_to_create],self.wallEnd[VR_to_create],self.num_WallTexture[VR_to_create])
        name1 = "%s_1" %(name)
        name2 = "%s_2" %(name)
        name3 = "%s_3" %(name)
        name4 = "%s_4" %(name)
        
        #Definiere Start-und Endtypen
        if end_start==0:
            s1 = -width/2
            s2 = -width/2
        elif end_start==1:
            s1 = width/2
            s2 = -width/2
        elif end_start==2:
            s1 = -width/2
            s2 = width/2
        elif end_start==3:
            s1 = width/2
            s2 = width/2
        elif end_start==4:
            s1 = 0
            s2 = 0
            
        if end_stop==0:
            e1 = width/2
            e2 = width/2
        elif end_stop==1:
            e1 = -width/2
            e2 = width/2
        elif end_stop==2:
            e1 = width/2
            e2 = -width/2
        elif end_stop==3:
            e1 = -width/2
            e2 = -width/2
        elif end_stop==4:
            e1 = 0
            e2 = 0
        
#        print("corridor numbers:", start, stop, width)
        if start[0]==stop[0]: # x coordinates are the same, track is in z direction
    # def _buildWall(self, sceneManager, name, richtung, start, stop, wall_height, texture, num_texture): #richtung: direction
            self._buildWall(sceneManager,name1,1,(start[0]-width/2,start[1]+s1),(stop[0]-width/2,stop[1]+e1),wall_height,texture_left,num_texture);
            self._buildWall(sceneManager,name2,2,(start[0]+width/2,start[1]+s2),(stop[0]+width/2,stop[1]+e2),wall_height,texture_right,num_texture);
            if front==1:
                #self._buildWall(sceneManager,name3,3,(start[0]-width/2,start[1]-width/2),(start[0]+width/2,start[1]-width/2),wall_height,texture_end,1);
                self._buildWall(sceneManager,name4,4,(stop[0]-width/2,stop[1]+width/2),(stop[0]+width/2,stop[1]+width/2),wall_height,texture_end,1);

        elif start[1]==stop[1]: # z coordinates are the same, track is in x direction
            self._buildWall(sceneManager,name1,3,(start[0]+s1,start[1]-width/2),(stop[0]+e1,stop[1]-width/2),wall_height,texture_left,num_texture);
            self._buildWall(sceneManager,name2,4,(start[0]+s2,start[1]+width/2),(stop[0]+e2,stop[1]+width/2),wall_height,texture_right,num_texture);

        return
            
             
    def _createVR(self,VR_to_create):
        if self.numMonitors>0:
            self._buildCorridor(self.sceneManager,"corridor_wall",self.nodes[VR_to_create][0],self.nodes[VR_to_create][1],0,0,1,self.width[VR_to_create],self.wall_height[VR_to_create],self.wallLeft[VR_to_create],self.wallRight[VR_to_create],self.wallEnd[VR_to_create],self.num_WallTexture[VR_to_create])
            self._buildCeil(self.sceneManager, self.width[VR_to_create], self.wall_height[VR_to_create],self.ceilPattern[VR_to_create], self.num_CeilTexture[VR_to_create])
            self._buildFloor(self.sceneManager, self.width[VR_to_create], self.floorPattern[VR_to_create], self.num_FloorTexture[VR_to_create])
            self.numR_delivered = [0,0,0,0]                                          # reset reward states
            self.numR_consumed = [0,0,0,0]                                          # reset reward states
            self.lick_detected = [0,0,0,0] 
            self.VRCreated = 1
        return


    def _destroyVR(self):
        if self.numMonitors>0:
            self.sceneManager.destroyEntity("pl_floor_%d" % self.currentVR)
            self.sceneManager.destroyEntity("pl_ceil_%d" % self.currentVR)
            self.sceneManager.destroyEntity("pl_corridor_wall_1_%d" % self.currentVR)
            self.sceneManager.destroyEntity("pl_corridor_wall_2_%d" % self.currentVR)
            self.sceneManager.destroyEntity("pl_corridor_wall_3_%d" % self.currentVR)
            self.sceneManager.destroyEntity("pl_corridor_wall_4_%d" % self.currentVR)
        return


class Application (sf.Application):
# the parent functions for the Application object are in the sf_OIS.py file in the packages_2.7/ogre/renderer/OGRE/        
    def __init__(self, current_stage, task):
        self.frameListener = None
        self.root = None
        self.camera = None
        self.renderWindow = None
        self.sceneManager = None
        self.world = None
        self.unittest = isUnitTest()
        self.current_stage = current_stage
        self.task = task

    def _createScene (self): # the orientation of the cameras - createCamera should run before createScene
#        Tkinter.Tk().withdraw()
#        global file_path

        # wb = load_workbook(file_path)
        # ws = wb.get_active_sheet()
            
        self.startingPos = 0 #ws.cell('C26').value
   
        sceneManager = self.sceneManager
          
        self.sceneManager.setAmbientLight((0.5, 0.5, 0.5))
 
        self.camNode = sceneManager.getRootSceneNode().createChildSceneNode('camNode', (0, 10, self.startingPos))
        self.camNode.pitch(ogre.Degree(0))
        
        self.camYawNode = self.camNode.createChildSceneNode('camYawNode')
        self.camYawNode.yaw(ogre.Degree(180))
        self.camYawNode.attachObject(self.camera)

        if self.numMonitors==3:
            self.camYawNode.attachObject(self.cameraL)
            self.cameraL.yaw(ogre.Degree(90))
            self.camYawNode.attachObject(self.cameraR)
            self.cameraR.yaw(ogre.Degree(-90))
        elif self.numMonitors==5:
            self.camYawNode.attachObject(self.cameraL)
            self.cameraL.yaw(ogre.Degree(45))
            self.camYawNode.attachObject(self.cameraR)
            self.cameraR.yaw(ogre.Degree(-45))
            self.camYawNode.attachObject(self.cameraLL)
            self.cameraLL.yaw(ogre.Degree(90))
            self.camYawNode.attachObject(self.cameraRR)
            self.cameraRR.yaw(ogre.Degree(-90)) 
                   

    def _chooseSceneManager (self):
        self.sceneManager = self.root.createSceneManager (ogre.ST_EXTERIOR_CLOSE, 'TerrainSM')
        
 
    def _createCamera(self): # creating the cameras
        
#        Tkinter.Tk().withdraw()
#        global file_path
#        self.openOpt = options = {}
#        options['initialdir'] = wDir+'\..\..\..\Configurations'
#        options['title'] = 'Select experiment parameters'
#        
        #file_path = tkFileDialog.askopenfilename(**self.openOpt) # asking which experiment to load?
        # file_path = u'C:\LuigsNeumann_Treadmill\Configurations\\20181001ReducedMazeSet.xlsx'
        # wb = load_workbook(file_path)
        # ws = wb.get_active_sheet()
        
        self.numMonitors = 3 #int(ws.cell('C46').value)
        self.resolution = [1024, 768] #map(int,re.findall(r'\d+',ws.cell('C47').value)) # 1024 x 768


		#This value represents the VERTICAL field-of-view. 
		# The horizontal field of view is calculated from this depending on the dimensions of the viewport 
		# (they will only be the same if the viewport is square).
        if self.numMonitors==5: 
            FOV = 34
        else:
            FOV = 73.74
            
        self.camera = self.sceneManager.createCamera('PlayerCam')
        self.camera.nearClipDistance = 1
        self.camera.setFOVy(ogre.Degree(FOV))
        camH = 4.9
#        self.camera.pitch(ogre.Degree(22)) # to pitch the camera towards the sky
        self.camera.setPosition(ogre.Vector3(0, camH, 0))
        
        if self.numMonitors>=3:
            self.cameraL = self.sceneManager.createCamera('CamL')
            self.cameraL.nearClipDistance = 1
            self.cameraL.setFOVy(ogre.Degree(FOV)) 
            self.cameraR = self.sceneManager.createCamera('CamR')
            self.cameraR.nearClipDistance = 1
            self.cameraR.setFOVy(ogre.Degree(FOV))
            self.cameraL.setPosition(ogre.Vector3(0, camH, 0))
            self.cameraR.setPosition(ogre.Vector3(0, camH, 0))



        if self.numMonitors==5:
            self.cameraLL = self.sceneManager.createCamera('CamLL')
            self.cameraLL.nearClipDistance = 1
            self.cameraLL.setFOVy(ogre.Degree(FOV)) 
            self.cameraRR = self.sceneManager.createCamera('CamRR')
            self.cameraRR.nearClipDistance = 1
            self.cameraRR.setFOVy(ogre.Degree(FOV))
                                 
        self.renderWindow.resize(max(1,self.numMonitors)*self.resolution[0],self.resolution[1]+30)        # window size and position                        
#        self.renderWindow.reposition(-(max(1,self.numMonitors)-1)/2*self.resolution[0],-30)
#        self.renderWindow.reposition(-3*self.resolution[0],-30)
        self.renderWindow.reposition(-3080,-30)

        
    def _createViewports(self):
        if self.numMonitors==5:
            viewport1 = self.renderWindow.addViewport(self.camera, ZOrder = 0, left = 0.4, top = 0, width = 0.2, height = 1)
            viewport2 = self.renderWindow.addViewport(self.cameraL, ZOrder = 1, left = 0.2, top = 0, width = 0.2, height = 1)
            viewport3 = self.renderWindow.addViewport(self.cameraR, ZOrder = 2, left = 0.6, top = 0, width = 0.2, height = 1)
            viewport4 = self.renderWindow.addViewport(self.cameraLL, ZOrder = 3, left = 0, top = 0, width = 0.2, height = 1)
            viewport5 = self.renderWindow.addViewport(self.cameraRR, ZOrder = 4, left = 0.8, top = 0, width = 0.2, height = 1)
        elif self.numMonitors==3:
            viewport1 = self.renderWindow.addViewport(self.camera, ZOrder = 0, left = 0.3333, top = 0, width = 0.3334, height = 1)
            viewport2 = self.renderWindow.addViewport(self.cameraL, ZOrder = 1, left = 0, top = 0, width = 0.3333, height = 1)
            viewport3 = self.renderWindow.addViewport(self.cameraR, ZOrder = 2, left = 0.6667, top = 0, width = 0.3333, height = 1)
        else:
            viewport1 = self.renderWindow.addViewport(self.camera, ZOrder = 0, left = 0, top = 0, width = 1, height = 1)

        
    def _createFrameListener(self):
        self.frameListener = FrameListener(self.renderWindow, self.camera, self.sceneManager, self.current_stage, self.task)
#        self.frameListener = FrameListener(self.renderWindow, self.camera, self.sceneManager, 1)
        self.root.addFrameListener(self.frameListener)
        self.frameListener.showDebugOverlay(False)
        
 
if __name__ == '__main__':
	# the parent functions for the Application object are in the sf_OIS.py file in the packages_2.7/ogre/renderer/OGRE/
    current_stage = int(sys.argv[1])
    task = sys.argv[2]
    experimenter = sys.argv[3]
    ta = Application (current_stage, task)
#    ta = Application()
    ta.go () 
    # line 103 of sf_OIS.py
