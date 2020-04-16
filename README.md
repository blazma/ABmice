### General information

Library for analysing the behavior of Mice in a virtual reality experiments

author: Balazs B Ujfalussy, 2020 `balazs.ujfalussy@gmail.com`

Also includes a script generating the VR in Ogre, which was originally written by Luigs-Neumann (http://www.luigs-neumann.com/) but later modified by BBU. 

Run the Mouse_View.py in the rigth folder to analyse data for a selected mouse. 

### Requirements

To run the code, you need python 2.7 or python 3.7 installed with the usual sciantific packages (numpy, scipy, matplotlib, tkinter). 

- PyOgre runs on python 2.7 thus the original data is saved by machines running only python 2.7. 
- The analysis of the imaging data requires python 3.7.
- The data can be copyed to other machines running python 3.7 for analysis, but DO NOT COPY any data back into the machines doing the recordings!

Folder structure: 
	* All python files should be placed together in the same folder. 
	* The behavioral data should be placed in the data subfolder. 
	* Within the data subfolder the location of the 
		- python datafiles is: data/NAME/NAME_TASK_NAME.pkl
		- Labview logfiles is: data/NAME_TASK_NAME/YYYY_MM_DD_HH_MM_SS/
		- Labview behavioral data is: data/NAME_TASK_NAME/behavioral_data/NAME_TASK_NAME_sessionX.txt/

	CAPITAL LETTERS denote strings to be specified by the experiment.



### Details

Stages.py: A framework for defining the behavioral stages in the VR behavioral experiments. 
	Two new object classes are defined: 

		* Stage: stores the properties of a single stage
		* Stage_collection: contains the Stages that belong to a single experiment.

	The stage collection is saved in a python pickle file with the following name: TASK_NAME_stages.pkl It is used by many of the scripts. 


Corridors.py: A framework for defining the corridor properties in the VR behavioral experiments. 
	Two new object classes are defined: 

		* Corridor: stores the properties of a single corridor
		* Corridor_list: contains the Corridors that belong to a single experiment.

	The Corridor_list is saved in a python pickle file with the following name: TASK_NAME_corridors.pkl It is used by many of the scripts. 


Mice.py: A framework for managing behavioral data in 'in vivo' experiments
	Three new object classes are defined:

		* Mouse: storing all the high level behavioral data for the mice, and its plotting methods.
		* Session_data: behavioral data of a single session
		* Read_Mouse: wrapper for reading the data of an existing mouse or creating a new one

	The data is saved in a single pickle file: data/NAME/NAME_TASK_NAME.pkl

Maze_Labview2Ogre.py: A script for setting up updating the virtual reality environment based on the mouse position. This is a highly modified version that requires the mouse position in the VR as input. In our implementation all sensors, including the rotary encoder, are read a separate prgram written in LabView. The Labview program also controls the actuators (lick pumps for reward). The logic is the following:
	
	1. The Labview program starts the receives the Mouse_Init.py where the user can select the name of the mouse and the experiment type. The script then reads the stage and corridor definitions and returns the corridor properties to Labview.

	2. The Labview starts Maze_Labview2Ogre.py that renders the VR during the experiment. The behavioral data is logged by Labview.

	3. At the end of the experiment, Labview starts Mouse_Close.py which reads the logfiles, allows plotting and saves the data.

LogAnal.py: Low-level functions for plotting the speed and licking of the mouse in different corridors. 

Mouse_Init.py: initializing mouse parameters for the VR setup
	* expects three string inputs: Mouse name, task and experimenter
	* provides a dialog for plotting the past behavior and selecting the next stage
	* returns the next stage and the corridor properties


Mouse_Close.py: A dialog for updating, plotting and saving the data for a selected mouse.

Mouse_View.py: A dialog for choosing the mouse to be analysed.


### Expected output

 - saving and plotting the data
