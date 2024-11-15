from ImageAnal import *

datapath = os.getcwd() + '/'  # current working directory - look for data and strings
date_time = '2021-11-02_17-02-32'  # date and time of the imaging session
name = 'KS028'  # mouse name
task = 'NearFarLong'  # task name

# locate the suite2p folder
suite2p_folder = datapath + 'data/' + name + '_imaging/KS028_110221/'

# the name and location of the imaging log file
imaging_logfile_name = suite2p_folder + 'KS028_TSeries-11022021-1017-001.xml'

# the name and location of the trigger voltage file
TRIGGER_VOLTAGE_FILENAME = suite2p_folder + 'KS028_TSeries-11022021-1017-001_Cycle00001_VoltageRecording_001.csv'

D1 = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME)
print(D1.n_laps)
