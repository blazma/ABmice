import os
from copy import deepcopy
import openpyxl
from ImageAnal import *


def read_meta(xlsx_file, folder, animal):
    wb = openpyxl.load_workbook(filename=f"{folder}/{xlsx_file}")
    ws = wb.worksheets[0]
    sessions = {}
    for row in ws.rows:
        session_name = row[4].value
        if animal in session_name and session_name not in sessions:
            sessions[session_name] = {
                "date_time": row[1].value,
                "name": row[2].value,
                "task": row[3].value,
                "suite2p_folder": session_name,
                "imaging_logfile_name": row[5].value,
                "TRIGGER_VOLTAGE_FILENAME": row[6].value,
            }
    return sessions

# read session metadata for animal
animal = "KS028"
folder = "shuffle_CA1_KS028"
xlsx_file = "CA1_meta.xlsx"
sessions = read_meta(xlsx_file, folder, animal)

# create data objects for each session
datapath = os.getcwd() + '/'
for session_name in sessions:
    date_time = sessions[session_name]["date_time"]
    name = sessions[session_name]["name"]
    task = sessions[session_name]["task"]
    suite2p_folder = datapath + 'data/' + f"{name}_imaging/" + sessions[session_name]["suite2p_folder"] + "/"
    imaging_logfile_name = suite2p_folder + sessions[session_name]["imaging_logfile_name"]
    TRIGGER_VOLTAGE_FILENAME = suite2p_folder + sessions[session_name]["TRIGGER_VOLTAGE_FILENAME"]
    ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=session_name)
    ISD.calc_shuffle(ISD.active_cells, 1000, 'shift', batchsize=10)
    del ISD
