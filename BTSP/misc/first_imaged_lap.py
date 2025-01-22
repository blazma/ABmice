import openpyxl
from pprint import pprint
from ImageAnal import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data_root = r"D:\CA1"
animals_sessions = {
    "KS028": ["KS028_103121",
              "KS028_110121",
              "KS028_110221", ],
    "KS029": ["KS029_110621",
              "KS029_110721",
              "KS029_110821",
              "KS029_110921", ],
    "KS030": ["KS030_102921",
              "KS030_103021",
              "KS030_103121",
              "KS030_110121", ],
    "srb128": ["srb128_210922",
               "srb128_210923",
               "srb128_210924"],
    "srb131": ["srb131_211015",
               "srb131_211016", ],
    "srb402": ["srb402_240319_CA1",
               "srb402_240320_CA1"],
    "srb410": ["srb410_240528_CA1"]
}


def read_meta(animal):
    wb = openpyxl.load_workbook(filename=f"{data_root}/CA1_meta.xlsx")
    ws = wb.worksheets[0]
    sessions = {}
    for row in ws.rows:
        animal_name = row[2].value
        session_name = row[4].value
        if session_name is None:
            continue
        if animal == animal_name and session_name not in sessions:
            sessions[session_name] = {
                "date_time": row[1].value,
                "name": animal_name,
                "task": row[3].value,
                "suite2p_folder": session_name,
                "imaging_logfile_name": row[5].value,
                "TRIGGER_VOLTAGE_FILENAME": row[6].value,
            }
    return sessions


def load_session(sessions_all, current_session):
    # set up ImagingSessionData parameters
    date_time = sessions_all[current_session]["date_time"]
    name = sessions_all[current_session]["name"]
    task = sessions_all[current_session]["task"]
    datapath = f"{data_root}/"
    suite2p_folder = f"{datapath}/data/{name}_imaging/{sessions_all[current_session]["suite2p_folder"]}/"
    imaging_logfile_name = suite2p_folder + sessions_all[current_session]["imaging_logfile_name"]
    TRIGGER_VOLTAGE_FILENAME = suite2p_folder + sessions_all[current_session]["TRIGGER_VOLTAGE_FILENAME"]
    ISD = ImagingSessionData(datapath, date_time, name, task, suite2p_folder, imaging_logfile_name, TRIGGER_VOLTAGE_FILENAME, sessionID=current_session)
    ISD.sessionID = current_session
    return ISD


first_imaged_laps = {}
for animal, sessions in animals_sessions.items():
    if animal != "srb128":
        continue

    animal_meta = read_meta(animal)
    for session in sessions:
        ISD = load_session(animal_meta, session)
        first_imaged_lap = ISD.i_Laps_ImData[0]
        first_imaged_laps[session] = first_imaged_lap
pprint(first_imaged_laps, width=1)
