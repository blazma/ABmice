from LogAnal import *
import openpyxl
import pandas as pd
import numpy as np
import tqdm


data_root = "D:\\data\\"
animals = [
    "srb201",
    "srb210",
    "srb267",
    "srb287",
    "srb297",
    "srb299",
    "srb318",
    "srb352",
    "srb360",
    "srb396",
    "srb399",
    "srb434",
    "srb449",
    "srb459",
]

sessions_list = []
for animal in animals:
    wb = openpyxl.load_workbook(filename=f"{data_root}/filler_input_excel_{animal}.xlsx")
    ws = wb.worksheets[0]
    for row in list(ws.rows)[1:]:
        animal_name = row[2].value
        session_name = row[4].value
        if session_name == None:
            continue
        session = {
            "date_time": row[1].value,
            "name": animal_name,
            "task": row[3].value,
            "suite2p_folder": session_name,
            "imaging_logfile_name": row[5].value,
            "TRIGGER_VOLTAGE_FILENAME": row[6].value,
        }
        sessions_list.append(session)
sessions_df = pd.DataFrame.from_dict(sessions_list)

# bs = behavior score
bs_dicts = []
for i_s, s in tqdm.tqdm(sessions_df.iterrows()):
    date_time, name, task, sessionID = s["date_time"], s["name"], s["task"], s["suite2p_folder"]
    sess = Session("D:\\", date_time, name, task, sessionID)
    sess.calc_behavior_score(corrA=14, corrB=15)

    bs_dict = {"sessionID": sessionID}
    bs_dict = bs_dict | sess.behavior_score_components  # merges the two dicts
    bs_dict["BEHAVIOR SCORE"] = sess.behavior_score

    # new corridor if exists
    corridor_types = np.unique(sess.i_corridors)
    if 17 in corridor_types:
        sess.calc_behavior_score(corrA=17)
        bs_dict["Lick index (17)"] = sess.behavior_score_components["Lick index (17)"]
        bs_dict["Speed index (17)"] = sess.behavior_score_components["Speed index (17)"]

    bs_dicts.append(bs_dict)

bs_df = pd.DataFrame.from_dict(bs_dicts)
bs_df.to_excel(f"{data_root}/behavior_scores.xlsx")
