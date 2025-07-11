import os
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas

area = "CA3"
base_folder = rf"D:\{area}\data"

animals = {
    "CA1": [#"KS028",
            #"KS029",
            #"KS030",
            #"srb128"],
            #"srb131",
            #"srb231",
            #"srb251",
            #"srb402"],
            "srb498"], #old animal - different proj.
            #"srb504",
            #"srb504a",
            #"srb517"],
    "CA3": [#"srb231",
            #"srb251",
            #"srb269",
            #"srb270",
            #"srb363",
            #"srb377",
            #"srb402"
            "srb529"]
}

meta_content = []
for animal in animals[area]:
    imaging_folder = f"{base_folder}/{animal}_imaging"
    _, subfolders, _ = next(os.walk(imaging_folder))
    sessions = [folder for folder in subfolders if animal in folder]
    _, behavior_folders, _ = next(os.walk(f"{base_folder}/{animal}_NearFarLong"))
    behav_dates = np.array([datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S") for folder in behavior_folders if "20" in folder])

    for session in sessions:
        print(session)
        _, _, files = next(os.walk(f"{imaging_folder}/{session}"))
        imaging_logfile_name = [f for f in files if "xml" in f and "VoltageRecording" not in f][0]
        trigger_voltage_filename = [f for f in files if "csv" in f][0]

        tree = ET.parse(f"{imaging_folder}/{session}/{imaging_logfile_name}")
        xml_date_str = next(tree.getroot().iter("PVScan")).get("date")
        xml_date = datetime.strptime(xml_date_str, "%m/%d/%Y %I:%M:%S %p")
        #date_time = behav_dates[np.where(xml_date > behav_dates)[0][-1]]  # melyik az utolsó olyan behav folder dátum aminél az XML-es még nagyobb
        date_time = behav_dates[np.argmin(abs(xml_date - behav_dates))]  # legkisebb idokulonbsegu datum

        meta_row = {
            "session id": session,
            "date_time": datetime.strftime(date_time, "%Y-%m-%d_%H-%M-%S"),
            "name": animal,
            "task": "NearFarLong",
            "suite2p_folder": session,
            "imaging_logfile_name": imaging_logfile_name,
            "TRIGGER_VOLTAGE_FILENAME": trigger_voltage_filename
        }
        meta_content.append(meta_row)

meta_df = pandas.DataFrame.from_dict(meta_content)
meta_df.to_excel(f"{base_folder}/{area}_meta_srb529.xlsx", index=False)

