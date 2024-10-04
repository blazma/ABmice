import numpy as np
import pandas as pd

depths_folder = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\depths"
data_folder = r"D:\CA3"

meta_df = pd.read_excel(f"{data_folder}/CA3_meta.xlsx")
for i_sess, sess in meta_df.iterrows():
    try:
        iscell = np.load(f"{data_folder}/data/{sess["name"]}_imaging/{sess["session id"]}/iscell.npy", allow_pickle=True)
        depths_df = pd.read_excel(f"{depths_folder}/cellDepth_depths_{sess["session id"]}.xlsx")
    except FileNotFoundError:
        print(f"FileNotFoundError in {sess["session id"]}")
        continue
    pass