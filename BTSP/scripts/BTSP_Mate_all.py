import numpy as np
import pandas as pd
from BtspAnalysis import BtspAnalysis


activity_tensor = np.load("data_Mate_full/activity_tensor.npy")
pf_df = pd.read_pickle("data_Mate_full/pf_df.pickle")

run = 0
roi_int_idx = 3
roi_df = pf_df.loc[(pf_df["roi_int_idx"] == roi_int_idx) & (pf_df["run"] == run)]
place_field_bounds = roi_df[["pf_start", "pf_end"]].values
rate_matrix = activity_tensor[:,roi_int_idx,:]

sessionID = "test_session"
analysis = BtspAnalysis(sessionID=sessionID, cellid=roi_int_idx, rate_matrix=rate_matrix, place_field_bounds=place_field_bounds)
analysis.find_btsp()
place_fields_df = analysis.combine_place_field_dataframes()
pass