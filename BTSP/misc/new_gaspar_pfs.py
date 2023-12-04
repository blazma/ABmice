import pandas as pd
from constants import ANIMALS
from utils import grow_df

animals = ANIMALS["CA3"]
gaspar_all_path = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\special\\BTSP_analysis_CA3_231114_gaspar_all"
wo_gaspar_all_path = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\BTSP_analysis_CA3_231106"

pfs_df_gaspar = None
pfs_df_wo_gaspar = None
for animal in animals:
    pfs_df_gaspar_animal = pd.read_pickle(f"{gaspar_all_path}\\{animal}_place_fields_df.pickle")
    pfs_df_wo_gaspar_animal = pd.read_pickle(f"{wo_gaspar_all_path}\\{animal}_place_fields_df.pickle")

    pfs_df_gaspar = grow_df(pfs_df_gaspar, pfs_df_gaspar_animal)
    pfs_df_wo_gaspar = grow_df(pfs_df_wo_gaspar, pfs_df_wo_gaspar_animal)

cols = ["session id", "cell id", "corridor", "lower bound", "upper bound", "formation lap", "end lap", "category"]
pass