import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from BTSP.Statistics_BothAreas import Statistics_BothAreas

data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

extra_info_CA1 = ""
extra_info_CA3 = ""
extra_info = ""

def preprocess():
    stats_both = Statistics_BothAreas(data_root, output_root,
                                      extra_info_CA1, extra_info_CA3, extra_info,
                                      create_output_folder=False)

