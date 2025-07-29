import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Statistics_BothAreas import Statistics_BothAreas


def fig2_NF_vs_ES():
    pass


if __name__ == "__main__":
    data_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"
    output_root = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\manual\\"

    extra_info_CA1 = "NFafter5Laps"
    extra_info_CA3 = "NFafter5Laps"
    extra_info = "NFafter5Laps"
    filter_overextended = True

    stats = Statistics_BothAreas(data_root, output_root, extra_info_CA1, extra_info_CA3, extra_info,
                                 filter_overextended=filter_overextended)
    stats.load_data()
    stats.run_tests()

