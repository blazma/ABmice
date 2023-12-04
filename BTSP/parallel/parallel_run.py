import multiprocessing
from utils import makedir_if_needed
from ..BtspAnalysis import BtspAnalysis
from ..BtspStatistics import BtspStatistics

data_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual"
output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\parallel"
output_stats_path = r"C:\Users\martin\home\phd\btsp_project\analyses\parallel\stats"
makedir_if_needed(output_stats_path)

N_PROCESSES = 5


def run_analysis(params_dict):
    FLW = params_dict["FORMATION_LAP_WINDOW"]
    FGW = params_dict["FORMATION_GAIN_WINDOW"]
    FT = params_dict["FORMATION_THRESHOLD"]
    extra_info = f"FLW={FLW}_FGW={FGW}_FT={FT}"
    print(f"running analysis with parameters {extra_info} ...")

    # run BTSP analysis
    export_path = f"{output_stats_path}/params_{extra_info}.ini"
    analysis = BtspAnalysis("CA1", data_path, output_path, params_dict=params_dict, extra_info=extra_info)
    analysis.run_btsp_analysis()
    analysis.export_parameters(export_path=export_path)

    # calculate BTSP-related statistics
    export_path = f"{output_stats_path}/results_{extra_info}.json"
    statistics = BtspStatistics("CA1", output_path, output_path, extra_info=extra_info)
    statistics.load_data()
    statistics.calc_shift_gain_distribution()
    statistics.export_results(export_path=export_path)


if __name__ == "__main__":
    pool = multiprocessing.Pool(N_PROCESSES)
    for FORMATION_LAP_WINDOW in [3, 5, 7]:
        for FORMATION_GAIN_WINDOW in [3, 5, 7]:
            for FORMATION_THRESHOLD in [0.05, 0.1, 0.2]:
                if FORMATION_GAIN_WINDOW > FORMATION_GAIN_WINDOW:
                    continue
                params_dict = {
                    "FORMATION_LAP_WINDOW": FORMATION_LAP_WINDOW,
                    "FORMATION_GAIN_WINDOW": FORMATION_GAIN_WINDOW,
                    "FORMATION_THRESHOLD": FORMATION_THRESHOLD
                }
                pool.apply_async(run_analysis, (params_dict,))
    pool.close()
    pool.join()
