from RUN_BTSP_STATISTICS import BtspStatistics

output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\special\\"
def run_btsp_stats(area, date, run_type):
    data_path = f"C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\special\\BTSP_analysis_{area}_{date}_{run_type}"
    btsp_statistics = BtspStatistics(area, data_path, output_path=output_path, extra_info=run_type)
    btsp_statistics.load_data()
    btsp_statistics.calc_place_field_proportions()
    btsp_statistics.calc_shift_gain_distribution()
    btsp_statistics.plot_cells()
    btsp_statistics.plot_place_fields()
    btsp_statistics.plot_place_fields_by_session()
    btsp_statistics.plot_place_field_proportions()
    btsp_statistics.plot_place_fields_criteria()
    btsp_statistics.plot_place_field_properties()
    btsp_statistics.plot_place_field_heatmap()
    btsp_statistics.plot_place_fields_criteria_venn_diagram()
    btsp_statistics.plot_shift_gain_distribution()
    btsp_statistics.plot_shift_gain_distribution(newlyf_only=True)
    btsp_statistics.plot_distance_from_RZ_distribution()
    btsp_statistics.plot_no_shift_criterion()

#run_types = ["ctrl", "tau0.8", "gaspar", "tau0.8_gaspar"]
run_types = ["gaspar_all"]
for run_type in run_types:
    print(run_type)
    run_btsp_stats("CA3", "231114", run_type)
