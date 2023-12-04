from BtspStatistics import BtspStatistics

extra_info_tags = ["", "FGactiveLaps", "FGactiveLaps_FLfix", "FGactiveLaps_FLfix_noCorr"]
shuffled_laps_tags = ["", "_shuffled_laps"]

area = "CA1"
data_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual"
output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual"

for shuffled_laps_tag in shuffled_laps_tags:
    for extra_info_tag in extra_info_tags:
        if extra_info_tag == "":
            extra_info = f"{shuffled_laps_tag[1:]}"
        else:
            extra_info = f"{extra_info_tag}{shuffled_laps_tag}"
        print(extra_info)
        btsp_statistics = BtspStatistics(area, data_path, output_path, extra_info)
        btsp_statistics.load_data()
        btsp_statistics.calc_place_field_proportions()
        btsp_statistics.calc_shift_gain_distribution()
        btsp_statistics.plot_shift_gain_distribution()
