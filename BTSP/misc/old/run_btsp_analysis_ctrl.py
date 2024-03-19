from RUN_BTSP_ANALYSIS import BtspAnalysis

output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\special\\"
data_paths_CA1 = {
    "ctrl": r"D:\special\CA1_ctrl\\",
}
selected_sessions_CA1 = [
    "KS030_103121",
    "KS030_110621",
]
for run_type, data_path in data_paths_CA1.items():
    analysis = BtspAnalysis("CA1", data_path, output_path=output_path, extra_info=run_type, is_masks=True,
                            is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False,
                            selected_sessions = selected_sessions_CA1)
    analysis.run_btsp_analysis()


data_paths_CA3 = {
    "ctrl": r"D:\special\CA3_ctrl\\",
}
selected_sessions_CA3 = [
    "srb270_230118",
    "srb270_230120",
]
for run_type, data_path in data_paths_CA3.items():
    analysis = BtspAnalysis("CA3", data_path, output_path=output_path, extra_info=run_type, is_masks=True,
                            is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False,
                            selected_sessions = selected_sessions_CA3)
    analysis.run_btsp_analysis()
