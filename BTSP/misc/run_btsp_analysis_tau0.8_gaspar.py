from RUN_BTSP_ANALYSIS import BtspAnalysis

#output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\special\\"
#data_paths_CA1 = {
    #"tau0.8": r"D:\special\CA1_tau0.8\\",
    #"gaspar": r"D:\special\CA1_gaspar\\",
#    "tau0.8_gaspar": r"D:\special\CA1_tau0.8_gaspar\\",
#}
#selected_sessions_CA1 = [
#    "KS030_103121",
#    "KS030_110621",
#]
#for run_type, data_path in data_paths_CA1.items():
#    btsp_statistics = BtspAnalysis("CA1", data_path, output_path=output_path, extra_info=run_type, is_masks=True,
#                            is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False,
#                            selected_sessions = selected_sessions_CA1)




#data_paths_CA3 = {
#    "tau0.8": r"D:\special\CA3_tau0.8\\",
#    "gaspar": r"D:\special\CA3_gaspar\\",
#    "tau0.8_gaspar": r"D:\special\CA3_tau0.8_gaspar\\",
#}
#selected_sessions_CA3 = [
#    "srb270_230118",
#    "srb270_230120",
#]
#for run_type, data_path in data_paths_CA3.items():
#    analysis = BtspAnalysis("CA3", data_path, output_path=output_path, extra_info=run_type, is_masks=True,
#                            is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False,
#                            selected_sessions = selected_sessions_CA3)
#    analysis.run_btsp_analysis()




##################### gaspar for all CA3
#data_path = "D:\\special\\all_CA3_gaspar\\"
#output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\special\\"
#analysis = BtspAnalysis("CA3", data_path, output_path=output_path, extra_info="gaspar_all", is_masks=True,
#                            is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False)
#analysis.run_btsp_analysis()






##################### tau0.8_gaspar for all CA3
#data_path = "D:\\special\\all_CA3_tau0.8_gaspar\\"
#output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\special\\"
#analysis = BtspAnalysis("CA3", data_path, output_path=output_path, extra_info="tau0.8_gaspar_all", is_masks=True,
#                        is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False)
#analysis.run_btsp_analysis()







##################### tau0.8_gaspar for all CA1 (random only)
data_path = "D:\\special\\all_CA1_tau0.8_gaspar\\"
output_path = "C:\\Users\\martin\\home\\phd\\btsp_project\\analyses\\special\\"
analysis = BtspAnalysis("CA1", data_path, output_path=output_path, extra_info="tau0.8_gaspar_all", is_masks=True,
                        is_shift_criterion_on=True, is_create_ratemaps=True, is_poster=False)
analysis.run_btsp_analysis()
