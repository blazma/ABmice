import numpy as np
import warnings
from packaging import version
from sklearn.metrics import mean_squared_error
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter


def preprocess(F, win_baseline, sig_baseline, fs):
## https://suite2p.readthedocs.io/en/latest/api/suite2p.extraction.html

    win = int(win_baseline*fs)
    Flow = gaussian_filter(F,    [0., sig_baseline])
    Flow = minimum_filter1d(Flow,    win)
    Flow = maximum_filter1d(Flow,    win)
    F = F - Flow
    return F

def spks_cleaner(fc, spks, event_snr_threshold=3.5, peri_median_width=0.5, pre_range=30, post_range='calulate'):
    # set up default post_range if  not providided
    if post_range == 'calulate':
        post_range = int(pre_range / 3)

    # peri median SD - a bit nonsense
    # build a mask including values around the roi median fluorescence
    roi_medians = np.median(fc, axis=1)
    peri_median_masks = (fc > (roi_medians[:, None] * (1 - peri_median_width))) & (
                fc < (roi_medians[:, None] * (1 + peri_median_width)))
    with warnings.catch_warnings():  # we suppres the warnings when the median is negative and correct the spks output at the last step
        warnings.simplefilter("ignore")
        if version.parse("1.20.0") > version.parse(np.__version__):
            # for numpy < 1.20.0
            fc_peri_median = fc.copy()
            fc_peri_median[~peri_median_masks] = np.nan
            peri_median_std = np.nanstd(fc_peri_median, axis=1)
        else:
            #             calculate SD  on values taken from the median's vicinity for each roi
            #             the following solution is for numpy >= 1.20.0 for older numpy we can't use where
            peri_median_std = np.std(fc, axis=1, where=peri_median_masks)

    #     print('peri_median_std', peri_median_std)
    # replace nans with 0 in peri_median_std
    np.nan_to_num(peri_median_std, copy=False)

    # pad spks and fc in the time axis
    spks = np.pad(spks, [(0, 0), (pre_range, post_range)])
    fc = np.pad(fc, [(0, 0), (pre_range, post_range)], mode='edge')
    # generte non-zero spks(n0_spks) indexes
    n0_spks_roi_idxs, n0_spks_time_idxs = np.where(spks > 0)
    # generate range object for time period around non-zero spks
    n0_spks_time_range_indxs = [range(spks_time_idx - pre_range, spks_time_idx + post_range + 1) for spks_time_idx in
                                n0_spks_time_idxs]
    # fancy index the region around  non-zero spks values from fc
    range_of_interest = fc[n0_spks_roi_idxs[:, None], n0_spks_time_range_indxs]
    # fit line on isolated time regions of fc
    fit_region_x_end_indx = int(pre_range * 2 / 3)
    x_vals = np.arange(fit_region_x_end_indx)
    y_vals = range_of_interest[:, :fit_region_x_end_indx].T
    fit = np.polynomial.polynomial.polyfit(x_vals, y_vals, 1)
    # generate predictions from  fits
    pred_y_vals = np.polynomial.polynomial.polyval(x_vals, fit).T
    # calculate rmse from prediction
    rmse = mean_squared_error(y_vals, pred_y_vals, squared=False, multioutput='raw_values')
    # find the local maximum  of fc value following the non-zero spks, and calculate a mean maximum including the surroundind neighburs (called:smoothed maximums)
    local_maximum_idxs = np.argmax(range_of_interest[:, pre_range:-1], axis=1) + pre_range
    local_maximum_range_idxs = [range(local_maximum_idx - 1, local_maximum_idx + 2) for local_maximum_idx in
                                local_maximum_idxs]
    range_of_interest_row_indices = np.arange(len(local_maximum_range_idxs))[:, None]
    smoothed_maximums = range_of_interest[range_of_interest_row_indices, local_maximum_range_idxs].mean(axis=1)
    # calculate the distance between smoothed local maximums and last predicted values as a multiple of rmse
    with np.errstate(
            divide='ignore'):  # padding fc with 'edge' can result in 0 rmse is the first spks value is  non zero
        snr_local_base = (smoothed_maximums - pred_y_vals[-1]) / rmse
        # calculate the distance between smoothed local maximums and median as a multiple of SD of the prei median range
        snr_roi_base = (smoothed_maximums - roi_medians[n0_spks_roi_idxs]) / peri_median_std[
            n0_spks_roi_idxs]  # nan perimedian std is changed to 0 which can cause true divide error warning
    # select n0_spks where local or roi based snr is lower or equal  to treshold they should be eleiminated
    bad_n0_spks_bidx = (snr_local_base <= event_snr_threshold) & (snr_roi_base <= event_snr_threshold)
    # Use Boolean index for filtering  unvantented n0_spks indexes (roi,time) and use the filtered indexes for fancy indexing of spks to overwrite its values to 0
    spks[n0_spks_roi_idxs[bad_n0_spks_bidx], n0_spks_time_idxs[bad_n0_spks_bidx]] = 0
    # slice out original spks from padded spks
    spks = spks[:, pre_range:-post_range]
    # set cleaned spks output to 0 where mediaan is negative
    spks[(roi_medians < 0), :] = 0
    return spks


################## selected sessions from CA1 and CA3

#base_folders = [
#    r"D:\special\CA1_gaspar\KS030_imaging\KS030_103121",
#    r"D:\special\CA1_gaspar\KS030_imaging\KS030_110621",
#    r"D:\special\CA1_tau0.8_gaspar\KS030_imaging\KS030_103121",
#    r"D:\special\CA1_tau0.8_gaspar\KS030_imaging\KS030_110621",
#    r"D:\special\CA3_gaspar\srb270_imaging\srb270_230118",
#    r"D:\special\CA3_gaspar\srb270_imaging\srb270_230120",
#    r"D:\special\CA3_tau0.8_gaspar\srb270_imaging\srb270_230118",
#    r"D:\special\CA3_tau0.8_gaspar\srb270_imaging\srb270_230120",
#]


################## CA3 all

base_folder_all_CA3 = "D:\\special\\all_CA3_tau0.8_gaspar\\data"
base_folders_all_CA3 = [
    f"{base_folder_all_CA3}\\srb231_imaging\\srb231_220804\\",
    f"{base_folder_all_CA3}\\srb231_imaging\\srb231_220808\\",
    f"{base_folder_all_CA3}\\srb231_imaging\\srb231_220809_002\\",
    f"{base_folder_all_CA3}\\srb231_imaging\\srb231_220809_004\\",
    f"{base_folder_all_CA3}\\srb231_imaging\\srb231_220812_001\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221027_T1\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221027_T2\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221028_T1\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221028_T2\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221122\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221125\\",
    f"{base_folder_all_CA3}\\srb251_imaging\\srb251_221205\\",
    f"{base_folder_all_CA3}\\srb269_imaging\\srb269_230119\\",
    f"{base_folder_all_CA3}\\srb269_imaging\\srb269_230120\\",
    f"{base_folder_all_CA3}\\srb269_imaging\\srb269_230125\\",
    f"{base_folder_all_CA3}\\srb270_imaging\\srb270_230117\\",
    f"{base_folder_all_CA3}\\srb270_imaging\\srb270_230118\\",
    f"{base_folder_all_CA3}\\srb270_imaging\\srb270_230119\\",
    f"{base_folder_all_CA3}\\srb270_imaging\\srb270_230120\\",
    f"{base_folder_all_CA3}\\srb270_imaging\\srb270_230124\\",
    f"{base_folder_all_CA3}\\srb270_imaging\\srb270_230127\\",
]

base_folders_newAnimals_CA3 = [
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231207\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231208a\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231211\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231212\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231212_a\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231213\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231213a\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231214\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231215\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231215afternoon\\",
    #f"{base_folder_all_CA3}\\srb363_imaging\\srb363_231218\\",
    #f"{base_folder_all_CA3}\\srb377_imaging\\srb377_240115\\",
    f"{base_folder_all_CA3}\\srb377_imaging\\srb377_240116\\",
    f"{base_folder_all_CA3}\\srb377_imaging\\srb377_240118\\",
    f"{base_folder_all_CA3}\\srb377_imaging\\srb377_240124\\"
]

################## CA1 all (random only)

base_folder_all_CA1 = "D:\\special\\all_CA1_tau0.8_gaspar\\data"
base_folders_all_CA1 = [
    f"{base_folder_all_CA1}\\KS028_imaging\\KS028_103121",
    f"{base_folder_all_CA1}\\KS028_imaging\\KS028_110121",
    f"{base_folder_all_CA1}\\KS028_imaging\\KS028_110221",
    f"{base_folder_all_CA1}\\KS028_imaging\\KS028_110621",
    f"{base_folder_all_CA1}\\KS028_imaging\\KS028_110721",
    f"{base_folder_all_CA1}\\KS029_imaging\\KS029_110421",
    f"{base_folder_all_CA1}\\KS029_imaging\\KS029_110621",
    f"{base_folder_all_CA1}\\KS029_imaging\\KS029_110921",
    f"{base_folder_all_CA1}\\KS029_imaging\\KS029_111321",
    f"{base_folder_all_CA1}\\KS029_imaging\\KS029_111421",
    f"{base_folder_all_CA1}\\KS029_imaging\\KS029_111521",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_102821",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_102921",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_103021",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_103121",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_110121",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_110521",
    f"{base_folder_all_CA1}\\KS030_imaging\\KS030_110621",
    f"{base_folder_all_CA1}\\srb131_imaging\\srb131_211015",
    f"{base_folder_all_CA1}\\srb131_imaging\\srb131_211016",
    f"{base_folder_all_CA1}\\srb131_imaging\\srb131_211017",
    f"{base_folder_all_CA1}\\srb131_imaging\\srb131_211021",
]

base_folders_forShuffle = [
    rf"C:\Users\martin\home\phd\btsp_project\analyses\other\for_reshuffle\srb270_imaging\230118"
]
for base_folder in base_folders_forShuffle:
    print(base_folder)
    spks_path = f"{base_folder}/spks_oasis_08s.npy"
    fluo_path = f"{base_folder}/F.npy"
    fpil_path = f"{base_folder}/Fneu.npy"
    #spks_gaspar_path = f"{base_folder}/spks_gaspar.npy"

    spks = np.load(spks_path)
    fluo = np.load(fluo_path)
    fpil = np.load(fpil_path)

    F = fluo-(0.7*fpil)  # subtract neuropil
    #Fc = preprocess(F=F, win_baseline=60, sig_baseline=10, fs=30)  # subtract baseline

    pmw = 0.4
    spks_gaspar_path = f"{base_folder}/spks.npy"
    spks_gaspar = spks_cleaner(F, spks, peri_median_width=pmw)
    with open(spks_gaspar_path, "wb") as spks_gaspar_file:
        np.save(spks_gaspar_file, spks_gaspar)
