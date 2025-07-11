import numpy as np
import warnings
from packaging import version
from sklearn.metrics import mean_squared_error
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import pandas as pd


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
    n0_spks_time_range_indxs = [range(spks_time_idx - pre_range, spks_time_idx + post_range + 1) for spks_time_idx in n0_spks_time_idxs]
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
    local_maximum_range_idxs = [range(local_maximum_idx - 1, local_maximum_idx + 2) for local_maximum_idx in local_maximum_idxs]
    range_of_interest_row_indices = np.arange(len(local_maximum_range_idxs))[:, None]
    smoothed_maximums = range_of_interest[range_of_interest_row_indices, local_maximum_range_idxs].mean(axis=1)
    # calculate the distance between smoothed local maximums and last predicted values as a multiple of rmse
    with np.errstate(divide='ignore'):  # padding fc with 'edge' can result in 0 rmse is the first spks value is  non zero
        snr_local_base = (smoothed_maximums - pred_y_vals[-1]) / rmse
        # calculate the distance between smoothed local maximums and median as a multiple of SD of the prei median range
        snr_roi_base = (smoothed_maximums - roi_medians[n0_spks_roi_idxs]) / peri_median_std[n0_spks_roi_idxs]  # nan perimedian std is changed to 0 which can cause true divide error warning
    # select n0_spks where local or roi based snr is lower or equal  to treshold they should be eleiminated
    bad_n0_spks_bidx = (snr_local_base <= event_snr_threshold) & (snr_roi_base <= event_snr_threshold)
    # Use Boolean index for filtering  unvantented n0_spks indexes (roi,time) and use the filtered indexes for fancy indexing of spks to overwrite its values to 0
    spks[n0_spks_roi_idxs[bad_n0_spks_bidx], n0_spks_time_idxs[bad_n0_spks_bidx]] = 0
    # slice out original spks from padded spks
    spks = spks[:, pre_range:-post_range]
    # set cleaned spks output to 0 where mediaan is negative
    spks[(roi_medians < 0), :] = 0
    return spks


area = "CA3"
base_folder = rf"D:\{area}"
meta_df = pd.read_excel(f"{base_folder}/{area}_meta.xlsx")

for i_row, row in meta_df.iterrows():
    print(row["session id"])
    session_folder = f"{base_folder}/data/{row["name"]}_imaging/{row["suite2p_folder"]}"
    spks = np.load(f"{session_folder}/spks_oasis_08s.npy")
    fluo = np.load(f"{session_folder}/F.npy")
    fpil = np.load(f"{session_folder}/Fneu.npy")

    F = fluo-(0.7*fpil)  # subtract neuropil
    #Fc = preprocess(F=F, win_baseline=60, sig_baseline=10, fs=30)  # subtract baseline

    pmw = 0.4
    spks_gaspar_path = f"{session_folder}/spks.npy"
    spks_gaspar = spks_cleaner(F, spks, peri_median_width=pmw)
    with open(spks_gaspar_path, "wb") as spks_gaspar_file:
        np.save(spks_gaspar_file, spks_gaspar)
