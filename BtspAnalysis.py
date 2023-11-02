import os
import csv
import numpy as np
import pandas as pd

import PlaceField as PF
from PlaceField import PlaceField
from matplotlib import pyplot as plt


class BtspAnalysis:
    def __init__(self, sessionID, cellid, rate_matrix, corridors=None,
                 place_field_bounds=None, bins_p95_geq_afr=None, i_laps=None, shift_criterion=True):
        """
        BTSP analysis for a single cell from a given session

        @param sessionID: name of session
        @param cellid: ID number of cell/ROI
        @param rate_matrix: firing rate matrix of cell/ROI with dimensions (bins x laps)
        @param corridors: list of corridor indices (defaults to a single corridor with index 0)
        @param place_field_bounds: list of tuples of the format (lower bound, upper bound), finds bounds if not provided
        @param bins_p95_geq_afr: list of bins where 95th percentile of shuffled rates is greater or equal to average firing rate
        @param i_laps: lap indices for a given corridor (defaults to an index list of all laps in rate matrix)
        """

        self.sessionID = sessionID
        if self.sessionID is None:
            self.animalID = None
        elif "_" not in self.sessionID:
            print(f"could not determine animal name from session {sessionID}")
            self.animalID = None
        else:
            self.animalID, _, _ = self.sessionID.partition("_")
        self.cellid = cellid
        self.rate_matrix = rate_matrix

        if corridors is None:
            self.corridors = np.zeros(1)
        else:
            self.corridors = corridors

        if place_field_bounds is None:
            assert bins_p95_geq_afr is not None, "shuffle 95th percentile values must be provided" \
                                                 "if place field bounds are not explicitly provided"
            self.place_field_bounds = self.find_place_fields(bins_p95_geq_afr)
        else:
            self.place_field_bounds = place_field_bounds

        if i_laps is None:
            self.i_laps = np.arange(self.rate_matrix.shape[1])  # because rate_matrix is of shape (bins x laps)
        else:
            self.i_laps = i_laps

        self.unreliable_place_fields = []
        self.early_place_fields = []
        self.transient_place_fields = []
        self.candidate_btsp_place_fields = []
        self.btsp_place_fields = []
        self.nonbtsp_novel_place_fields = []

        self.nmf_matrix = None
        self.shift_criterion = shift_criterion

    def __add__(self, other):
        self.unreliable_place_fields += other.unreliable_place_fields
        self.early_place_fields += other.early_place_fields
        self.transient_place_fields += other.transient_place_fields
        self.candidate_btsp_place_fields += other.candidate_btsp_place_fields
        self.btsp_place_fields += other.btsp_place_fields
        self.nonbtsp_novel_place_fields += other.nonbtsp_novel_place_fields
        return self

    def find_place_fields(self, bins_p95_geq_afr):
        candidate_pfs = []
        idx = 0
        while idx < len(bins_p95_geq_afr) - 1:
            consecutive = True
            candidate_pf = []
            while consecutive:
                if bins_p95_geq_afr[idx + 1] - bins_p95_geq_afr[idx] == 1:
                    consecutive = True
                    candidate_pf.append(bins_p95_geq_afr[idx])
                    if idx + 1 == len(bins_p95_geq_afr) - 1:  # we reached the end of all the bins
                        candidate_pf.append(bins_p95_geq_afr[idx + 1])
                        consecutive = False
                        idx += 1  # to turn off the outer loop
                    else:
                        idx += 1
                else:
                    consecutive = False
                    idx += 1
            if candidate_pf:
                candidate_pfs.append(candidate_pf)

        # filter out too short candidate place fields
        candidate_pfs_filtered = list(filter(lambda cpf: len(cpf) > PF.FORMATION_CONSECUTIVE_BINS, candidate_pfs))
        place_field_bounds = [(cpf[0], cpf[-1]) for cpf in candidate_pfs_filtered]
        return place_field_bounds

    """
    def categorize_place_fields_allow_multiple(self, cor_index=0):
        '''
        Run BTSP analysis, categorize place fields in a given corridor
        Looks for multiple place fields in a given spatial range.

        @param cor_index: corridor index, defaults to 0
        '''

        for i_cpf in range(len(self.place_field_bounds)):
            start_lap = 0
            is_reliable = True
            while start_lap < len(self.i_laps) - PF.FORMATION_LAP_WINDOW and is_reliable:
                place_field = PlaceField()

                place_field.animalID = self.animalID
                place_field.sessionID = self.sessionID
                place_field.cellid = self.cellid
                place_field.cor_index = cor_index
                place_field.i_laps = self.i_laps
                place_field.rate_matrix = self.rate_matrix
                place_field.N_pos_bins = self.rate_matrix.shape[0]

                place_field.set_bounds(self.place_field_bounds[i_cpf])
                place_field.find_formation_lap(start_lap=start_lap)
                place_field.find_end_lap()

                # filter place fields that with only intermittent activity
                if place_field.formation_lap == -1:
                    if start_lap != 0:
                        start_lap += 1 # this happens when reliable place fields have been found and we want to iterate through the remaining laps
                    else:
                        place_field.category = "unreliable"
                        self.unreliable_place_fields.append(place_field)
                        is_reliable = False
                    continue

                # skip current place field laps during the search for the next place field in the same spatial range
                start_lap = place_field.end_lap + 1

                # filter place fields that form too early to assess novelty
                if place_field.formation_lap_uncorrected < 2:
                    place_field.category = "early"
                    self.early_place_fields.append(place_field)
                    continue

                # filter place fields that are too short to meaningfully assess shift
                place_field.find_active_laps()
                if len(place_field.active_laps) <= PF.SHIFT_WINDOW + PF.SPEARMAN_SKIP_LAPS + 2:  # +2 points are at least necessary to calculate spearman correlation (else nan)
                    place_field.category = "transient"
                    self.transient_place_fields.append(place_field)
                    continue

                place_field.calculate_spearman()
                place_field.shift_score_linear_fit()
                place_field.evaluate_drift()

                if self.shift_criterion:
                    btsp_criteria = [place_field.has_high_gain, place_field.has_backwards_shift, place_field.has_no_drift]
                else:
                    btsp_criteria = [place_field.has_high_gain, place_field.has_no_drift]

                if all(btsp_criteria):
                    place_field.category = "btsp"
                    self.btsp_place_fields.append(place_field)
                else:
                    place_field.category = "non-btsp"
                    self.nonbtsp_novel_place_fields.append(place_field)
                self.candidate_btsp_place_fields.append(place_field)
    """

    def categorize_place_fields(self, cor_index=0):
        """
        Run BTSP analysis, categorize place fields in a given corridor.
        Starting from first lap, it only finds the first place field and ignores later ones.

        @param cor_index: corridor index, defaults to 0
        """

        for i_cpf in range(len(self.place_field_bounds)):
                place_field = PlaceField()

                place_field.animalID = self.animalID
                place_field.sessionID = self.sessionID
                place_field.cellid = self.cellid
                place_field.cor_index = cor_index
                place_field.i_laps = self.i_laps
                place_field.rate_matrix = self.rate_matrix
                place_field.N_pos_bins = self.rate_matrix.shape[0]

                place_field.set_bounds(self.place_field_bounds[i_cpf])
                place_field.find_formation_lap(start_lap=0)
                place_field.find_end_lap()

                # filter place fields that with only intermittent activity
                if place_field.formation_lap == -1:
                    place_field.category = "unreliable"
                    self.unreliable_place_fields.append(place_field)
                    continue

                place_field.prepare_for_nmf()
                place_field.find_active_laps()

                # calculate formation gain and shift for all reliable place fields
                # which have the minimum number of laps required for shift calculation
                if len(place_field.i_laps) > PF.FORMATION_GAIN_WINDOW and len(place_field.active_laps) > PF.SHIFT_WINDOW:
                    place_field.calculate_formation_gain()
                    place_field.calculate_shift()

                # filter place fields that form too early to assess novelty
                if place_field.formation_lap_uncorrected < 2:
                    place_field.category = "early"
                    self.early_place_fields.append(place_field)
                    continue

                # filter place fields that are too short to meaningfully assess drift
                if len(place_field.active_laps) <= PF.SHIFT_WINDOW + PF.SPEARMAN_SKIP_LAPS + 2:  # +2 points are at least necessary to calculate spearman correlation (else nan)
                    place_field.category = "transient"
                    self.transient_place_fields.append(place_field)
                    continue

                # calculate drift for place fields that are newly born + long enough for drift to be evaluated
                place_field.evaluate_drift()

                if self.shift_criterion:
                    btsp_criteria = [place_field.has_high_gain, place_field.has_backwards_shift, place_field.has_no_drift]
                else:
                    btsp_criteria = [place_field.has_high_gain, place_field.has_no_drift]

                if all(btsp_criteria):
                    place_field.category = "btsp"
                    self.btsp_place_fields.append(place_field)
                else:
                    place_field.category = "non-btsp"
                    self.nonbtsp_novel_place_fields.append(place_field)
                self.candidate_btsp_place_fields.append(place_field)

    def combine_place_field_dataframes(self):
        columns = [
            "session id", "cell id", "corridor", "category", "lower bound", "upper bound", "formation lap", "end lap",
            "is BTSP", "has high gain", "has no drift", "has backwards shift", "formation gain", "initial shift",
            "spearman r", "spearman p", "linear fit m", "linear fit b"
        ]
        place_fields_df = pd.DataFrame(columns=columns)
        place_field_categories = [
            self.unreliable_place_fields,
            self.early_place_fields,
            self.transient_place_fields,
            self.nonbtsp_novel_place_fields,
            self.btsp_place_fields
        ]
        for place_field_category in place_field_categories:
            for place_field in place_field_category:
                place_fields_df = pd.concat([place_fields_df, place_field.to_dataframe()], ignore_index=True, axis=0)
        if len(self.corridors) == 1:
            del place_fields_df["corridor"]
        return place_fields_df

    def plot_BTSP_shift_by_lap_histograms(self, save_path="BTSP"):
        place_fields_lowgain = [pf for pf in self.candidate_btsp_place_fields if not pf.has_high_gain]
        place_fields_highgain = [pf for pf in self.candidate_btsp_place_fields if pf.has_high_gain]
        if not self.candidate_btsp_place_fields or not place_fields_highgain or not place_fields_lowgain:
            print("no candidate BTSP place fields found")
            return

        output_folder = save_path + "/shift_histograms"
        folder_exists = os.path.exists(output_folder)
        if not folder_exists:
            os.makedirs(output_folder)

        # create (n_lap x n_placefields) matrix that stores shift vals for each place field in each lap since formation
        max_num_laps_lowgain = max([pf.end_lap - pf.formation_lap for pf in place_fields_lowgain])
        shift_matrix_lowgain = np.empty((max_num_laps_lowgain, len(place_fields_lowgain)))
        shift_matrix_lowgain.fill(np.nan)
        for i_pf, pf in enumerate(place_fields_lowgain):
            pf_laps = len(pf.active_laps)
            shift_matrix_lowgain[:pf_laps-PF.SHIFT_WINDOW-1,i_pf] = pf.com_diffs[1:]  # -1 because we skip the formation lap

        max_num_laps_highgain = max([pf.end_lap - pf.formation_lap for pf in place_fields_highgain])
        shift_matrix_highgain = np.empty((max_num_laps_highgain, len(place_fields_highgain)))
        shift_matrix_highgain.fill(np.nan)
        for i_pf, pf in enumerate(place_fields_highgain):
            pf_laps = len(pf.active_laps)
            shift_matrix_highgain[:pf_laps-PF.SHIFT_WINDOW-1, i_pf] = pf.com_diffs[1:]  # -1 because we skip the formation lap

        # plot distributions of shift scores for each lap after the formation lap
        for i_lap in range(min([max_num_laps_highgain, max_num_laps_lowgain])):
            if list(np.where(np.isnan(shift_matrix_lowgain[i_lap,:]) == False)[0]):  # check if it's nan everywhere, skip if it is
                plt.hist(shift_matrix_lowgain[i_lap,:], color="red", alpha=0.5)
            if list(np.where(np.isnan(shift_matrix_highgain[i_lap, :]) == False)[0]):
                plt.hist(shift_matrix_highgain[i_lap,:], color="green", alpha=0.5)
            plt.xlim([-16,10])
            plt.savefig(output_folder + f"/hist_{i_lap+1}.pdf")
            plt.close()

    def plot_BTSP_shift_gain_by_lap_scatterplots(self, save_path="BTSP"):
        place_fields_lowgain = [pf for pf in self.candidate_btsp_place_fields if not pf.has_high_gain]
        place_fields_highgain = [pf for pf in self.candidate_btsp_place_fields if pf.has_high_gain]
        if not self.candidate_btsp_place_fields or not place_fields_highgain or not place_fields_lowgain:
            print("no candidate BTSP place fields found")
            return

        output_folder = save_path + "/shift_gain_scatterplots"
        folder_exists = os.path.exists(output_folder)
        if not folder_exists:
            os.makedirs(output_folder)

        # create (n_lap x n_placefields) matrix that stores shift vals for each place field in each lap since formation
        max_num_laps_lowgain = max([pf.end_lap - pf.formation_lap for pf in place_fields_lowgain])
        shift_matrix_lowgain = np.empty((max_num_laps_lowgain, len(place_fields_lowgain)))
        shift_matrix_lowgain.fill(np.nan)
        for i_pf, pf in enumerate(place_fields_lowgain):
            pf_laps = len(pf.active_laps)
            shift_matrix_lowgain[:pf_laps-PF.SHIFT_WINDOW-1,i_pf] = pf.com_diffs[1:]  # -1 because we skip the formation lap

        max_num_laps_highgain = max([pf.end_lap - pf.formation_lap for pf in place_fields_highgain])
        shift_matrix_highgain = np.empty((max_num_laps_highgain, len(place_fields_highgain)))
        shift_matrix_highgain.fill(np.nan)
        for i_pf, pf in enumerate(place_fields_highgain):
            pf_laps = len(pf.active_laps)
            shift_matrix_highgain[:pf_laps-PF.SHIFT_WINDOW-1, i_pf] = pf.com_diffs[1:]  # -1 because we skip the formation lap

        # plot formation gains against shift scores in each lap after formation lap
        for i_lap in range(min([max_num_laps_highgain, max_num_laps_lowgain])):
            formation_gains_low = np.array([pf.formation_gain for pf in self.candidate_btsp_place_fields if pf.formation_gain < 1.0])
            formation_gains_high = np.array([pf.formation_gain for pf in self.candidate_btsp_place_fields if pf.formation_gain >= 1.0])

            plt.scatter(shift_matrix_lowgain[i_lap, :], formation_gains_low, c="r")
            plt.scatter(shift_matrix_highgain[i_lap, :], formation_gains_high, c="g")
            plt.xlim([-16,10])
            plt.savefig(output_folder + f"/scatter_{i_lap+1}.pdf")
            plt.close()

    def plot_BTSP_maxrates_all_fields_from_formation(self, save_path="BTSP", reverse=False):
        if not self.candidate_btsp_place_fields:
            print("no candidate BTSP place fields found")
            return

        fig = plt.figure(figsize=(12, 4))
        n_laps = max([pf.end_lap - pf.formation_lap for pf in self.candidate_btsp_place_fields])
        max_rates_matrix = np.empty((n_laps,len(self.candidate_btsp_place_fields)))
        max_rates_matrix.fill(np.nan)  # fill it with nan so that actual 0s can be differentiated from the end of the laps for a place field
        for i_pf, pf in enumerate(self.candidate_btsp_place_fields):
            max_rates_pf = pf.rate_matrix_pf.max(axis=0) / pf.rate_matrix_pf.max()
            if reverse:
                max_rates_matrix[-(pf.end_lap - pf.formation_lap):, i_pf] = max_rates_pf
            else:
                max_rates_matrix[:pf.end_lap - pf.formation_lap, i_pf] = max_rates_pf

        high_gain_cbtsp_fields = [pf for pf in self.candidate_btsp_place_fields if pf.formation_gain > 1.0]
        max_rates_matrix_highgain = np.empty((n_laps,len(high_gain_cbtsp_fields)))
        for i_pf, pf in enumerate(high_gain_cbtsp_fields):
            max_rates_pf = pf.rate_matrix_pf.max(axis=0) / pf.rate_matrix_pf.max()
            if reverse:
                max_rates_matrix_highgain[-(pf.end_lap - pf.formation_lap):, i_pf] = max_rates_pf
            else:
                max_rates_matrix_highgain[:pf.end_lap - pf.formation_lap, i_pf] = max_rates_pf

        low_gain_cbtsp_fields = [pf for pf in self.candidate_btsp_place_fields if pf.formation_gain <= 1.0]
        max_rates_matrix_lowgain = np.empty((n_laps,len(low_gain_cbtsp_fields)))
        max_rates_matrix_lowgain.fill(np.nan)
        for i_pf, pf in enumerate(low_gain_cbtsp_fields):
            max_rates_pf = pf.rate_matrix_pf.max(axis=0) / pf.rate_matrix_pf.max()
            if reverse:
                max_rates_matrix_lowgain[-(pf.end_lap - pf.formation_lap):, i_pf] = max_rates_pf
            else:
                max_rates_matrix_lowgain[:pf.end_lap - pf.formation_lap, i_pf] = max_rates_pf

        # plot only those laps where the majority of place fields do not have a nan rate
        if reverse:
            i_lap_majority_nan = 0
            for i_lap in range(1,n_laps+1):
                n_nans = np.count_nonzero(np.isnan(max_rates_matrix[-i_lap, :]))
                if n_nans >= len(self.candidate_btsp_place_fields) // 2:
                    i_lap_majority_nan = i_lap
                    break
        else:
            i_lap_majority_nan = -1
            for i_lap in range(n_laps):
                n_nans = np.count_nonzero(np.isnan(max_rates_matrix[i_lap,:]))
                if n_nans >= len(self.candidate_btsp_place_fields) // 2:
                    i_lap_majority_nan = i_lap
                    break

        if reverse:
            mean_all = np.nanmean(max_rates_matrix[-i_lap_majority_nan:, :], axis=1)
            mean_highgain = np.nanmean(max_rates_matrix_highgain[-i_lap_majority_nan:, :], axis=1)
            mean_lowgain = np.nanmean(max_rates_matrix_lowgain[-i_lap_majority_nan:, :], axis=1)

            std_all = np.nanstd(max_rates_matrix[-i_lap_majority_nan:, :], axis=1)
            std_highgain = np.nanstd(max_rates_matrix_highgain[-i_lap_majority_nan:, :], axis=1)
            std_lowgain = np.nanstd(max_rates_matrix_lowgain[-i_lap_majority_nan:, :], axis=1)
        else:
            mean_all = np.nanmean(max_rates_matrix[:i_lap_majority_nan, :], axis=1)
            mean_highgain = np.nanmean(max_rates_matrix_highgain[:i_lap_majority_nan, :], axis=1)
            mean_lowgain = np.nanmean(max_rates_matrix_lowgain[:i_lap_majority_nan, :], axis=1)

            std_all = np.nanstd(max_rates_matrix[:i_lap_majority_nan,:], axis=1)
            std_highgain = np.nanstd(max_rates_matrix_highgain[:i_lap_majority_nan,:], axis=1)
            std_lowgain = np.nanstd(max_rates_matrix_lowgain[:i_lap_majority_nan,:], axis=1)

        laps = np.arange(i_lap_majority_nan)

        #plt.fill_between(laps, mean_all + std_all, mean_all - std_all, color="grey", alpha=0.15)
        plt.fill_between(laps, mean_lowgain + std_lowgain, mean_lowgain - std_lowgain, color="red", alpha=0.15)
        plt.fill_between(laps, mean_highgain + std_highgain, mean_highgain - std_highgain, color="green", alpha=0.15)

        plt.plot(laps, mean_all, color="grey", linestyle="--", linewidth=2, label="all cBTSP", zorder=0)
        plt.plot(laps, mean_lowgain, color="r", linewidth=2, label="low gain cBTSP", zorder=0)
        plt.plot(laps, mean_highgain, color="g", linewidth=2, label="high gain cBTSP", zorder=0)

        plt.xlabel("laps")
        plt.legend(loc="upper right")
        filename = "/max_rates_reverse.pdf" if reverse else "/max_rates.pdf"
        plt.savefig(save_path + filename)
        plt.close()

    def plot_BTSP_shift_all_fields_from_formation(self, save_path="BTSP", reverse=False):
        if not self.candidate_btsp_place_fields:
            print("no candidate BTSP place fields found")
            return

        fig = plt.figure(figsize=(12, 4))
        n_laps = max([len(pf.active_laps) for pf in self.candidate_btsp_place_fields])

        shifts_matrix = np.empty((n_laps,len(self.candidate_btsp_place_fields)))
        shifts_matrix.fill(np.nan)  # fill it with nan so that actual 0s can be differentiated from the end of the laps for a place field
        for i_pf, pf in enumerate(self.candidate_btsp_place_fields):
            if reverse:
                shifts_matrix[-(len(pf.active_laps) - PF.SHIFT_WINDOW):, i_pf] = pf.com_diffs
            else:
                shifts_matrix[:len(pf.active_laps) - PF.SHIFT_WINDOW, i_pf] = pf.com_diffs

        high_gain_cbtsp_fields = [pf for pf in self.candidate_btsp_place_fields if pf.formation_gain > 1.0]
        shifts_matrix_highgain = np.empty((n_laps,len(high_gain_cbtsp_fields)))
        shifts_matrix_highgain.fill(np.nan)
        for i_pf, pf in enumerate(high_gain_cbtsp_fields):
            if reverse:
                shifts_matrix_highgain[-(len(pf.active_laps) - PF.SHIFT_WINDOW):, i_pf] = pf.com_diffs
            else:
                shifts_matrix_highgain[:len(pf.active_laps) - PF.SHIFT_WINDOW, i_pf] = pf.com_diffs

        low_gain_cbtsp_fields = [pf for pf in self.candidate_btsp_place_fields if pf.formation_gain <= 1.0]
        shifts_matrix_lowgain = np.empty((n_laps,len(low_gain_cbtsp_fields)))
        shifts_matrix_lowgain.fill(np.nan)
        for i_pf, pf in enumerate(low_gain_cbtsp_fields):
            if reverse:
                shifts_matrix_lowgain[-(len(pf.active_laps) - PF.SHIFT_WINDOW):, i_pf] = pf.com_diffs
            else:
                shifts_matrix_lowgain[:len(pf.active_laps) - PF.SHIFT_WINDOW, i_pf] = pf.com_diffs

        # plot only those laps where the majority of place fields do not have a nan shift
        if reverse:
            i_lap_majority_nan = 0
            for i_lap in range(1, n_laps + 1):
                n_nans = np.count_nonzero(np.isnan(shifts_matrix[-i_lap, :]))
                if n_nans >= len(self.candidate_btsp_place_fields) // 2:
                    i_lap_majority_nan = i_lap
                    break
        else:
            i_lap_majority_nan = -1
            for i_lap in range(n_laps):
                n_nans = np.count_nonzero(np.isnan(shifts_matrix[i_lap, :]))
                if n_nans >= len(self.candidate_btsp_place_fields) // 2:
                    i_lap_majority_nan = i_lap
                    break

        if reverse:
            mean_all = np.nanmean(shifts_matrix[-i_lap_majority_nan:, :], axis=1)
            mean_highgain = np.nanmean(shifts_matrix_highgain[-i_lap_majority_nan:, :], axis=1)
            mean_lowgain = np.nanmean(shifts_matrix_lowgain[-i_lap_majority_nan:, :], axis=1)

            std_all = np.nanstd(shifts_matrix[-i_lap_majority_nan:, :], axis=1)
            std_highgain = np.nanstd(shifts_matrix_highgain[-i_lap_majority_nan:, :], axis=1)
            std_lowgain = np.nanstd(shifts_matrix_lowgain[-i_lap_majority_nan:, :], axis=1)
        else:
            mean_all = np.nanmean(shifts_matrix[:i_lap_majority_nan, :], axis=1)
            mean_highgain = np.nanmean(shifts_matrix_highgain[:i_lap_majority_nan, :], axis=1)
            mean_lowgain = np.nanmean(shifts_matrix_lowgain[:i_lap_majority_nan, :], axis=1)

            std_all = np.nanstd(shifts_matrix[:i_lap_majority_nan, :], axis=1)
            std_highgain = np.nanstd(shifts_matrix_highgain[:i_lap_majority_nan, :], axis=1)
            std_lowgain = np.nanstd(shifts_matrix_lowgain[:i_lap_majority_nan, :], axis=1)

        laps = np.arange(i_lap_majority_nan)

        #plt.fill_between(laps, mean_all + std_all, mean_all - std_all, color="grey", alpha=0.15)
        plt.fill_between(laps, mean_lowgain + std_lowgain, mean_lowgain - std_lowgain, color="red", alpha=0.15)
        plt.fill_between(laps, mean_highgain + std_highgain, mean_highgain - std_highgain, color="green", alpha=0.15)

        plt.plot(laps, mean_all, color="grey", linestyle="--", linewidth=2, label="all cBTSP", zorder=0)
        plt.plot(laps, mean_lowgain, color="r", linewidth=2, label="low gain cBTSP", zorder=0)
        plt.plot(laps, mean_highgain, color="g", linewidth=2, label="high gain cBTSP", zorder=0)

        plt.xlabel("active laps")
        plt.legend(loc="upper right")
        filename = "/shift_scores_reverse.pdf" if reverse else "/shift_scores.pdf"
        plt.savefig(save_path + filename)
        plt.close()

    def plot_BTSP_shift_all_fields_sorted_by_correlation(self, save_path="BTSP"):
        if not self.candidate_btsp_place_fields:
            print("no candidate BTSP place fields found")
            return

        place_fields_sorted = sorted(self.candidate_btsp_place_fields, key=lambda pf: pf.spearman_corrs[0] if not np.isnan(pf.spearman_corrs[0]) else 2)  # r can be in range [-1,1] so 2 places nans at the end
        n_rows = len(self.candidate_btsp_place_fields) // 10 + 1
        n_cols = 10
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(36,n_rows*6))
        for i_pf, pf in enumerate(place_fields_sorted):
            r, p = pf.spearman_corrs
            i_row = i_pf // 10
            i_col = i_pf % 10
            laps = np.arange(len(pf.com_diffs))

            if np.isnan(p) or np.isnan(r):
                color = 'b'
            elif p <= 0.05:
                color = 'r'
            else:
                color = 'k'

            ax[i_row, i_col].plot(laps[:PF.SPEARMAN_SKIP_LAPS+1], pf.com_diffs[:PF.SPEARMAN_SKIP_LAPS+1], color, linestyle="--", label="_nolegend_")
            ax[i_row, i_col].plot(laps[PF.SPEARMAN_SKIP_LAPS:], pf.com_diffs[PF.SPEARMAN_SKIP_LAPS:], color, linewidth=2, label=f"r={np.round(r,2)}, p={np.round(p,3)}")
            if ax[i_row, i_col].lines: ax[i_row, i_col].legend(loc="upper right")
            lb, ub = pf.bounds
            ax[i_row, i_col].title.set_text(f"cell: {pf.cellid}, pf bounds: {lb,ub}")

            if not np.isnan(pf.linear_coeffs[0]):
                m, b = pf.linear_coeffs
                ax[i_row, i_col].axline(xy1=(PF.SPEARMAN_SKIP_LAPS,b), slope=m, color="green")
                ax[i_row, i_col].plot(0, m*(-PF.SPEARMAN_SKIP_LAPS)+b, marker="x", markersize=10, color="green")

            ax[i_row, i_col].axvline(0, color="k", linestyle="dotted")
            ax[i_row, i_col].axhline(0, color="k", linestyle="dotted")

        # hide empty plots
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                if not ax[i_row, i_col].lines: ax[i_row, i_col].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path + "/correlations_sorted.pdf")
        plt.close()

    def save_BTSP_statistics(self, save_path="BTSP"):
        # number of place fields
        n_unreliable_pf = len(self.unreliable_place_fields)
        n_early_pf = len(self.early_place_fields)
        n_transient_pfs = len(self.transient_place_fields)
        n_cbtsp_pf = len(self.candidate_btsp_place_fields)
        n_btsp_pf = len(self.btsp_place_fields)
        n_total_pf = n_unreliable_pf + n_early_pf + n_transient_pfs + n_cbtsp_pf  # don't add 'pure' BTSPs because they're a subset of cBTSPs

        # number of various groups of candidate BTSP place fields
        n_low_gain_pf = len([pf for pf in self.nonbtsp_novel_place_fields if not pf.has_high_gain])
        n_drifting_pf = len([pf for pf in self.nonbtsp_novel_place_fields if not pf.has_no_drift])
        n_forward_shifting_pf = len([pf for pf in self.nonbtsp_novel_place_fields if not pf.has_backwards_shift])

        # proportion of place fields
        pct_unreliable_pf = round(100 * n_unreliable_pf / n_total_pf, 1) if n_total_pf > 0 else None
        pct_early_pf = round(100 * n_early_pf / n_total_pf, 1) if n_total_pf > 0 else None
        pct_transient_pf = round(100 * n_transient_pfs / n_total_pf, 1) if n_total_pf > 0 else None
        pct_cbtsp_pf = round(100 * n_cbtsp_pf / n_total_pf, 1) if n_total_pf > 0 else None
        pct_btsp_pf = round(100 * n_btsp_pf / n_total_pf, 1) if n_total_pf > 0 else None

        # proportion of various groups of cBTSP-PF
        pct_low_gain_pf = round(100 * n_low_gain_pf / len(self.nonbtsp_novel_place_fields), 1) if len(self.nonbtsp_novel_place_fields) > 0 else None
        pct_drifting_pf = round(100 * n_drifting_pf / len(self.nonbtsp_novel_place_fields), 1) if len(self.nonbtsp_novel_place_fields) > 0 else None
        pct_forward_shifting_pf = round(100 * n_forward_shifting_pf / len(self.nonbtsp_novel_place_fields), 1) if len(self.nonbtsp_novel_place_fields) > 0 else None
        pct_btsp_pf_out_of_candidates = round(100 * n_btsp_pf / n_cbtsp_pf, 1) if n_cbtsp_pf > 0 else None

        # number of cells for various place field categories
        n_cells_unreliable_pf = len(set([f"{pf.sessionID}_{pf.cellid}" for pf in self.unreliable_place_fields]))
        n_cells_early_pf = len(set([f"{pf.sessionID}_{pf.cellid}" for pf in self.early_place_fields]))
        n_cells_transient_pf = len(set([f"{pf.sessionID}_{pf.cellid}" for pf in self.transient_place_fields]))
        n_cells_cbtsp_pf = len(set([f"{pf.sessionID}_{pf.cellid}" for pf in self.candidate_btsp_place_fields]))
        n_cells_btsp_pf = len(set([f"{pf.sessionID}_{pf.cellid}" for pf in self.btsp_place_fields]))
        n_cells_nonbtsp_pf = len(set([f"{pf.sessionID}_{pf.cellid}" for pf in self.nonbtsp_novel_place_fields]))

        # total number of cells with any kind of detected place field
        pf_subsets = [self.unreliable_place_fields, self.early_place_fields, self.transient_place_fields, self.candidate_btsp_place_fields]
        cellids = []
        for pf_subset in pf_subsets:
            cellids += [f"{pf.sessionID}_{pf.cellid}" for pf in pf_subset]
        unique_cellids = set(cellids)
        n_total_cells = len(unique_cellids)

        # proportion of cells
        pct_cells_unreliable_pf = round(100 * n_cells_unreliable_pf / n_total_cells, 1) if n_total_cells > 0 else None
        pct_cells_early_pf = round(100 * n_cells_early_pf / n_total_cells, 1) if n_total_cells > 0 else None
        pct_cells_transient_pf = round(100 * n_cells_transient_pf / n_total_cells, 1) if n_total_cells > 0 else None
        pct_cells_cbtsp_pf = round(100 * n_cells_cbtsp_pf / n_total_cells, 1) if n_total_cells > 0 else None
        pct_cells_btsp_pf = round(100 * n_cells_btsp_pf / n_total_cells, 1) if n_total_cells > 0 else None
        pct_cells_non_btsp_pf = round(100 * n_cells_nonbtsp_pf / n_total_cells, 1) if n_total_cells > 0 else None

        stats_list = [
            ("Number of place fields", ""),
            ("unreliable", n_unreliable_pf),
            ("early", n_early_pf),
            ("transient", n_transient_pfs),
            ("candidate BTSP (cBTSP)", n_cbtsp_pf),
            ("total", n_total_pf),
            ("BTSP", n_btsp_pf),
            ("", ""),
            ("Number of candidate PFs with missing signatures", ""),
            ("drifting cBTSP", n_drifting_pf),
            ("low gain cBTSP", n_low_gain_pf),
            ("forward shifting cBTSP", n_forward_shifting_pf),
            ("", ""),
            ("Distribution of place fields (%)", ""),
            ("unreliable", pct_unreliable_pf),
            ("early", pct_early_pf),
            ("transient", pct_transient_pf),
            ("candidate BTSP", pct_cbtsp_pf),
            ("BTSP (out of all PFs)", pct_btsp_pf),
            ("BTSP (out of candidates)", pct_btsp_pf_out_of_candidates),
            ("", ""),
            ("Distribution of missing signatures among candidates (%)", ""),
            ("drifting cBTSP", pct_drifting_pf),
            ("low gain cBTSP", pct_low_gain_pf),
            ("forward shifting cBTSP", pct_forward_shifting_pf),
            ("", ""),
            ("Number of cells", ""),
            ("with unreliable PF", n_cells_unreliable_pf),
            ("with early PF", n_cells_early_pf),
            ("with transient PF", n_cells_transient_pf),
            ("with candidate BTSP PF", n_cells_cbtsp_pf),
            ("total", n_total_cells),
            ("with BTSP PF", n_cells_btsp_pf),
            ("with non-BTSP novel PF", n_cells_nonbtsp_pf),
            ("", ""),
            ("Distribution of cells with various PFs (%)", ""),
            ("with unreliable PF", pct_cells_unreliable_pf),
            ("with early PF", pct_cells_early_pf),
            ("with transient PF", pct_cells_transient_pf),
            ("with candidate BTSP PF", pct_cells_cbtsp_pf),
            ("with BTSP PF", pct_cells_btsp_pf),
            ("with non-BTSP novel PF", pct_cells_non_btsp_pf)
        ]

        with open(f'{save_path}/btsp_stats.csv', 'w') as stats_file:
            writer = csv.writer(stats_file, delimiter=",")
            for s_name, s_val in stats_list:
                writer.writerow([s_name, s_val])

    def write_corrs_into_file(self, save_path):
        for pf in self.candidate_btsp_place_fields:
            r, p = pf.spearman_corrs
            if r >= 0 or p >= 0.05:
                continue
            filename = f"cell {pf.cellid}, r={round(r,2)}, p={round(p,2)}"
            with open(f"{save_path}/{filename}", "w") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(["lap", "shift"])
                com_diffs_after_shift = pf.com_diffs[PF.SPEARMAN_SKIP_LAPS:]
                lap_indices_after_shift = np.arange(0, len(com_diffs_after_shift))
                for i in range(len(lap_indices_after_shift)):
                    writer.writerow([lap_indices_after_shift[i], com_diffs_after_shift[i]])

    def save_BTSP_place_field_categories(self, save_path="BTSP"):
        categories = {"unreliable": self.unreliable_place_fields,
                      "early": self.early_place_fields,
                      "transient": self.transient_place_fields,
                      "non-btsp": self.nonbtsp_novel_place_fields,
                      "btsp": self.btsp_place_fields}
        with open(f"{save_path}/placefield_categories.csv", "w") as categories_file:
            writer = csv.writer(categories_file, delimiter=",")
            writer.writerow(["category", "session", "cellid", "corridor", "first bin", "last bin"])
            for category_name, place_fields in categories.items():
                place_fields_info = [(pf.sessionID, pf.cellid, pf.cor_index, pf.bounds[0], pf.bounds[1]) for pf in place_fields]
                #place_fields_info = [(pf.sessionID, pf.cellid, self.corridors[pf.cor_index], pf.bounds[0], pf.bounds[1]) for pf in place_fields]
                for pf_info in place_fields_info:
                    writer.writerow([category_name, *pf_info])

    def create_nmf_matrix(self, type=""):
        if type == "non_btsp":
            place_fields = self.nonbtsp_novel_place_fields
        elif type == "newly":
            place_fields = self.transient_place_fields + self.nonbtsp_novel_place_fields + self.btsp_place_fields
        elif type == "candidate_btsp":
            place_fields = self.btsp_place_fields + self.nonbtsp_novel_place_fields
        elif type == "btsp":
            place_fields = self.btsp_place_fields
        else:
            place_fields = self.early_place_fields + self.transient_place_fields + self.btsp_place_fields + self.nonbtsp_novel_place_fields

        nmf_matrix = None
        for place_field in place_fields:
            if place_field.rate_matrix_pf_centered is None:
                continue
            rate_matrix_flat = place_field.rate_matrix_pf_centered.T.flatten()
            if nmf_matrix is None:
                nmf_matrix = rate_matrix_flat
            else:
                nmf_matrix = np.vstack((nmf_matrix, rate_matrix_flat))
        return nmf_matrix
