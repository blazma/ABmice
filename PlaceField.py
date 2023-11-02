import pandas as pd
import numpy as np
import scipy

FORMATION_CONSECUTIVE_BINS = 3  # number of consecutive bins that needs to show significant activity in order to consider place field at all
FORMATION_THRESHOLD = 0.1  # percent of max activity across whole place field that activity in laps must surpass to consider them active laps
FORMATION_LAP_WINDOW = 5
FORMATION_MINIMUM_ACTIVE_LAPS = 3  # number of laps that need to be active in the formation lap window
FORMATION_GAIN_WINDOW = 5
END_THRESHOLD = 0.1  # percent of max activity across whole place field that activity in laps must surpass to consider them active laps
END_LAP_WINDOW = 3
FORWARDS_BOUND_EXTENSION = 5 # number of bins to extend place fields with in the forwards direction
SHIFT_WINDOW = 3
SHIFT_THRESHOLD = 0.1
SPEARMAN_SKIP_LAPS = 3 # skip some laps following the formation lap to allow for shifting
NMF_MINIMUM_LAPS = 10  # minimum number of laps for inclusion in non-negative matrix factorization
NMF_STD_PF_WIDTH = 6  # comes from CA1 data

class PlaceField:
    def __init__(self):
        # measurement attribute
        self.animalID = None
        self.sessionID = None
        self.cellid = -1
        self.cor_index = -1
        self.i_laps = []
        self.rate_matrix = np.empty(0)
        self.N_pos_bins = -1

        # place field attributes
        self.bins = []
        self.bounds = (-1, -1) # lb, ub
        self.formation_lap_uncorrected = -1
        self.formation_lap = -1
        self.formation_bin = -1
        self.end_lap = -1
        self.formation_gain = np.nan
        self.com_diffs = [0]  # shift scores
        self.spearman_corrs = (np.nan, np.nan)  # r, p
        self.linear_coeffs = (np.nan, np.nan)  # m, b; from the linear regression of shift scores
        self.rate_matrix_pf = np.empty(0)  # rate matrix restricted to [formation_lap:end_lap] and [lb:ub]
        self.active_laps = []  # active laps based on shift threshold
        self.category = ""
        self.notes = ""
        self.rate_matrix_pf_centered = None  # used for NMF

        # BTSP signatures`
        self.has_high_gain = False
        self.has_no_drift = False
        self.has_backwards_shift = False

    def add_note(self, note):
        if self.notes == "":
            self.notes = note
        else:
            self.notes += f"; {note}"

    def to_dataframe(self):
        is_btsp = self.has_high_gain and self.has_no_drift and self.has_backwards_shift
        initial_shift = np.mean(self.com_diffs[1:1+SPEARMAN_SKIP_LAPS])
        data = [
            self.animalID, self.sessionID, self.cellid, self.cor_index, self.category, self.bounds[0], self.bounds[1], self.formation_lap,
            self.formation_bin, self.end_lap, is_btsp, self.has_high_gain, self.has_no_drift, self.has_backwards_shift, self.formation_gain,
            initial_shift, self.spearman_corrs[0], self.spearman_corrs[1], self.linear_coeffs[0], self.linear_coeffs[1], self.notes
        ]
        columns = [
            "animal id", "session id", "cell id", "corridor", "category", "lower bound", "upper bound", "formation lap",
            "formation bin", "end lap", "is BTSP", "has high gain", "has no drift", "has backwards shift", "formation gain", "initial shift",
            "spearman r", "spearman p", "linear fit m", "linear fit b", "notes"
        ]
        df = pd.DataFrame(columns=columns)
        df.loc[0] = data
        return df

    def set_bounds(self, bounds):
        lb, ub = bounds[0], bounds[1]
        if ub + FORWARDS_BOUND_EXTENSION < self.N_pos_bins:
            ub += FORWARDS_BOUND_EXTENSION
        else:
            ub = self.N_pos_bins
        self.bounds = (lb, ub)

    def find_formation_lap(self, start_lap=0):
        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        rate_matrix_cpf = self.rate_matrix[lb:ub, :]

        formation_lap = -1
        formation_lap_uncorrected = -1
        for i in range(start_lap, len(self.i_laps) - FORMATION_LAP_WINDOW):
            rate_matrix_window = rate_matrix_cpf[:, i:i + FORMATION_LAP_WINDOW]
            max_rates_window = np.nanmax(rate_matrix_window, axis=0) / np.nanmax(rate_matrix_cpf)  # % of max rates in window wrt whole candidate pf
            active_laps_window = np.where(max_rates_window > FORMATION_THRESHOLD)[0]
            if len(active_laps_window) >= FORMATION_MINIMUM_ACTIVE_LAPS:
                formation_lap_uncorrected = i + active_laps_window[0]
                geq_50pct = np.where(max_rates_window >= 0.5)[0]
                if len(geq_50pct) > 0:
                    formation_lap = i + geq_50pct[0]
                else:
                    formation_lap = i + formation_lap_uncorrected
                break
        self.formation_lap_uncorrected = formation_lap_uncorrected
        self.formation_lap = formation_lap

        try:
            if formation_lap > -1:
                pf_bins_range = np.arange(0, ub - lb)
                rate_matrix_from_formation_lap = self.rate_matrix[lb:ub, self.formation_lap:]
                formation_bin = lb + np.average(pf_bins_range, weights=rate_matrix_from_formation_lap[:, 0])
                self.formation_bin = formation_bin
        except Exception:
            pass  # TODO: ezt jobban megnézni, van amikor ZeroDivisionError, máskor IndexError

    def find_end_lap(self):
        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        rate_matrix_cpf = self.rate_matrix[lb:ub, :]
        end_lap = self.formation_lap
        for i in range(self.formation_lap, len(self.i_laps)):
            laps_left = len(self.i_laps) - i  # number of laps between end of window and last lap
            if laps_left <= END_LAP_WINDOW:
                rate_matrix_window = rate_matrix_cpf[:, i:i + laps_left]
            else:
                rate_matrix_window = rate_matrix_cpf[:, i:i + END_LAP_WINDOW]
            max_rates_window = np.nanmax(rate_matrix_window,axis=0) / np.nanmax(rate_matrix_cpf)  # % of max rates in window wrt whole candidate pf
            active_laps_window = np.where(max_rates_window > END_THRESHOLD)[0]
            if len(active_laps_window) == 0:
                break
            end_lap = i
        self.end_lap = end_lap
        self.rate_matrix_pf = self.rate_matrix[lb:ub, self.formation_lap:self.end_lap]

    def calculate_formation_gain(self):
        formation_rate = np.nanmax(self.rate_matrix_pf[:, 0])
        window_rate = np.nanmax(self.rate_matrix_pf[:, 1:1 + FORMATION_GAIN_WINDOW],axis=0).mean()
        formation_gain = formation_rate / window_rate
        self.formation_gain = formation_gain
        if formation_gain > 1.0:
            self.has_high_gain = True

    def find_active_laps(self):
        active_laps = []
        for i in range(self.end_lap - self.formation_lap):
            max_rate_normalized = np.nanmax(self.rate_matrix_pf[:,i]) / np.nanmax(self.rate_matrix_pf)  # TODO: during normalization, should we consider the whole field or just between formation lap and end lap?
            if (max_rate_normalized >= SHIFT_THRESHOLD):
                active_laps.append(i)
        self.active_laps = active_laps

    def calculate_shift(self):
        rate_matrix_active_laps = self.rate_matrix_pf[:,self.active_laps]

        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        pf_bins_range = np.arange(0, ub - lb)

        if all(self.rate_matrix_pf[:,0] == 0):
            self.add_note("empty formation lap")
            return

        formation_bin = np.average(pf_bins_range, weights=self.rate_matrix_pf[:, 0])
        for i in range(1, len(self.active_laps) - SHIFT_WINDOW):
            rate_matrix_window = rate_matrix_active_laps[:, i:i + SHIFT_WINDOW]

            # moving average of centers of mass (com)
            coms_window = np.zeros(SHIFT_WINDOW)
            for i in range(SHIFT_WINDOW):
                rates_lap = rate_matrix_window[:,i]
                coms_window[i] = np.average(pf_bins_range, weights=rates_lap)
            avg_com_window = coms_window.mean()
            com_diff = avg_com_window - formation_bin
            self.com_diffs.append(com_diff)
        if len(self.com_diffs) >= 1+SPEARMAN_SKIP_LAPS:
            if np.mean(self.com_diffs[1:1+SPEARMAN_SKIP_LAPS]) < 0:
                self.has_backwards_shift = True

    def calculate_spearman(self):
        com_diffs_after_shift = self.com_diffs[SPEARMAN_SKIP_LAPS:]
        lap_indices_after_shift = np.arange(0, len(com_diffs_after_shift))
        spearman = scipy.stats.spearmanr(lap_indices_after_shift, com_diffs_after_shift)
        self.spearman_corrs = (spearman.correlation, spearman.pvalue)

    def shift_score_linear_fit(self):
        com_diffs_after_shift = self.com_diffs[SPEARMAN_SKIP_LAPS:]
        lap_indices_after_shift = np.arange(0, len(com_diffs_after_shift))
        self.linear_coeffs = np.polyfit(lap_indices_after_shift, com_diffs_after_shift, 1)

    def evaluate_drift(self):
        if len(self.com_diffs) == 0:
            self.add_note("empty shift score list")
        elif len(self.com_diffs) < SPEARMAN_SKIP_LAPS + 2:
            self.add_note(f"shift score list too short (len={len(self.com_diffs)})")
            return

        self.calculate_spearman()
        self.shift_score_linear_fit()

        r, p = self.spearman_corrs
        m, b = self.linear_coeffs
        shift_fit_formation_lap = m * (-SPEARMAN_SKIP_LAPS) + b  # extrapolation of linear fit on shift score to formation lap
        first_laps_avg = np.mean(self.com_diffs[:SPEARMAN_SKIP_LAPS+1])
        if r > 0 or p > 0.05 or shift_fit_formation_lap < 0:  # we don't care about forward drift here
            self.has_no_drift = True

    def prepare_for_nmf(self):
        lb, ub = self.bounds
        if len(self.i_laps) - self.formation_lap > NMF_MINIMUM_LAPS and lb > FORWARDS_BOUND_EXTENSION:
            lb = lb-FORWARDS_BOUND_EXTENSION  # in order to center PF we need to account for FBE and extend in the other direction too
        else:
            return
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)

        rate_matrix_pf_uncentered = self.rate_matrix[lb:ub, self.formation_lap:self.formation_lap+NMF_MINIMUM_LAPS]
        pf_bins_range = np.arange(0, ub - lb)
        coms_laps = np.zeros(NMF_MINIMUM_LAPS)
        for i_lap in range(NMF_MINIMUM_LAPS):
            rates_lap = rate_matrix_pf_uncentered[:,i_lap]
            if all(rates_lap == 0):
                coms_laps[i_lap] = np.nan
            else:
                coms_laps[i_lap] = np.average(pf_bins_range, weights=rates_lap)
        avg_com = np.nanmean(coms_laps)
        center_bin = int(lb + np.round(avg_com))
        centered_lb = center_bin - 4*NMF_STD_PF_WIDTH
        centered_ub = center_bin + 4*NMF_STD_PF_WIDTH

        if centered_lb <= 0 or centered_ub >= self.N_pos_bins:
            return

        self.rate_matrix_pf_centered = self.rate_matrix[centered_lb:centered_ub, self.formation_lap:self.formation_lap+NMF_MINIMUM_LAPS]
