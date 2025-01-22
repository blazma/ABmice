import pandas as pd
import numpy as np
import scipy


class PlaceField:
    def __init__(self):
        # measurement attribute
        self.animalID = None
        self.sessionID = None
        self.cellid = -1
        self.cor_index = -1
        self.history = "ALL"  # can be ALL, STAY or CHANGE depending on lap history
        self.i_laps = []
        self.rate_matrix = np.empty(0)
        self.N_pos_bins = -1
        self.lap_histories = []
        self.formation_lap_history = None

        # params
        self.params = {}

        # place field attributes
        self.bins = []
        self.bounds = (-1, -1) # lb, ub
        self.formation_lap_uncorrected = -1
        self.formation_lap = -1
        self.formation_bin = -1
        self.end_lap = -1
        self.dF_F_maxima = np.empty(0)
        self.formation_gain = np.nan
        self.formation_gain_AL5 = np.nan  # AL5 = after (active) lap 5
        self.initial_shift = np.nan
        self.initial_shift_AL5 = np.nan
        self.coms = []  # stores COMs of each active lap
        self.com_diffs = [0]  # shift scores
        self.spearman_corrs = (np.nan, np.nan)  # r, p
        self.linear_coeffs = (np.nan, np.nan)  # m, b; from the linear regression of shift scores
        self.rate_matrix_pf = np.empty(0)  # rate matrix restricted to [formation_lap:end_lap] and [lb:ub]
        self.active_laps = []  # active laps based on shift threshold
        self.category = ""
        self.notes = ""
        self.rate_matrix_pf_centered = None  # used for NMF
        self.formation_rate_sum = -1  # integral of firing rate should be proportional to Ca2+ signal size
        self.cv_coms = -1
        self.com_diffs_lap_by_lap = []

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
        df_dict = {
            "animal id": self.animalID,
            "session id": self.sessionID,
            "cell id": self.cellid,
            "corridor": self.cor_index,
            "history": self.history,
            "formation lap history": self.formation_lap_history,
            "category": self.category,
            "lower bound": self.bounds[0],
            "upper bound": self.bounds[1],
            "formation lap": self.formation_lap,
            "formation lap uncorrected": self.formation_lap_uncorrected,
            "formation bin": self.formation_bin,
            "end lap": self.end_lap,
            "is BTSP": is_btsp,
            "has high gain": self.has_high_gain,
            "has no drift": self.has_no_drift,
            "has backwards shift": self.has_backwards_shift,
            "formation gain": self.formation_gain,
            "formation gain AL5": self.formation_gain_AL5,
            "initial shift": self.initial_shift,
            "initial shift AL5": self.initial_shift_AL5,
            "spearman r": self.spearman_corrs[0],
            "spearman p": self.spearman_corrs[1],
            "linear fit m": self.linear_coeffs[0],
            "linear fit b": self.linear_coeffs[1],
            "formation rate sum": self.formation_rate_sum,
            "CV(COMs)": self.cv_coms,
            "shift scores": [self.com_diffs],
            "dF/F maxima": [self.dF_F_maxima],
            "dCOM (lap-by-lap)": [self.com_diffs_lap_by_lap],
            "session length": self.rate_matrix.shape[1],
            "notes": self.notes
        }
        df = pd.DataFrame.from_dict([df_dict])
        return df

    def set_bounds(self, bounds):
        lb, ub = bounds[0], bounds[1]
        if ub + self.params["FORWARDS_BOUND_EXTENSION"] < self.N_pos_bins:
            ub += self.params["FORWARDS_BOUND_EXTENSION"]
        else:
            ub = self.N_pos_bins
        self.bounds = (lb, ub)

    def find_formation_lap(self, start_lap=0, correction=False):
        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        rate_matrix_cpf = self.rate_matrix[lb:ub, :]

        formation_lap = -1
        formation_lap_uncorrected = -1
        for i in range(start_lap, len(self.i_laps) - self.params["FORMATION_LAP_WINDOW"]):
            rate_matrix_window = rate_matrix_cpf[:, i:i + self.params["FORMATION_LAP_WINDOW"]]
            max_rates_window = np.nanmax(rate_matrix_window, axis=0) / np.nanmax(rate_matrix_cpf)  # % of max rates in window wrt whole candidate pf
            active_laps_window = np.where(max_rates_window > self.params["FORMATION_THRESHOLD"])[0]
            if len(active_laps_window) >= self.params["FORMATION_MINIMUM_ACTIVE_LAPS"]:
                formation_lap_uncorrected = i + active_laps_window[0]
                geq_50pct = np.where(max_rates_window >= 0.5)[0]
                if len(geq_50pct) > 0:
                    formation_lap = i + geq_50pct[0]
                else:
                    formation_lap = formation_lap_uncorrected
                break

        self.formation_lap_uncorrected = formation_lap_uncorrected
        self.formation_lap = formation_lap_uncorrected
        if correction == True:
            self.formation_lap = formation_lap

        if self.formation_lap > -1:
            self.formation_lap_history = self.lap_histories[self.formation_lap]
        else:
            self.formation_lap_history = "N/A"

        try:
            if formation_lap > -1:
                pf_bins_range = np.arange(0, ub - lb)
                rate_matrix_from_formation_lap = self.rate_matrix[lb:ub, self.formation_lap:]
                formation_bin = lb + np.average(pf_bins_range, weights=rate_matrix_from_formation_lap[:, 0])
                self.formation_bin = formation_bin
                self.formation_rate_sum = np.sum(rate_matrix_from_formation_lap[:,0])
        except Exception:
            pass  # TODO: ezt jobban megnézni, van amikor ZeroDivisionError, máskor IndexError

    def find_end_lap(self):
        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        #ub = ub if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        rate_matrix_cpf = self.rate_matrix[lb:ub, :]
        end_lap = self.formation_lap
        for i in range(self.formation_lap, len(self.i_laps)):
            laps_left = len(self.i_laps) - i  # number of laps between end of window and last lap
            if laps_left <= self.params["END_LAP_WINDOW"]:
                rate_matrix_window = rate_matrix_cpf[:, i:i + laps_left]
            else:
                rate_matrix_window = rate_matrix_cpf[:, i:i + self.params["END_LAP_WINDOW"]]
            max_rates_window = np.nanmax(rate_matrix_window,axis=0) / np.nanmax(rate_matrix_cpf)  # % of max rates in window wrt whole candidate pf
            active_laps_window = np.where(max_rates_window > self.params["END_THRESHOLD"])[0]
            if len(active_laps_window) == 0:
                break
            end_lap = i
        self.end_lap = end_lap
        self.rate_matrix_pf = self.rate_matrix[lb:ub, self.formation_lap:self.end_lap]

    def calculate_formation_gain(self):
        formation_rate = np.nanmax(self.rate_matrix_pf[:, 0])

        #### active laps only
        window_rate = np.nanmax(self.rate_matrix_pf[:,self.active_laps[1:1+self.params["FORMATION_GAIN_WINDOW"]]],axis=0).mean()
        formation_gain = formation_rate / window_rate

        self.formation_gain = formation_gain
        if formation_gain > 1.0:
            self.has_high_gain = True

        # self-control - formation gain calculated at 5th active lap
        if len(self.active_laps) < 5+self.params["FORMATION_GAIN_WINDOW"]:
            return
        reference_rate = np.nanmax(self.rate_matrix_pf[:, self.active_laps[5]])
        window_rate_AL5 = np.nanmax(self.rate_matrix_pf[:, self.active_laps[6:6 + self.params["FORMATION_GAIN_WINDOW"]]],axis=0).mean()
        formation_gain_AL5 = reference_rate / window_rate_AL5
        self.formation_gain_AL5 = formation_gain_AL5

        #### all laps
        #window_rate = np.nanmax(self.rate_matrix_pf[:, 1:1 + self.params["FORMATION_GAIN_WINDOW"]],axis=0).mean()
        #formation_gain = formation_rate / window_rate

        #### 1st window to 2nd window ratio
        #w1_rate = np.nanmax(self.rate_matrix_pf[:,self.active_laps[0:self.params["FORMATION_GAIN_WINDOW"]]], axis=0).mean()
        #w2_rate = np.nanmax(self.rate_matrix_pf[:,self.active_laps[self.params["FORMATION_GAIN_WINDOW"]]:2*self.params["FORMATION_GAIN_WINDOW"]], axis=0).mean()
        #formation_gain = w1_rate / w2_rate

        #### 1st lap to 2nd lap ratio
        #formation_gain = np.nanmax(self.rate_matrix_pf[:, self.active_laps[0]]) / np.nanmax(self.rate_matrix_pf[:, self.active_laps[1]])

    def find_active_laps(self):
        active_laps = []
        for i in range(self.end_lap - self.formation_lap):
            max_rate_normalized = np.nanmax(self.rate_matrix_pf[:,i]) / np.nanmax(self.rate_matrix_pf)  # TODO: during normalization, should we consider the whole field or just between formation lap and end lap?
            if (max_rate_normalized >= self.params["SHIFT_THRESHOLD"]):
                active_laps.append(i)
        self.active_laps = active_laps

    def calculate_dF_F_maxima(self, frames_pos_bins, frames_dF_F):
        lb, ub = self.bounds
        lifespan = self.end_lap - self.formation_lap
        dF_F_maxima = []
        for i_lap in range(self.formation_lap, self.end_lap):
            frames_pf_mask = (frames_pos_bins[i_lap] >= lb) & (frames_pos_bins[i_lap] <= ub)
            frames_pf_dF_F = frames_dF_F[i_lap][frames_pf_mask]
            dF_F_maxima.append(np.nanmax(frames_pf_dF_F))
        self.dF_F_maxima = np.array(dF_F_maxima)

    def calculate_lap_by_lap_COM_diffs(self):
        com_diffs = []
        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        #ub = ub if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        pf_bins_range = np.arange(0, ub - lb)
        lifespan = self.end_lap - self.formation_lap
        for i_lap in range(lifespan-1):
            try:
                coms_lap = np.average(pf_bins_range, weights=self.rate_matrix_pf[:,i_lap])
                coms_consec_lap = np.average(pf_bins_range, weights=self.rate_matrix_pf[:,i_lap+1])
                com_diffs.append(coms_consec_lap - coms_lap)
            except ZeroDivisionError:  # happens when lap has no activity at all
                com_diffs.append(np.nan)
        self.com_diffs_lap_by_lap = np.array(com_diffs)

    def calculate_shift_scores(self):
        rate_matrix_active_laps = self.rate_matrix_pf[:,self.active_laps]

        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        #ub = ub if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        pf_bins_range = np.arange(0, ub - lb)

        if all(self.rate_matrix_pf[:,0] == 0):
            self.add_note("empty formation lap")
            return

        formation_bin = np.average(pf_bins_range, weights=self.rate_matrix_pf[:, 0])
        self.coms.append(formation_bin)
        for i in range(1, len(self.active_laps) - self.params["SHIFT_WINDOW"]):
            rate_matrix_window = rate_matrix_active_laps[:, i:i + self.params["SHIFT_WINDOW"]]

            # moving average of centers of mass (com)
            coms_window = np.zeros(self.params["SHIFT_WINDOW"])
            for i in range(self.params["SHIFT_WINDOW"]):
                rates_lap = rate_matrix_window[:,i]
                coms_window[i] = np.average(pf_bins_range, weights=rates_lap)
            avg_com_window = coms_window.mean()
            self.coms.append(avg_com_window)
            com_diff = avg_com_window - formation_bin
            self.com_diffs.append(com_diff)

    def calculate_initial_shift(self):
        self.initial_shift = np.mean(self.com_diffs[1:1+self.params["SPEARMAN_SKIP_LAPS"]])
        if len(self.com_diffs) >= 1+self.params["SPEARMAN_SKIP_LAPS"]:
            if self.initial_shift < 0:
                self.has_backwards_shift = True

        # initial shift after 5 laps
        coms_AL5 = self.coms[5:]
        if len(coms_AL5) == 0:
            return  # not enough active laps to calculate
        reference_bin = coms_AL5[0]
        coms_diff_AL5 = [coms_AL5[i]-reference_bin for i in range(len(coms_AL5))]
        self.initial_shift_AL5 = np.mean(coms_diff_AL5[1:1+self.params["SPEARMAN_SKIP_LAPS"]])

    def calculate_spearman(self):
        com_diffs_after_shift = self.com_diffs[self.params["SPEARMAN_SKIP_LAPS"]:]
        lap_indices_after_shift = np.arange(0, len(com_diffs_after_shift))
        spearman = scipy.stats.spearmanr(lap_indices_after_shift, com_diffs_after_shift)
        self.spearman_corrs = (spearman.correlation, spearman.pvalue)

    def shift_score_linear_fit(self):
        com_diffs_after_shift = self.com_diffs[self.params["SPEARMAN_SKIP_LAPS"]:]
        lap_indices_after_shift = np.arange(0, len(com_diffs_after_shift))
        self.linear_coeffs = np.polyfit(lap_indices_after_shift, com_diffs_after_shift, 1)

    def evaluate_drift(self):
        if len(self.com_diffs) == 0:
            self.add_note("empty shift score list")
        elif len(self.com_diffs) < self.params["SPEARMAN_SKIP_LAPS"] + 2:
            self.add_note(f"shift score list too short (len={len(self.com_diffs)})")
            return

        self.calculate_spearman()
        self.shift_score_linear_fit()

        r, p = self.spearman_corrs
        m, b = self.linear_coeffs
        shift_fit_formation_lap = m * (-self.params["SPEARMAN_SKIP_LAPS"]) + b  # extrapolation of linear fit on shift score to formation lap
        first_laps_avg = np.mean(self.com_diffs[:self.params["SPEARMAN_SKIP_LAPS"]+1])
        if r > 0 or p > 0.05 or shift_fit_formation_lap < 0:  # we don't care about forward drift here
            self.has_no_drift = True

    def prepare_for_nmf(self):
        lb, ub = self.bounds
        if len(self.i_laps) - self.formation_lap > self.params["NMF_MINIMUM_LAPS"] and lb > self.params["FORWARDS_BOUND_EXTENSION"]:
            lb = lb-self.params["FORWARDS_BOUND_EXTENSION"]  # in order to center PF we need to account for FBE and extend in the other direction too
        else:
            return
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)

        rate_matrix_pf_uncentered = self.rate_matrix[lb:ub, self.formation_lap:self.formation_lap+self.params["NMF_MINIMUM_LAPS"]]
        pf_bins_range = np.arange(0, ub - lb)
        coms_laps = np.zeros(self.params["NMF_MINIMUM_LAPS"])
        for i_lap in range(self.params["NMF_MINIMUM_LAPS"]):
            rates_lap = rate_matrix_pf_uncentered[:,i_lap]
            if all(rates_lap == 0):
                coms_laps[i_lap] = np.nan
            else:
                coms_laps[i_lap] = np.average(pf_bins_range, weights=rates_lap)
        avg_com = np.nanmean(coms_laps)
        try:
            center_bin = int(lb + np.round(avg_com))
            centered_lb = center_bin - 4*self.params["NMF_STD_PF_WIDTH"]
            centered_ub = center_bin + 4*self.params["NMF_STD_PF_WIDTH"]
        except ValueError:
            print(f"ERROR in preparing PF for NMF in {self.sessionID}, cellid={self.cellid},"
                  f"corridor={self.cor_index}, bounds=({lb},{ub}), fl,el=({self.formation_lap},{self.end_lap})")
            return

        if centered_lb <= 0 or centered_ub >= self.N_pos_bins:
            return

        self.rate_matrix_pf_centered = self.rate_matrix[centered_lb:centered_ub, self.formation_lap:self.formation_lap+self.params["NMF_MINIMUM_LAPS"]]

    def evaluate_quality(self):
        rate_matrix_active_laps = self.rate_matrix_pf[:, self.active_laps]

        lb, ub = self.bounds
        ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        #ub = ub if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
        pf_bins_range = np.arange(0, ub - lb)

        coms = []
        for i in range(len(self.active_laps)):
            rates_lap = rate_matrix_active_laps[:, i]
            com =  np.average(pf_bins_range, weights=rates_lap)
            coms.append(com)
        coms = np.array(coms)
        cv_coms = np.var(coms) / np.mean(coms)
        self.cv_coms = cv_coms
