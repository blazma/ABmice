import numpy as np
from utils import grow_df
from BTSP.PlaceField import PlaceField


class BtspAnalysisSingleCell:
    def __init__(self, sessionID, cellid, rate_matrix, frames_pos_bins, frames_dF_F, params=None, corridors=None,
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
        self.params = params

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

        self.frames_pos_bins = frames_pos_bins
        self.frames_dF_F = frames_dF_F

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
        candidate_pfs_filtered = list(filter(lambda cpf: len(cpf) > self.params["FORMATION_CONSECUTIVE_BINS"], candidate_pfs))
        place_field_bounds = [(cpf[0], cpf[-1]) for cpf in candidate_pfs_filtered]
        return place_field_bounds

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
                place_field.params = self.params

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
                if len(place_field.i_laps) > self.params["FORMATION_GAIN_WINDOW"] and len(place_field.active_laps) > self.params["FORMATION_GAIN_WINDOW"]:
                #if len(place_field.i_laps) > PF.FORMATION_GAIN_WINDOW and len(place_field.active_laps) > PF.SHIFT_WINDOW:
                    place_field.calculate_formation_gain()
                    place_field.calculate_shift()
                    place_field.evaluate_quality()

                #place_field.calculate_dF_F_maxima(self.frames_pos_bins, self.frames_dF_F)
                #place_field.calculate_lap_by_lap_COM_diffs()

                # filter place fields that form too early to assess novelty
                if place_field.formation_lap_uncorrected < 2:
                    place_field.category = "early"
                    self.early_place_fields.append(place_field)
                    continue

                ### filter out newly formed PFs who are born relatively early:
                #if place_field.formation_lap_uncorrected < 10:
                #    continue

                # filter place fields that are too short to meaningfully assess drift
                if len(place_field.active_laps) <= self.params["SHIFT_WINDOW"] + self.params["SPEARMAN_SKIP_LAPS"] + 2:  # +2 points are at least necessary to calculate spearman correlation (else nan)
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
        place_field_categories = [
            self.unreliable_place_fields,
            self.early_place_fields,
            self.transient_place_fields,
            self.nonbtsp_novel_place_fields,
            self.btsp_place_fields
        ]
        place_fields_df = None
        for place_field_category in place_field_categories:
            for place_field in place_field_category:
                place_fields_df = grow_df(place_fields_df, place_field.to_dataframe())
        return place_fields_df

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
