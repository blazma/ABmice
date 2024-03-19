class TunedCell:
    def __init__(self, sessionID, cellid, rate_matrix, corridor, bins_p95_geq_afr,
                 frames_pos_bins, frames_dF_F, n_events, total_time):
        self.sessionID = sessionID
        self.cellid = cellid
        self.rate_matrix = rate_matrix
        self.corridor = corridor

        # list of spatial bins where tuning curve > 95th percentile of shuffled activity
        self.bins_p95_geq_afr = bins_p95_geq_afr

        # list of positions of animal (binned) at each frame with corresponding dF_F in that frame
        self.frames_pos_bins = frames_pos_bins
        self.frames_dF_F = frames_dF_F

        # number of Ca2+ events and total imaging time
        self.n_events = n_events
        self.total_time = total_time
