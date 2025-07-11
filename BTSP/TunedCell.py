class TunedCell:
    def __init__(self, sessionID, cellid, rate_matrix, corridor, history, bins_p95_geq_afr,
                 lap_histories, frames_pos_bins, frames_dF_F, n_events, total_time, spike_matrix, corridor_laps):
        self.sessionID = sessionID
        self.cellid = cellid
        self.rate_matrix = rate_matrix
        self.corridor = corridor
        self.history = history  # history condition - can only take 3 kinds of values: ALL, STAY, CHANGE

        # only applies to ALL history condition:
        self.lap_histories = lap_histories  # list of STAY / CHANGE values, each belonging to a lap

        # list of spatial bins where tuning curve > 95th percentile of shuffled activity
        self.bins_p95_geq_afr = bins_p95_geq_afr

        # list of positions of animal (binned) at each frame with corresponding dF_F in that frame
        self.frames_pos_bins = frames_pos_bins
        self.frames_dF_F = frames_dF_F

        # number of Ca2+ events and total imaging time
        self.n_events = n_events
        self.total_time = total_time

        # spike matrix - for Barna's modeling
        self.spike_matrix = spike_matrix

        # list of lap indices of the current corridor
        self.corridor_laps = corridor_laps


class Cell:
    def __init__(self, sessionID, cellid, history, n_events, total_time):
        self.sessionID = sessionID
        self.cellid = cellid
        self.history = history  # history condition - can only take 3 kinds of values: ALL, STAY, CHANGE

        # number of Ca2+ events and total imaging time
        self.n_events = n_events
        self.total_time = total_time
