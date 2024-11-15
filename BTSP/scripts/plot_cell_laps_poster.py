def plot_cell_laps(self, cellid, corridor=-1):
    if (corridor == -1):
        corridor = self.corridors
    else:
        if corridor in self.corridors:
            corridor = np.array([corridor])
        else:
            print('Warning: specified corridor does not exist in this session!')
            return

    is_cell_with_unreliable_pf = cellid in [pf.cellid for pf in self.btsp_analysis.unreliable_place_fields]
    is_cell_with_early_pf = cellid in [pf.cellid for pf in self.btsp_analysis.early_place_fields]
    is_cell_with_transient_pf = cellid in [pf.cellid for pf in self.btsp_analysis.transient_place_fields]
    is_cell_with_nonbtsp_novel_pf = cellid in [pf.cellid for pf in
                                               self.btsp_analysis.nonbtsp_novel_place_fields]
    is_cell_with_btsp_pf = cellid in [pf.cellid for pf in self.btsp_analysis.btsp_place_fields]

    fig, ax = plt.subplots(4, corridor.size, squeeze=False, sharex=True, sharey='row',
                           figsize=(6 * corridor.size, 10))
    nbins = self.activity_tensor.shape[0]
    for cor_index in range(corridor.size):
        if corridor.size == 1:
            corridor_to_plot = corridor
        else:
            corridor_to_plot = corridor[cor_index]
        cell_info = self.CreateTitle(corridor_to_plot, cellid)

        # select the laps in the corridor (these are different indexes from upper ones!)
        # only laps with imaging data are selected - this will index the activity_tensor
        i_laps = np.nonzero(self.i_corridors[self.i_Laps_ImData] == corridor_to_plot)[0]

        # calculate rate matrix
        total_spikes = self.activity_tensor[:, cellid, i_laps]
        total_time = self.activity_tensor_time[:, i_laps]
        rate_matrix = nan_divide(total_spikes, total_time, where=total_time > 0.025)

        # calculate average rates for plotting
        average_firing_rate = np.nansum(rate_matrix, axis=1) / i_laps.size
        std = np.nanstd(rate_matrix, axis=1) / np.sqrt(i_laps.size)
        errorbar_x = np.arange(self.N_pos_bins)

        # plotting
        title_string = 'ratemap of cell ' + str(cellid) + ' in corridor ' + str(corridor_to_plot) + cell_info
        ax[0, cor_index].set_title(title_string)
        ax[1, cor_index].fill_between(errorbar_x, average_firing_rate + std, average_firing_rate - std, alpha=0.3)
        ax[1, cor_index].plot(average_firing_rate, zorder=0)

        if is_cell_with_unreliable_pf:
            unreliable_place_fields = [pf for pf in self.btsp_analysis.unreliable_place_fields if
                                       pf.cellid == cellid and pf.cor_index == cor_index]
            for unreliable_field in unreliable_place_fields:
                # plot place field bounds (spatial bins)
                lb, ub = unreliable_field.bounds
                ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)

                ax[0, cor_index].axvspan(lb, ub, color="black", alpha=0.15)
                ax[1, cor_index].axvspan(lb, ub, color="black", alpha=0.15)

        if is_cell_with_early_pf:
            early_place_fields = [pf for pf in self.btsp_analysis.early_place_fields if
                                  pf.cellid == cellid and pf.cor_index == cor_index]
            for early_field in early_place_fields:
                # plot place field bounds (spatial bins)
                lb, ub = early_field.bounds
                ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                fl_uc = early_field.formation_lap_uncorrected / len(i_laps)
                fl = early_field.formation_lap / len(i_laps)
                el = (early_field.end_lap + 1) / len(i_laps)

                # plot place field bounds (spatial bins)
                ax[0, cor_index].axvspan(lb, ub, ymin=fl_uc, ymax=el, color="cyan", alpha=0.15)
                ax[1, cor_index].axvspan(lb, ub, color="cyan", alpha=0.15)

                # plot max rates in each lap between place field boundaries
                rate_matrix_pf = early_field.rate_matrix[lb:ub, :]
                max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                active_laps = np.where(max_rates_pf > PF.FORMATION_THRESHOLD)[0].size  # laps where max activity exceeds 10% of max activity along all laps in pf

                # plot uncorrected formation lap
                ax[0, cor_index].plot(lb, early_field.formation_lap_uncorrected, marker=5, markersize=5, markeredgecolor="cyan", markerfacecolor="None", linestyle='None')
                ax[0, cor_index].plot(ub, early_field.formation_lap_uncorrected, marker=4, markersize=5, markeredgecolor="cyan", markerfacecolor="None", linestyle='None')

                # plot formation lap
                ax[0, cor_index].plot(lb, early_field.formation_lap, marker=5, markersize=5, color="cyan", linestyle='None')
                ax[0, cor_index].plot(ub, early_field.formation_lap, marker=4, markersize=5, color="cyan", linestyle='None')

                # plot end lap
                ax[0, cor_index].plot(lb, early_field.end_lap, marker=5, markersize=5, color="cyan", linestyle='None')
                ax[0, cor_index].plot(ub, early_field.end_lap, marker=4, markersize=5, color="cyan", linestyle='None')

        if is_cell_with_transient_pf:
            transient_fields = [pf for pf in self.btsp_analysis.transient_place_fields if
                                pf.cellid == cellid and pf.cor_index == cor_index]
            for transient_field in transient_fields:
                lb, ub = transient_field.bounds
                ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                fl_uc = transient_field.formation_lap_uncorrected / len(i_laps)
                fl = transient_field.formation_lap / len(i_laps)
                el = (transient_field.end_lap + 1) / len(i_laps)

                # plot place field bounds (spatial bins)
                ax[0, cor_index].axvspan(lb, ub, ymin=fl, ymax=el, color="orange", alpha=0.15)
                ax[1, cor_index].axvspan(lb, ub, color="orange", alpha=0.15)

                # plot max rates in each lap between place field boundaries
                rate_matrix_pf = transient_field.rate_matrix[lb:ub, :]
                max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                active_laps = np.where(max_rates_pf > PF.FORMATION_THRESHOLD)[0].size  # laps where max activity exceeds 10% of max activity along all laps in pf

                # plot uncorrected formation lap
                ax[0, cor_index].plot(lb, transient_field.formation_lap_uncorrected, marker=5, markersize=5, markeredgecolor="orange", markerfacecolor="None", linestyle='None')
                ax[0, cor_index].plot(ub, transient_field.formation_lap_uncorrected, marker=4, markersize=5, markeredgecolor="orange", markerfacecolor="None", linestyle='None')

                # plot formation lap
                ax[0, cor_index].plot(lb, transient_field.formation_lap, marker=5, markersize=5, color="orange", linestyle='None')
                ax[0, cor_index].plot(ub, transient_field.formation_lap, marker=4, markersize=5, color="orange", linestyle='None')

                # plot end lap
                ax[0, cor_index].plot(lb, transient_field.end_lap, marker=5, markersize=5, color="orange", linestyle='None')
                ax[0, cor_index].plot(ub, transient_field.end_lap, marker=4, markersize=5, color="orange", linestyle='None')

        if is_cell_with_nonbtsp_novel_pf or is_cell_with_btsp_pf:
            # plot 95th percentile from shuffling over each bin to average firing plot
            place_cells = np.union1d(self.tuned_cells[0],
                                     self.tuned_cells[1])  # these are the cells from shuffle
            i_cell = np.where(place_cells == cellid)[0]
            p95_cell = self.p95[cor_index][:, i_cell]
            ax[1, cor_index].plot(p95_cell, "r")

            # select place fields belonging to the current cell, current corridor
            place_fields_cell = list(filter(lambda pf: pf.cellid == cellid and pf.cor_index == cor_index,
                                            self.btsp_analysis.candidate_btsp_place_fields))
            for place_field in place_fields_cell:
                lb, ub = place_field.bounds
                ub = ub + 1 if ub != self.N_pos_bins else ub  # ub+1 because we want to include ub (closed set from right too)
                fl_uc = place_field.formation_lap_uncorrected / len(i_laps)
                fl = place_field.formation_lap / len(i_laps)
                el = (place_field.end_lap + 1) / len(i_laps)

                # plot place field bounds
                criteria_missed = []
                color = "darkviolet"
                dark_color = "purple"
                if not place_field.has_high_gain:
                    color = "red"
                    dark_color = "crimson"
                    criteria_missed.append("low gain")
                if not place_field.has_no_drift:
                    color = "red"
                    dark_color = "crimson"
                    criteria_missed.append("drifting")
                if not place_field.has_backwards_shift:
                    color = "red"
                    dark_color = "crimson"
                    criteria_missed.append("no backshift")
                ax[0, cor_index].axvspan(lb, ub, ymin=fl, ymax=el, color=color, alpha=0.15)
                ax[1, cor_index].axvspan(lb, ub, color=color, alpha=0.15)

                # plot max rates in each lap between place field boundaries
                rate_matrix_pf = place_field.rate_matrix[lb:ub, :]
                max_rates_pf = np.nanmax(rate_matrix_pf, axis=0) / np.nanmax(rate_matrix_pf)
                label = f"[{lb}-{ub - 1}]"
                for criterion_missed in criteria_missed:
                    label = f"{label}, {criterion_missed}"

                # plot uncorrected formation lap
                ax[0, cor_index].plot(lb, place_field.formation_lap_uncorrected, marker=5, markersize=5, markeredgecolor=color, markerfacecolor="None", linestyle='None')
                ax[0, cor_index].plot(ub, place_field.formation_lap_uncorrected, marker=4, markersize=5, markeredgecolor=color, markerfacecolor="None", linestyle='None')

                # plot formation lap
                ax[0, cor_index].plot(lb, place_field.formation_lap, marker=5, markersize=5, color=color, linestyle='None')
                ax[0, cor_index].plot(ub, place_field.formation_lap, marker=4, markersize=5, color=color, linestyle='None')

                # plot end lap
                ax[0, cor_index].plot(lb, place_field.end_lap, marker=5, markersize=5, color=color, linestyle='None')
                ax[0, cor_index].plot(ub, place_field.end_lap, marker=4, markersize=5, color=color, linestyle='None')

        n_laps = rate_matrix.shape[1]

        if corridor.size > 1:
            im1 = ax[0, cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower', cmap='binary')
        else:
            im1 = ax[0, cor_index].imshow(np.transpose(rate_matrix), aspect='auto', origin='lower', cmap='binary')

        plt.colorbar(im1, orientation='horizontal', ax=ax[1, cor_index])
        ax[0, cor_index].set_xlim(0, nbins + 1)
        ax[1, cor_index].set_xlim(0, nbins + 1)
        ax[0, cor_index].set_facecolor(matcols.CSS4_COLORS['palegreen'])

    ## add reward zones - rewardZones
    for cor_index in range(corridor.size):
        zone_starts = self.corridor_list.corridors[corridor[cor_index]].reward_zone_starts
        if (len(zone_starts) > 0):
            zone_ends = self.corridor_list.corridors[corridor[cor_index]].reward_zone_ends
            bottom, top = ax[0, cor_index].get_ylim()
            for i_zone in range(len(zone_starts)):
                left = zone_starts[i_zone] * nbins
                right = zone_ends[i_zone] * nbins
                polygon = Polygon(np.array([[left, bottom], [left, top], [right, top], [right, bottom]]), True,
                                  color='green', alpha=0.15)
                ax[0, cor_index].add_patch(polygon)
                # print('adding reward zone to the ', cor_index, 'th corridor, ', self.corridors[cor_index+1])

    fig.tight_layout()
    filename = f"C:/home/phd/btsp_poster/pf_examples/{self.sessionID}_cell{cellid}.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()
