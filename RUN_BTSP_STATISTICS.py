import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib_venn import venn3, venn3_unweighted
import matplotlib
import seaborn as sns
#import cmasher
import scipy

#########################
area = "CA1"
#date_CA1 = "230911_fixed"
date_CA1 = "231030_calcSGforEarly"
date_CA3 = "230917_noshift"
#########################

# add font
#from matplotlib import font_manager
#font_dirs = ['C:\\home\\phd\\']
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#for font_file in font_files:
#    font_manager.fontManager.addfont(font_file)
#plt.rcParams['font.family'] = 'Trade Gothic Next LT Pro BdCn'

animals_CA1 = ["KS028",
               "KS029",
               "KS030",
               "srb131"]
animals_CA3 = ["srb231",
               "srb251",
               "srb269",
               "srb270"]

def load_data(area, date):
    # configure run
    #data_root = f"analyses/BTSP_analysis_{area}_{date}"
    data_root = f"BTSP_analysis_{area}_{date}"
    if area == "CA1":
        animals = animals_CA1
    elif area == "CA3":
        animals = animals_CA3
    else:
        raise Exception("choose CA1 or CA3")

    # read place fields and cell statistics dataframes for each animal
    place_fields_df = None
    cell_stats_df = None
    for animal in animals:
        try:
            place_fields_df_animal = pd.read_pickle(f"{data_root}/{animal}/{animal}_place_fields_df.pickle")
            cell_stats_df_animal = pd.read_pickle(f"{data_root}/{animal}/{animal}_cell_stats_df.pickle")
        except Exception:
            print(f"ERROR occured for animal {animal} during DF pickle loading; skipping")
            continue
        if cell_stats_df is None:
            cell_stats_df = cell_stats_df_animal
        else:
            cell_stats_df = pd.concat((cell_stats_df_animal, cell_stats_df))
        if place_fields_df is None:
            place_fields_df = place_fields_df_animal
        else:
            place_fields_df = pd.concat((place_fields_df_animal, place_fields_df))
    place_fields_df = place_fields_df.reset_index().drop("index", axis=1)
    cell_stats_df = cell_stats_df.reset_index().drop("index", axis=1)
    return place_fields_df, cell_stats_df

categories_order = {
    "unreliable": 0,
    "early": 1,
    "transient": 2,
    "non-btsp": 3,
    "btsp": 4
}
categories_colors_RGB = {
    "unreliable": "#a2a1a1",
    "early": "#00e3ff",
    "transient": "#ffe000",
    "non-btsp": "#ff000f",
    "btsp": "#be70ff"
}

date = date_CA1 if area == "CA1" else date_CA3
place_fields_df, cell_stats_df = load_data(area, date)
pfs_by_category = place_fields_df.groupby("category").count().sort_values(by="category", key=lambda x: x.map(categories_order))
pfs_by_category_and_animal = place_fields_df.groupby(["category", "session id"]).count().sort_values(by="category", key=lambda x: x.map(categories_order))

pfs_by_category_proportions = None
sessions = place_fields_df["session id"].unique()
for session in sessions:
    animalID, _, _ = session.partition("_")
    place_fields_session = place_fields_df.loc[place_fields_df["session id"] == session]
    place_fields_session_counts = place_fields_session.groupby("category").count()
    n_place_fields_session = len(place_fields_session)
    place_field_proportions_session = place_fields_session_counts / n_place_fields_session
    place_field_proportions_session = place_field_proportions_session["session id"].to_dict()  # select arbitrary column - they all contain the same values anyway
    pf_proportions_dict = {
        "animalID": animalID,
        "session id": session,
        "unreliable": 0 if "unreliable" not in place_field_proportions_session else place_field_proportions_session["unreliable"],
        "early": 0 if "early" not in place_field_proportions_session else place_field_proportions_session["early"],
        "transient": 0 if "transient" not in place_field_proportions_session else place_field_proportions_session["transient"],
        "non-btsp": 0 if "non-btsp" not in place_field_proportions_session else place_field_proportions_session["non-btsp"],
        "btsp": 0 if "btsp" not in place_field_proportions_session else place_field_proportions_session["btsp"]
    }
    if pfs_by_category_proportions is None:
        pfs_by_category_proportions = pd.DataFrame.from_dict([pf_proportions_dict])
    else:
        pfs_by_category_proportions = pd.concat((pfs_by_category_proportions, pd.DataFrame.from_dict([pf_proportions_dict])))
pfs_by_category_proportions = pfs_by_category_proportions.reset_index().drop("index", axis=1)

def plot_cells():
    fig, ax = plt.subplots(figsize=(2.5,4), dpi=125)
    sns.boxplot(data=cell_stats_df, ax=ax, width=0.8, palette="flare", showfliers=False)
    sns.swarmplot(data=cell_stats_df, ax=ax,x=1, y="total cells", hue="animalID", palette="Blues", alpha=1.0)
    sns.swarmplot(data=cell_stats_df, ax=ax,x=2, y="active cells", hue="animalID", palette="Blues", alpha=1.0, legend=False)
    sns.swarmplot(data=cell_stats_df, ax=ax,x=3, y="tuned cells", hue="animalID", palette="Blues", alpha=1.0, legend=False)
    ax.set_ylabel("# cells")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
    ax.legend(loc='upper right')
    plt.tight_layout()
    #plt.title(f"[{area}] Number of cells for each session")

def plot_place_fields(area):
    fig, axs = plt.subplots(1,2, figsize=(6,3), dpi=125)
    n_place_fields = place_fields_df.shape[0]

    pfs_by_category.plot(kind="bar", y="session id", ax=axs[0], color=categories_colors_RGB.values(), legend=False)
    pfs_by_category.plot(kind="pie", y="session id", ax=axs[1], colors=categories_colors_RGB.values(), legend=False, autopct=lambda pct: f'{np.round(pct,1)}%')
    axs[1].set_ylabel("")
    axs[0].set_ylabel("# place fields")
    axs[0].set_xlabel("")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    axs[0].set_ylim([0, 250])
    #axs[0].set_yticks(np.linspace(0,pfs_by_category["session id"].values.max(),num=5))
    axs[0].spines[['right', 'top']].set_visible(False)
    #plt.suptitle(f"[{area}] Distribution of place fields by PF categories across all animals")
    plt.tight_layout()

def plot_place_fields_by_animal(area):
    fig, ax = plt.subplots()
    pf_counts = pfs_by_category_and_animal.reset_index()[["category", "session id", "cell id"]]
    sns.boxplot(data=pf_counts, x="category", y="cell id", ax=ax, palette=categories_colors_RGB, showfliers=False)
    sns.swarmplot(data=pf_counts, x="category", y="cell id", ax=ax, color="black", alpha=0.5)
    plt.title(f"[{area}] Number of various place fields for each session")
    ax.set_ylabel("# place fields")
    ax.set_xlabel("")

def plot_place_field_proportions_by_animal(area):
    fig, ax = plt.subplots(figsize=(6,3), dpi=125)
    sns.boxplot(data=pfs_by_category_proportions, ax=ax, palette=categories_colors_RGB, showfliers=False)
    sns.swarmplot(data=pfs_by_category_proportions, ax=ax, color="black", alpha=0.5)
    #plt.title(f"[{area}] Proportion of various place field categories by session")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim([0,1])
    ax.set_ylabel("proportion")
    ax.set_xlabel("")

def plot_place_fields_criteria(area):
    fig, ax = plt.subplots()
    plt.title(f"[{area}] Number of place fields with a given BTSP criterion satisfied")
    nonbtsp_pfs_w_criteria_df = place_fields_df[place_fields_df["category"] == "non-btsp"][["has high gain", "has backwards shift", "has no drift"]].apply(pd.value_counts).transpose()
    nonbtsp_pfs_w_criteria_df.plot(kind="bar", stacked=True, color=["r", "g"], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

def plot_place_field_properties(area):
    fig, ax = plt.subplots()
    place_fields_df["pf width"] = pd.to_numeric(place_fields_df["upper bound"] - place_fields_df["lower bound"])
    pf_width_means = place_fields_df.groupby("category")["pf width"].mean()
    pf_width_stds = place_fields_df.groupby("category")["pf width"].std()
    sns.violinplot(data=place_fields_df[["category", "pf width"]], x="category", y="pf width", ax=ax, palette=categories_colors_RGB)
    for i_category, category in enumerate(categories_order.keys()):
        x = i_category+0.1
        y = pf_width_means[category] + 5*pf_width_stds[category]
        text = f"mean: {np.round(pf_width_means[category],1)}\nstd: {np.round(pf_width_stds[category],1)}"
        plt.annotate(text, (x, y))
    pass

def plot_place_field_heatmap(area):
    n_bins = place_fields_df["upper bound"].max() - place_fields_df["lower bound"].min()  # 75 bins
    n_laps = place_fields_df["end lap"].max() - place_fields_df["formation lap"].min()  # 225 laps, ez tuti túl sok biztos benne vannak 3 korridorosok

    earlies = np.zeros((n_laps, n_bins))
    transients = np.zeros((n_laps, n_bins))
    nonbtsps = np.zeros((n_laps, n_bins))
    btsps = np.zeros((n_laps, n_bins))

    fig, axs = plt.subplots(5, 4, sharex=True)

    # TODO: these are guesses based on plots, needs to be exact
    reward_zones = {
        0: [38, 46],  # corridor 14
        1: [61, 69]   # corridor 15
    }

    categories = ["early", "transient", "non-btsp", "btsp"]
    colormaps = ["Blues", "Oranges", "Reds", "Purples"]
    for i_corridor in range(2):
        i_ax = 2 * i_corridor

        print(f"corridor: {i_corridor}")
        RZ_start, RZ_end = reward_zones[i_corridor]
        place_fields_corridor = place_fields_df[((place_fields_df["category"] == "early") |
                                                 (place_fields_df["category"] == "transient") |
                                                 (place_fields_df["category"] == "non-btsp") |
                                                 (place_fields_df["category"] == "btsp")) &
                                                 (place_fields_df["corridor"] == 0)]
        N_place_fields_corridor = place_fields_corridor['session id'].count()
        for i_category, category in enumerate(categories):

            full_span = None
            if category == "early":
                full_span = earlies
            elif category == "transient":
                full_span = transients
            elif category == "non-btsp":
                full_span = nonbtsps
            elif category == "btsp":
                full_span = btsps

            ax = axs[i_category, i_ax]
            colormap = colormaps[i_category]
            pfs = place_fields_df.loc[(place_fields_df["category"] == category) & (place_fields_df["corridor"] == i_corridor)]
            print(category, 1 / len(pfs.index))
            for _, pf in pfs.iterrows():
                lb = pf["lower bound"]
                ub = pf["upper bound"]
                fl = pf["formation lap"]
                el = pf["end lap"]

                pf_span = np.ones((el-fl, ub-lb))
                full_span[fl:el,lb:ub] += pf_span
            full_span = 100 * full_span / N_place_fields_corridor
            ax.axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
            full_span = full_span[:40,:]
            im = ax.imshow(full_span, cmap=colormap, origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax)

            ax = axs[i_category, i_ax+1]
            full_span_marginal = np.sum(full_span, axis=0)
            color = colormap.lower()[:-1]
            ax.bar(np.arange(len(full_span_marginal)), full_span_marginal, width=1.0, color=color)
            ax.axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
            ax.set_ylim([0, 1.1*full_span_marginal.max()])

        ax = axs[4, i_ax]
        btsp_nonbtsp_proportion = np.divide(btsps[:40,:], nonbtsps[:40,:])
        #ax.axvspan(RZ_start, RZ_end, color="green", alpha=0.1)
        ax.set_ylim([2, 40])
        im = ax.imshow(btsp_nonbtsp_proportion, cmap="binary", origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax)
    plt.show()

def plot_place_fields_criteria_venn_diagram(area, with_btsp=False):
    if with_btsp:
        place_fields_filtered = place_fields_df[(place_fields_df["category"] == "non-btsp") | (place_fields_df["category"] == "btsp")]
    else:
        place_fields_filtered = place_fields_df[place_fields_df["category"] == "non-btsp"]
    criteria = ["has high gain", "has backwards shift", "has no drift"]
    criteria_counts = place_fields_filtered.groupby(criteria).count()["session id"].reset_index()

    def select_row(c1, is_c1, c2, is_c2, c3, is_c3):
        cond1 = criteria_counts[c1] if is_c1 else ~criteria_counts[c1]
        cond2 = criteria_counts[c2] if is_c2 else ~criteria_counts[c2]
        cond3 = criteria_counts[c3] if is_c3 else ~criteria_counts[c3]
        try:
            return criteria_counts[(cond1) & (cond2) & (cond3)]["session id"].iloc[0] / criteria_counts["session id"].sum()
        except IndexError:  # no element found means no such combo existed
            return 0

    gain, shift, drift = criteria
    subsets = (
        select_row(gain, True, shift, False, drift, False),  # Set 1
        select_row(gain, False, shift, True, drift, False),  # Set 2
        select_row(gain, True, shift, True, drift, False),  # Set 1n2
        select_row(gain, False, shift, False, drift, True),  # Set 3
        select_row(gain, True, shift, False, drift, True),  # Set 1n3
        select_row(gain, False, shift, True, drift, True),  # Set 2n3
        select_row(gain, True, shift, True, drift, True) if with_btsp else 0, # Set 1n2n3
    )
    plt.figure(figsize=(5,3), dpi=100)
    none_satisfied = select_row(gain, False, shift, False, drift, False)
    plt.text(0.6, -0.3, f"none: {np.round(100*none_satisfied,2)}%")
    v = venn3_unweighted(subsets, set_labels=criteria, subset_label_formatter=lambda label: f"{np.round(100*label,2)}%")

    norm = matplotlib.colors.Normalize(vmin=min(subsets), vmax=max(subsets), clip=True)
    cmap = cmasher.get_sub_cmap('Reds', 0.1, 0.6)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    patch_ids = ['100', '010', '110', '001', '101', '011', '111']
    for i_patch_id, patch_id in enumerate(patch_ids):
        patch = v.get_patch_by_id(patch_id)
        patch.set_color(mapper.to_rgba(subsets[i_patch_id]))

def shift_gain_correlations():
    place_fields_df_CA1, _ = load_data("CA1", date_CA1)
    cbtsp_df_CA1 = place_fields_df_CA1[(place_fields_df_CA1["category"] == "non-btsp") | (place_fields_df_CA1["category"] == "btsp")]
    shift_gain_df_CA1 = cbtsp_df_CA1[["initial shift", "formation gain"]].reset_index(drop=True)
    shift_gain_df_CA1 = shift_gain_df_CA1[(shift_gain_df_CA1["initial shift"].notna()) & (shift_gain_df_CA1["formation gain"].notna())]
    #shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"] < 2.5) & (shift_gain_df["initial shift"] > -5)]

    place_fields_df_CA3, _ = load_data("CA3", date_CA3)
    cbtsp_df_CA3 = place_fields_df_CA3[(place_fields_df_CA3["category"] == "non-btsp") | (place_fields_df_CA3["category"] == "btsp")]
    shift_gain_df_CA3 = cbtsp_df_CA3[["initial shift", "formation gain"]].reset_index(drop=True)
    shift_gain_df_CA3 = shift_gain_df_CA3[(shift_gain_df_CA3["initial shift"].notna()) & (shift_gain_df_CA3["formation gain"].notna())]
    #shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"] < 2.5) & (shift_gain_df["initial shift"] > -5)]

    #N_pfs_CA3 = shift_gain_df_CA3.shape[0]
    #shift_gain_df_CA1 = shift_gain_df_CA1.sample(n=N_pfs_CA3)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(shift_gain_df_CA1["initial shift"].values, shift_gain_df_CA1["formation gain"].values)
    ax = sns.regplot(data=shift_gain_df_CA1, x="initial shift", y="formation gain", scatter_kws={'alpha':0.1},
                line_kws={'label':"y={0:.2f}x+{1:.2f}, r={2:.2f}, p={3:.2f}".format(slope,intercept,r_value,p_value)})
    ax.set(yscale='log')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(shift_gain_df_CA3["initial shift"].values, shift_gain_df_CA3["formation gain"].values)
    ax = sns.regplot(data=shift_gain_df_CA3, x="initial shift", y="formation gain", scatter_kws={'alpha':0.1},
                line_kws={'label':"y={0:.2f}x+{1:.2f}, r={2:.2f}, p={3:.2f}".format(slope,intercept,r_value,p_value)})
    ax.set(yscale='log')
    plt.legend()

def plot_shift_gain_distribution(plot_all=True, logy=False, zoom=False):
    place_fields_df, _ = load_data("CA1", date_CA1)
    place_fields_df["newly formed"] = np.where(place_fields_df["category"].isin(["transient", "non-btsp", "btsp"]), True, False)
    shift_gain_df = place_fields_df[["newly formed", "initial shift", "formation gain"]].reset_index(drop=True)
    shift_gain_df = shift_gain_df[(shift_gain_df["initial shift"].notna()) & (shift_gain_df["formation gain"].notna())]

    if plot_all:
        g = sns.jointplot(data=shift_gain_df.reset_index(drop=True), x="initial shift", y="formation gain",
                      hue="newly formed", alpha=1, s=3, marginal_kws={"common_norm": False})
    else: # plot newly formed only
        shift_gain_df_CA1 = shift_gain_df[shift_gain_df["newly formed"] == True]
        g = sns.jointplot(data=shift_gain_df_CA1.reset_index(drop=True), x="initial shift", y="formation gain",
                      alpha=1, s=3, marginal_kws={"common_norm": False}, color="orange")

    # log y scale
    if logy:
        g.ax_joint.set_yscale("log")
        if zoom:
            plt.ylim([0.1,10])

    # regular y scale:
    if zoom:
        plt.ylim([0,10])
    else:
        plt.ylim(bottom=0)

    plt.xlim([-15, 15])
    plt.axvline(x=0, c="k", linestyle="--")
    plt.axhline(y=1, c="k", linestyle="--")
    plt.show()

    # subsampling
    shift_diffs = []
    gain_diffs = []
    for seed in range(1000):
        shift_gain_df_subs = shift_gain_df.sample(n=274, random_state=seed)

        mean_shift_newlyf = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == True]['initial shift'].mean()
        mean_shift_stable = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == False]['initial shift'].mean()
        shift_diff = mean_shift_newlyf - mean_shift_stable
        shift_diffs.append(shift_diff)

        mean_gain_newlyf = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == True]['formation gain'].mean()
        mean_gain_stable = shift_gain_df_subs[shift_gain_df_subs['newly formed'] == False]['formation gain'].mean()
        gain_diff = mean_gain_newlyf - mean_gain_stable
        gain_diffs.append(gain_diff)
    fig, axs = plt.subplots(2,1)
    sns.kdeplot(data=shift_diffs, ax=axs[0], color="k")
    sns.kdeplot(data=gain_diffs, ax=axs[1], color="k")
    axs[0].set_title("shift")
    axs[1].set_title("gain")

    # plot CA3 values on distributions
    shift_diff_CA3 = -0.091
    gain_diff_CA3 = 0.252
    axs[0].axvline(shift_diff_CA3, c="red", label="CA3")
    axs[1].axvline(gain_diff_CA3, c="red", label="CA3")

    # plot confidence intervals
    axs[0].axvline(np.percentile(shift_diffs, q=95), c="blue", label="CA1 95th p.")
    axs[0].axvline(np.percentile(shift_diffs, q=99), c="cyan", label="CA1 99th p.")
    axs[1].axvline(np.percentile(gain_diffs, q=5), c="blue", label="CA1 5th p.")
    axs[1].axvline(np.percentile(gain_diffs, q=1), c="cyan", label="CA1 1st p.")

    # add legends
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def no_shift_criterion_plot():
    def get_pf_counts(df):
        return df[(df["category"] == "non-btsp")|(df["category"] == "btsp")].groupby("category").count()["session id"]

    yes_shift_CA1 = "230911_fixed"
    yes_shift_CA3 = "230911_fixed"
    no_shift_CA1 = "230918_withFormationBin_noshift"
    no_shift_CA3 = "230917_noshift"

    pfs_Yshift_CA1, _ = load_data("CA1", yes_shift_CA1)
    pfs_Yshift_CA3, _ = load_data("CA3", yes_shift_CA3)
    pfs_Nshift_CA1, _ = load_data("CA1", no_shift_CA1)
    pfs_Nshift_CA3, _ = load_data("CA3", no_shift_CA3)

    counts_Nshift_CA1 = get_pf_counts(pfs_Nshift_CA1)
    counts_Nshift_CA3 = get_pf_counts(pfs_Nshift_CA3)
    counts_Yshift_CA1 = get_pf_counts(pfs_Yshift_CA1)
    counts_Yshift_CA3 = get_pf_counts(pfs_Yshift_CA3)

    counts_CA1 = pd.DataFrame({
        "with shift": counts_Yshift_CA1 / counts_Yshift_CA1.sum(),
        "without shift": counts_Nshift_CA1 / counts_Yshift_CA1.sum(),
    })
    counts_CA3 = pd.DataFrame({
        "with shift": counts_Yshift_CA3 / counts_Yshift_CA3.sum(),
        "without shift": counts_Nshift_CA3 / counts_Yshift_CA3.sum(),
    })

    factor = 0.8
    fig, ax = plt.subplots(figsize=(2*factor,4*factor), dpi=120)
    ax = counts_CA1.T.plot.bar(stacked=True, rot=45, color=["#be70ff", "#ff000f"], width=0.9, legend=False, ax=ax)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(2*factor,4*factor), dpi=120)
    ax = counts_CA3.T.plot.bar(stacked=True, rot=45, color=["#be70ff", "#ff000f"], width=0.9, legend=False, ax=ax)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()


#plot_cells()
#plot_place_fields(area)
#plot_place_fields_by_animal(area)
#plot_place_field_proportions_by_animal(area)
#plot_place_field_properties(area)
#plot_place_fields_criteria(area)
#plot_place_field_heatmap(area)
#plot_place_fields_criteria_venn_diagram(area, with_btsp=True)
#shift_gain_correlations()
#no_shift_criterion_plot()

#plot_shift_gain_distribution(plot_all=True, zoom=False, logy=False)
plot_shift_gain_distribution(plot_all=True, zoom=True, logy=True)
#plot_shift_gain_distribution(plot_all=False, zoom=False, logy=False)
#plot_shift_gain_distribution(plot_all=False, zoom=True, logy=True)

plt.show()