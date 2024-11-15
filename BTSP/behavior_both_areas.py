import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from BehavioralStatistics import BehaviorStatistics
import os
import shutil
import scipy
from sklearn.decomposition import PCA
from utils import grow_df, makedir_if_needed
from constants import CATEGORIES, ANIMALS, VSEL_NORMALIZATION

data_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual"
output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\misc"

behav_stats_CA1 = BehaviorStatistics("CA1", data_path, output_path, extra_info="")
behav_stats_CA1.calc_behavior_score()
behav_stats_CA3 = BehaviorStatistics("CA3", data_path, output_path, extra_info="")
behav_stats_CA3.calc_behavior_score()

# calc behavior scores -- pool animals from both areas
y_cols = ['area', 'animalID', 'sessionID', 'P-correct (14)', 'P-correct (15)', 'Speed index (14)', 'Speed index (15)',
          'Speed selectivity', 'Lick index (14)', 'Lick index (15)', 'Lick selectivity', 'behavior score']
#y_cols = ['area', 'animalID', 'sessionID', 'Pcorrect(14)', 'Pcorrect(15)', 'Vsel(14)', 'Vsel(15)',
#          'Vsel(X-corr)', 'Lsel(14)', 'Lsel(15)', 'Lsel(X-corr)', 'behavior score']
df_CA1 = behav_stats_CA1.behavior_df[y_cols].set_index(['area', 'animalID', 'sessionID'])
df_CA3 = behav_stats_CA3.behavior_df[y_cols].set_index(['area', 'animalID', 'sessionID'])
scores_df = pd.concat([df_CA1, df_CA3])

def plot_behavior_scores():
    makedir_if_needed(f"{output_path}/statistics/behavior_bothAreas")

    plt.figure()
    sns.histplot(data=scores_df.reset_index()[["area", "behavior score"]].melt(id_vars="area"), x="value", hue="area", bins=7, binrange=[1,8], element="step")
    plt.savefig(f"{output_path}/statistics/behavior_bothAreas/behavior_scores_by_area.pdf")
    plt.close()

    plt.figure()
    sns.histplot(data=scores_df.reset_index()[["area", "behavior score"]].melt(id_vars="area"), x="value", bins=7, binrange=[1,8], element="step", color="k")
    plt.savefig(f"{output_path}/statistics/behavior_bothAreas/behavior_scores.pdf")
    plt.close()

    fig, axs = plt.subplots(2,1)
    scores = scores_df.reset_index()
    scores_CA1 = scores[scores["area"] == "CA1"]
    axs[0] = sns.histplot(data=scores_CA1[["animalID", "behavior score"]].melt(id_vars="animalID"), x="value", bins=7,
                 binrange=[1,8], element="step", color="k", ax=axs[0], hue="animalID", multiple="stack")
    sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
    scores_CA3 = scores[scores["area"] == "CA3"]
    axs[1] = sns.histplot(data=scores_CA3[["animalID", "behavior score"]].melt(id_vars="animalID"), x="value", bins=7,
                 binrange=[1,8], element="step", color="k", ax=axs[1], hue="animalID", multiple="stack")
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{output_path}/statistics/behavior_bothAreas/behavior_scores_by_area_and_session.pdf")
    plt.close()

def save_behavior_scores():
    # save scores to excel
    scores_df.reset_index().sort_values(by="behavior score").to_excel(f"{output_path}/statistics/behavior_bothAreas/behavior_scores.xlsx")

def sort_behavior_plots_by_score():
    # order lap-by-lap plots by behavior score
    src_dir = f"{output_path}/statistics/behavior_bothAreas/lap_by_lap_unsorted"
    dst_dir = f"{output_path}/statistics/behavior_bothAreas/lap_by_lap_sorted"
    for _, [sessionID, score] in scores_df.reset_index()[["sessionID", "behavior score"]].iterrows():
        src_file = f"{src_dir}/{sessionID}_lap_by_lap.pdf"
        dst_file = f"{dst_dir}/{np.round(score,3)}_{sessionID}_lap_by_lap.pdf"
        shutil.copyfile(src_file, dst_file)

def plot_place_fields_by_score():
    # load place fields
    pfs_df = None
    for animal in behav_stats_CA1.animals:
        pfs_df_animal = pd.read_pickle(f"{data_path}/place_fields/CA1{behav_stats_CA1.extra_info}/{animal}_place_fields_df.pickle")
        pfs_df = grow_df(pfs_df, pfs_df_animal)
    for animal in behav_stats_CA3.animals:
        pfs_df_animal = pd.read_pickle(f"{data_path}/place_fields/CA3{behav_stats_CA3.extra_info}/{animal}_place_fields_df.pickle")
        pfs_df = grow_df(pfs_df, pfs_df_animal)

    scores_df_sorted = scores_df.reset_index()#.sort_values(by="behavior score")
    pfs_scores_df = None
    for i_row, row in scores_df_sorted.iterrows():
        pfs_sess = pfs_df[pfs_df["session id"] == row["sessionID"]]
        if len(pfs_sess) == 0:
            continue
        pfs_counts = pfs_sess.groupby("category").count()["animal id"]
        pfs_scores_df_dict = {
            "area": row["area"],
            "animalID": row["sessionID"].partition("_")[0],  # TODO(?) sensitive to session nomenclature
            "sessionID": row["sessionID"],
            "behavior score": np.round(row["behavior score"],2),
            "unreliable": pfs_counts.get("unreliable", 0),
            "early": pfs_counts.get("early", 0),
            "transient": pfs_counts.get("transient", 0),
            "non-btsp": pfs_counts.get("non-btsp", 0),
            "btsp": pfs_counts.get("btsp", 0),
            "total": pfs_counts.sum()
        }
        pfs_scores_df_sess = pd.DataFrame.from_dict([pfs_scores_df_dict])
        pfs_scores_df = grow_df(pfs_scores_df, pfs_scores_df_sess)

    by_animal = True
    cats = list(CATEGORIES.keys())
    cols = ["sessionID", "behavior score", "unreliable", "early", "transient", "non-btsp", "btsp"]
    colors = [cat.color for cat in CATEGORIES.values()]

    if by_animal:
        n_animals = [7, 7]  # CA1, CA3
        for i_area, area in enumerate(["CA1", "CA3"]):
            fig, axs = plt.subplots(2, n_animals[i_area], figsize=(n_animals[i_area]*3,6))
            for i_animal, animal in enumerate(ANIMALS[area]):
                dff = pfs_scores_df[(pfs_scores_df["area"] == area) & (pfs_scores_df["animalID"] == animal)][cols]
                dff = dff.reset_index(drop=True).set_index("sessionID")

                dff[cats].plot(kind="bar", ax=axs[0, i_animal], color=colors, legend=False, sharex=axs[1, i_animal], width=0.75)
                axs[0, i_animal].set_title(animal)
                axs[0, i_animal].set_ylabel("number of PFs")

                dff_norm = dff[cats].div(dff[cats].sum(axis=1), axis=0)
                dff_norm.plot(kind="bar", stacked=True, ax=axs[1, i_animal], color=colors, legend=False, width=1)

                axs[1, i_animal].set_ylim([0,1])
                axs[1, i_animal].set_ylabel("PF proportion")
                secxis = axs[1, i_animal].twinx()
                secxis.set_ylim([0,8])
                secxis.set_ylabel("behavior score")
                dff.plot(y="behavior score", drawstyle="steps-mid", ax=secxis, legend=False, color="k", linewidth=3)
                axs[1, i_animal].set_xticklabels(axs[1, i_animal].get_xticklabels(), rotation=45, ha="right")
                axs[1, i_animal].set_xlabel("")
            fig.tight_layout()
            fig.savefig(f"{output_path}/statistics/behavior_scores_PFs_{area}.pdf")
            plt.close()
    else:
        fig, axs = plt.subplots(2,2)
        for i_area, area in enumerate(["CA1", "CA3"]):
            dff = pfs_scores_df
            dff = dff[dff["area"] == area][cols]
            dff = dff.reset_index(drop=True).set_index("behavior score")

            dff.plot(kind="bar", ax=axs[0, i_area], color=colors, legend=False, sharex=axs[1,i_area], width=0.75)
            axs[0, i_area].set_title(area)

            dff_norm = dff[cats].div(dff[cats].sum(axis=1), axis=0)
            dff_norm.plot(kind="bar", stacked=True, ax=axs[1, i_area], color=colors, legend=False, width=1)

            def calc_regression(pf_category):
                x = dff_norm.reset_index()[["behavior score", pf_category]].values[:, 0]
                y = dff_norm.reset_index()[["behavior score", pf_category]].values[:, 1]
                _, _, r_lr, p_lr, _ = scipy.stats.linregress(x,y)
                res = scipy.stats.spearmanr(x,y)
                r_sprm, p_sprm = res.statistic, res.pvalue
                return r_lr**2, p_lr, r_sprm, p_sprm

            fig2, axs2 = plt.subplots(1,5, figsize=(10,2))
            for i_cat, cat in enumerate(cats):
                r2_lr, p_lr, r_sprm, p_sprm = calc_regression(cat)
                sns.regplot(data=dff_norm.reset_index(), x="behavior score", y=cat, ax=axs2[i_cat], color=colors[i_cat],
                            line_kws={"label": f"reg: r2={r2_lr:.2f}, p={p_lr:.3f}\nsprm: r={r_sprm:.2f}, p={p_sprm:.3f}"})
                axs2[i_cat].legend()
            fig2.suptitle(area)
            fig2.tight_layout(h_pad=-5, w_pad=-2)
        fig.tight_layout()
        plt.show()

        fig, axs = plt.subplots()
        event_rates_df = pd.read_pickle(f"{data_path}/statistics/mean_event_rates.pickle")
        event_rates_df = event_rates_df.set_index(["area", "sessionID"])
        pfs_scores_df = pfs_scores_df.set_index(["area", "sessionID"])
        eventrates_scores_df = event_rates_df.join(pfs_scores_df)[["event rate", "behavior score"]].reset_index()
        sns.scatterplot(data=eventrates_scores_df, x="behavior score", y="event rate", hue="area")
        plt.show()

def behavior_pca():
    df_pca = df - df.mean()
    df_pca = df_pca / df_pca.std()
    covariance_matrix = df_pca.cov()
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # sort eigenvalues and eigenvectors (from largest eigenvalue to smallest)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # project data onto PC1 and PC2
    B = eigenvectors[:,:2]
    proj = df_pca @ B
    plt.scatter(proj.values[:, 0], proj.values[:, 1], label="manual (z-scored)")
    #plt.show()

    #stds = np.atleast_2d(df.std().values).repeat(repeats=len(df), axis=0)
    #means = np.atleast_2d(df.mean().values).repeat(repeats=len(df), axis=0)
    #df_pca_tf = np.multiply((df_pca @ B @ B.T), stds) + means

    pca = PCA(n_components=2)
    df_tf = pca.fit_transform(df)
    plt.scatter(df_tf[:,0], df_tf[:,1], label="sklearn")

    # z-scored PCA
    df_tf = pca.fit_transform(df_pca)
    plt.scatter(df_tf[:,0], df_tf[:,1], label="sklearn (z-scored)")
    plt.legend()
    plt.show()

plot_behavior_scores()
#plot_place_fields_by_score()