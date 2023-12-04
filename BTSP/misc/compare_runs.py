import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from constants import CATEGORIES
from utils import grow_df, makedir_if_needed

run_types = ["ctrl", "tau0.8", "gaspar", "tau0.8_gaspar"]
base_dir = r"C:\Users\martin\home\phd\btsp_project\analyses\special"
makedir_if_needed(f"{base_dir}/compare_runs")

def plot_cell_stats(area, date, animal):
    cell_stats_df = None
    for run_type in run_types:
        run_type_df = pd.read_pickle(f"{base_dir}\\BTSP_analysis_{area}_{date}_{run_type}\\{animal}_cell_stats_df.pickle")
        run_type_df["run type"] = run_type
        cell_stats_df = grow_df(cell_stats_df, run_type_df)
    cell_stats_df = cell_stats_df.melt(id_vars=["animalID", "sessionID", "run type"], var_name="cell type", value_name="cell counts")
    g = sns.catplot(cell_stats_df, x="cell type", y="cell counts", hue="run type", kind="swarm")
    sns.move_legend(g, "right", bbox_to_anchor=(1, 0.5), title='Species')
    #plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f"{base_dir}/compare_runs/cell_counts_{area}.pdf")
    plt.close()

plot_cell_stats("CA1", "231114", "KS030")
plot_cell_stats("CA3", "231114", "srb270")

def plot_place_fields(area, date, animal):
    pfs_df = None
    for run_type in run_types:
        run_type_df = pd.read_pickle(f"{base_dir}\\BTSP_analysis_{area}_{date}_{run_type}\\{animal}_place_fields_df.pickle")
        run_type_df["run type"] = run_type
        pfs_df = grow_df(pfs_df, run_type_df)
    pf_counts = pfs_df[["session id", "category", "run type"]].groupby(["category", "run type"]).count().reset_index()
    g = sns.catplot(pf_counts, x="category", y="session id", hue="run type", kind="bar")
    g.set_ylabels("number of place fields")
    sns.move_legend(g, "right", bbox_to_anchor=(1, 0.5), title='Species')
    #plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.savefig(f"{base_dir}/compare_runs/place_field_counts_{area}.pdf")
    plt.close()

    categories_colors_RGB = [category.color for _, category in CATEGORIES.items()]
    #for category in CATEGORIES:
    #    cond = pf_counts["category"] == category
    #    pf_counts.loc[cond, "category_order"] = CATEGORIES[category].order
    #pf_counts = pf_counts.sort_values(by="category_order")
    pf_counts = pf_counts.pivot(index="run type", columns="category").droplevel(0, axis=1).reindex(CATEGORIES.keys(),axis=1)
    ax = pf_counts.plot(kind="bar", stacked=True, color=categories_colors_RGB)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title(area)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/compare_runs/place_field_counts_{area}_stacked.pdf")
    plt.close()

plot_place_fields("CA1", "231114", "KS030")
plot_place_fields("CA3", "231114", "srb270")
