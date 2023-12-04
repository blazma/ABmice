import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils import grow_df

output_stats_path = r"C:\Users\martin\home\phd\btsp_project\analyses\parallel\stats"

df = None
for FLW in [3, 5, 7]:
    for FGW in [3, 5, 7]:
        for FT in [0.05, 0.1, 0.2]:
            if FGW > FLW:
                continue

            params = {
                "formation lap window": FLW,
                "formation gain window": FGW,
                "formation threshold": FT
            }
            df_params = pd.DataFrame.from_dict([params])

            results_json_path = f"{output_stats_path}/results_FLW={FLW}_FGW={FGW}_FT={FT}.json"
            with open(results_json_path, "r") as results_json:
                results_dict = json.load(results_json)
            df_results = pd.DataFrame.from_dict([results_dict])
            df_run = pd.concat((df_params, df_results), axis=1)
            df = grow_df(df, df_run)

results_scores = ['mean(shift) newlyF', 'mean(shift) establ', 'median(shift) newlyF', 'median(shift) establ',
                  'mean(gain) newlyF', 'mean(gain) establ', 'median(gain) newlyF', 'median(gain) establ',
                  'mean(log10(gain)) newlyF', 'mean(log10(gain)) establ', 'median(log10(gain)) newlyF',
                  'median(log10(gain)) establ']

diff_scores = {
    "diff mean(shift)": ['mean(shift) newlyF', 'mean(shift) establ'],
    "diff median(shift)": ['median(shift) newlyF', 'median(shift) establ'],
    "diff mean(log10(gain))": ['mean(log10(gain)) newlyF', 'mean(log10(gain)) establ'],
    "diff median(log10(gain))": ['median(log10(gain)) newlyF', 'median(log10(gain)) establ'],
}
for diff_score, (newly_score, estab_score) in diff_scores.items():
    df[diff_score] = df[newly_score]-df[estab_score]
results_scores = results_scores + list(diff_scores.keys())

for score in results_scores:
    fig, axs = plt.subplots(1, 3, figsize=(6.1,3))
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    vmin = df[score].min()
    vmax = df[score].max()
    for i_FT, FT in enumerate([0.05, 0.1, 0.2]):
        df_filt = df[df["formation threshold"]==FT][["formation lap window", "formation gain window", score]]
        df_pivot = df_filt.pivot(index="formation gain window", columns="formation lap window", values=score)
        if "newlyF" in score:
            cmap = "Oranges"
        elif "estab" in score:
            cmap = "Blues"
        else:
            cmap = "Greys"
        sns.heatmap(df_pivot, ax=axs[i_FT], vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax, cmap=cmap, annot=True)
        axs[i_FT].invert_yaxis()
        axs[i_FT].set_title(f"FT = {FT}")
    plt.suptitle(score)
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    plt.savefig(f"{output_stats_path}/{score}.pdf")
