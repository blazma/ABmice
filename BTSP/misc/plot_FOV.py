import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_dir = r"D:\CA3\data"
out_dir = r"C:\Users\martin\home\phd\misc\fovs"
sessions = pd.read_excel(rf"D:\CA3\CA3_meta.xlsx")
for i_row, row in sessions.iterrows():
    session = row["session id"]
    animal = row["name"]
    print(session)
    ops = np.load(f"{base_dir}/{animal}_imaging/{session}/ops.npy", allow_pickle=True).item()
    fig, axs = plt.subplots(1,2, figsize=(16,8))
    axs[0].imshow(ops["meanImg"], cmap="gray", origin="lower", interpolation=None)
    axs[0].set_title("meanImg")
    axs[1].imshow(ops["max_proj"], cmap="gray", origin="lower", interpolation=None)
    axs[1].set_title("max_proj")
    plt.suptitle(session)
    plt.savefig(f"{out_dir}/{session}.pdf")
    plt.close()
