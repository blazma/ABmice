import numpy as np
import pickle
from BTSP.BtspStatistics import BtspStatistics
from matplotlib import pyplot as plt
from tqdm import tqdm


def calc_synthetic_autocorr():
    rng = np.random.default_rng(1234)

    N = 100
    ts = np.arange(0, N * 2 * np.pi, 0.1)
    Xs = np.sin(ts) + 10 * rng.random(len(ts))
    Xs = Xs / Xs.max()
    plt.plot(ts, Xs)

    R = np.correlate(Xs, Xs, mode="full")
    plt.plot(ts, R[R.size // 2:] / R.max())
    plt.axvline(2 * np.pi, c="k", linestyle="dashed")

########################################################

data_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual"
tuned_cells_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\tuned_cells\CA1"
output_path = r"C:\Users\martin\home\phd\btsp_project\analyses\manual\misc\autocorr"
animals_sessions = {
    "KS028": ["KS028_103121",
              "KS028_110121",
              "KS028_110221", ],
    "KS029": ["KS029_110621",
              "KS029_110721",
              "KS029_110821",
              "KS029_110921", ],
    "KS030": ["KS030_102921",
              "KS030_103021",
              "KS030_103121",
              "KS030_110121", ],
    "srb131": ["srb131_211015",
               "srb131_211016", ],
    "srb402": ["srb402_240319_CA1",
               "srb402_240320_CA1"],
    "srb410": ["srb410_240528_CA1"]
}

def calc_autocorr(cell_subset=""):
    if cell_subset == "reliablePCs":
        stat = BtspStatistics("CA1", data_path, output_path)
        stat.load_data()
        stat.filter_low_behavior_score()
        stat.calc_shift_gain_distribution()

    for animal, sessions in tqdm(animals_sessions.items()):
        for session in sessions:
            Rs = []
            session_path = rf"D:\CA1\data\{animal}_imaging\{session}"
            save_path = f"{output_path}/autocorr_{session}"

            spks = np.load(f"{session_path}/spks.npy")
            iscell = np.load(f"{session_path}/iscell.npy")

            spks_cells = spks[iscell[:,0] == 1, :]
            if cell_subset == "tuned":
                with open(f"{tuned_cells_path}/tuned_cells_{session}.pickle", "rb") as tcl_file:
                    tcl = pickle.load(tcl_file)
                tc_idxs = np.unique(np.array([tc.cellid for tc in tcl]))
                spks_cells = spks_cells[tc_idxs, :]
                save_path = f"{save_path}_tuned"
            elif cell_subset == "reliablePCs":
                sg_df = stat.shift_gain_df
                cells_with_reliable_pfs = sg_df[sg_df["session id"] == session]["cell id"].unique()
                spks_cells = spks_cells[cells_with_reliable_pfs,:]
                save_path = f"{save_path}_reliablePCs"

            #T = spks_cells.shape[1]
            for cell in range(spks_cells.shape[0]):
                if spks_cells[cell,:].sum() == 0:
                    continue
                # OG method:
                #R = np.correlate(spks_cells[cell,:],spks_cells[cell,:],mode="full")
                #T = np.argmax(R)  # "aligned" suffix
                #R = R[T:T+1000]

                # diffmethod:
                mean = np.mean(spks_cells[cell,:])
                var = np.var(spks_cells[cell,:])
                spks_normed = spks_cells[cell,:] - mean
                R = np.correlate(spks_normed, spks_normed, "full")[len(spks_normed)-1:]
                R = R / var /len(spks_normed)
                R = R[:1000]

                Rs.append(R)
            Rs = np.array(Rs).mean(axis=0)
            np.save(f"{save_path}_diffmethod.npy", Rs)


def plot_autocorr(cell_subset=""):
    fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
    ax_idx = 0
    dt = 1/30  # 30 Hz sampling rate
    t = np.arange(0, 1000 * dt, dt)[:-1]
    for animal, sessions in animals_sessions.items():
        ax = axs[ax_idx%2,ax_idx%3]
        ax.axvline(0,c="k",linestyle="--")
        ax.axhline(0,c="k",linestyle="--")
        [ax.axvline(t_,c="k", alpha=0.1) for t_ in np.arange(0,1000*dt,6*dt)[:-1]]
        for session in sessions:
            load_path = f"{output_path}/autocorr_{session}"
            Rs = np.load(f"{load_path}_{cell_subset}_diffmethod.npy")
            Rs = Rs/Rs.max()
            ax.plot(t, Rs.T, label=session, marker="o", markersize=4, markerfacecolor="none")
        ax_idx += 1
        ax.legend()
    fig.suptitle(f"{cell_subset}")

cell_subset = "reliablePCs"
#calc_autocorr(cell_subset)
plot_autocorr(cell_subset)
plt.show()