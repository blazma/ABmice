from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import scipy

# add font
#from matplotlib import font_manager
#font_dirs = ['C:\\home\\phd\\']
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#for font_file in font_files:
#    font_manager.fontManager.addfont(font_file)
#plt.rcParams['font.family'] = 'Trade Gothic Next LT Pro BdCn'

def stack(v1, v2):
    if v1 is None:
        return v2
    else:
        return np.vstack((v1, v2))

def synthetic_data():
    N_place_fields = 1000
    N_laps = 10
    pf_width = 50
    p_btsp = 0.1
    gain_btsp = 1.3
    shift_btsp = 5  # bins

    # synthetic data
    seed = 1234
    rng = np.random.default_rng(seed)
    nmf_matrix = None
    btsp_pfs = rng.binomial(1, p_btsp, N_place_fields)
    for j in range(N_place_fields):
        pf = None
        for i in range(N_laps):
            random_fluct = rng.random()
            if btsp_pfs[j] and i == 0:
                norm = rng.normal(loc=(pf_width//2)+shift_btsp, scale=1.5, size=100)
                hist, _ = np.histogram(norm, bins=pf_width, range=(0, pf_width))
                hist = gain_btsp * random_fluct * hist
            else:
                norm = rng.normal(loc=pf_width//2, scale=1.5, size=100)
                hist, _ = np.histogram(norm, bins=pf_width, range=(0, pf_width))
                hist = random_fluct * hist
            pf = stack(pf, hist)
        pf_flat = pf.flatten()
        nmf_matrix = stack(nmf_matrix, pf_flat)
    return nmf_matrix

nmf_matrix = synthetic_data()
factor = 0.3
plt.figure(figsize=(factor * 15, factor * 5), dpi=120)
im = plt.imshow(nmf_matrix, cmap="binary", aspect="auto")
N_laps = 10
pf_width = 50

for j in range(N_laps):
    plt.axvline(j * pf_width + pf_width // 2, color="red", linestyle="dashed", alpha=0.5)
    if j < N_laps - 1:
        plt.axvline((j + 1) * pf_width, color="red", alpha=0.5)
# plt.savefig(f"{analysis_folder}/nmf_matrix_{nmf_type}.png", dpi=200)
# plt.ylim([0,200])
plt.ylabel("place field")
plt.xlabel("spatial position bin x lap")
plt.tight_layout()
# plt.close()

# running NMF
N_comps = 11
model = NMF(n_components=N_comps, max_iter=2000, l1_ratio=1.0, alpha_H=0.0025)
W = model.fit_transform(nmf_matrix)
H = model.components_
# H_norm = H / H.max(axis=1, keepdims=True)

H_reordered = np.stack(sorted(list(H), key=lambda row: row.argmax()), axis=0)
factor = 0.9
plt.figure(figsize=(factor * 5, factor * 3), dpi=130)
plt.pcolormesh(H_reordered, cmap='binary')
ax = plt.gca()
# ax.set_aspect("auto")
ax.invert_yaxis()
N_laps = 10
for j in range(N_laps):
    plt.axvline(j * pf_width + pf_width // 2, color="red", linestyle="dashed", alpha=0.5)
    if j < N_laps - 1:
        plt.axvline((j + 1) * pf_width, color="red", alpha=0.5)
plt.ylabel("NMF component")
plt.xlabel("spatial position bin x lap")
plt.tight_layout()
plt.show()

"""
def actual_data(analysis_folder, animals, nmf_type):
    # compilation of NMF matrix
    nmf_matrix_withnan = None
    for animal in animals:
        with open(f"{analysis_folder}/{animal}/{animal}_NMF_matrix_{nmf_type}.npy", "rb") as nmf_file:
            nmf_matrix_withnan = stack(nmf_matrix_withnan, np.load(nmf_file))
    nmf_matrix = nmf_matrix_withnan[~np.isnan(nmf_matrix_withnan).any(axis=1)]
    nmf_matrix = nmf_matrix / nmf_matrix.max(axis=1, keepdims=True)  # normalize activity for each pf
    return nmf_matrix


animals_CA1 = ["KS028",
               "KS029",
               "KS030",
               "srb131"]
animals_CA3 = ["srb231",
               "srb251",
               "srb269",
               "srb270"]
# narrow window
#analysis_folder = "BTSP_analysis_CA1_230912"  # narrow nmf
#pf_width = 24  # narrow_nmf
#nmf_types = ["all", "btsp"]

# wide window
#analysis_folder = "BTSP_analysis_CA1_230918_newlyNMF"
analysis_folder = "BTSP_analysis_CA3_230918_newlyNMF"
animals = animals_CA3
pf_width = 48  # wide_nmf
nmf_types = ["newly"] #["all", "non_btsp", "btsp"]

for nmf_type in nmf_types:
    # loading nmf_matrix
    nmf_matrix = actual_data(analysis_folder, animals, nmf_type)

    sigma = 2.0
    nmf_matrix = actual_data(analysis_folder, animals, nmf_type)
    nmf_matrix = scipy.ndimage.gaussian_filter1d(nmf_matrix, sigma=sigma, axis=1)

    # subsample
    #N_rows = nmf_matrix.shape[0]
    #N_selection = 90
    #rand_pf_idxs = np.random.randint(0,N_rows,N_selection)
    #nmf_matrix = nmf_matrix[rand_pf_idxs,:]

    factor = 0.3
    plt.figure(figsize=(factor*15,factor*5), dpi=120)
    im = plt.imshow(nmf_matrix, cmap="binary", aspect="auto")
    N_laps = 10
    for j in range(N_laps):
        plt.axvline(j * pf_width + pf_width // 2, color="red", linestyle="dashed", alpha=0.5)
        if j<N_laps-1:
            plt.axvline((j+1) * pf_width, color="red", alpha=0.5)
    #plt.savefig(f"{analysis_folder}/nmf_matrix_{nmf_type}.png", dpi=200)
    #plt.ylim([0,200])
    plt.ylabel("place field")
    plt.xlabel("spatial position bin x lap")
    plt.tight_layout()
    plt.show()
    #plt.close()

    # running NMF
    N_comps = 11
    model = NMF(n_components=N_comps, max_iter=2000, l1_ratio=1.0, alpha_H=0.0025)
    W = model.fit_transform(nmf_matrix)
    H = model.components_
    #H_norm = H / H.max(axis=1, keepdims=True)

    H_reordered = np.stack(sorted(list(H), key=lambda row: row.argmax()), axis=0)
    factor = 0.9
    plt.figure(figsize=(factor*5,factor*3), dpi=130)
    plt.pcolormesh(H_reordered, cmap='binary')
    ax = plt.gca()
    # ax.set_aspect("auto")
    ax.invert_yaxis()
    N_laps = 10
    for j in range(N_laps):
        plt.axvline(j * pf_width + pf_width // 2, color="red", linestyle="dashed", alpha=0.5)
        if j<N_laps-1:
            plt.axvline((j+1) * pf_width, color="red", alpha=0.5)
    plt.ylabel("NMF component")
    plt.xlabel("spatial position bin x lap")
    plt.tight_layout()
    plt.show()

    # errors = []
    # for i in [0.005, 0.0075, 0.01, 0.02, 0.04, 0.05]:
    #     model = NMF(n_components=11, max_iter=2000, l1_ratio=1.0, alpha_H=i)
    #     W = model.fit_transform(nmf_matrix)
    #     H = model.components_
    #     errors.append(model.reconstruction_err_)
    #
    #     plt.pcolormesh(H, cmap='viridis_r')
    #     ax = plt.gca()
    #     # ax.set_aspect("auto")
    #     ax.invert_yaxis()
    #     N_laps = 10
    #     for j in range(N_laps):
    #         plt.axvline(j * pf_width + pf_width // 2, color="white", linestyle="dashed", alpha=0.5)
    #         plt.axvline(j * pf_width, color="white", alpha=0.5)
    #     plt.savefig(f"{analysis_folder}/nmf_components_Ncomp_11_{nmf_type}_{i}.png")
    #     plt.close()
    # print(errors)


    #plt.savefig(f"{analysis_folder}/nmf_components_Ncomp_{N_comps}_{nmf_type}.png")
    #plt.close()

    #reconstruction errors
    # N_comps = 20
    # errors = []
    # comps = list(range(1,N_comps+1))
    # for n_comp in comps:
    #     print(f"running NMF with {n_comp} components...")
    #     model = NMF(n_components=n_comp, max_iter=2000)
    #     W = model.fit_transform(nmf_matrix)
    #     H = model.components_
    #     error = model.reconstruction_err_
    #     errors.append(error)
    # plt.figure()
    # plt.plot(comps,errors)
    # plt.xticks(comps, labels=comps)
    # plt.show()
    #plt.savefig(f"{analysis_folder}/error_Ncomp_{N_comps}_{nmf_type}.png")
    #plt.close()
"""