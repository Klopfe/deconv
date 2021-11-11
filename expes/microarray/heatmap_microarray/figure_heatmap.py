import matplotlib.pyplot as plt
from sparse_ho.utils_plot import configure_plt
import numpy as np
import seaborn as sns

configure_plt()
current_palette = sns.color_palette("colorblind")

dict_methods = {}
dict_methods["ssvr"] = "AC-SVR"
dict_methods["cibersort"] = "Cibersort"
dict_methods["sols"] = "Simplex OLS"
dict_methods["fardeep"] = "FARDEEP"
dict_methods["hspe"] = "HSPE"
dict_methods["EPIC"] = "EPIC"


methods = ["ssvr",  "cibersort", "sols", "fardeep", "EPIC", "hspe"]

fig, axarr = plt.subplots(
    2, len(methods) // 2, sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)
RMSE = np.load('results/RMSE_%s.npy' % 'cibersort')
v_max = 0.3


for i, method in enumerate(methods):
    if i >= len(methods) // 2:
        idx_row = 1
    else:
        idx_row = 0
    RMSE = np.load('results/RMSE_%s.npy' % method)
    if np.sum(np.isnan(RMSE)) > 0:
        is_nan = np.isnan(RMSE)
        RMSE[is_nan] = np.max(RMSE[np.logical_not(is_nan)])
    RMSE[RMSE < 0] = 0.0
    cax = axarr[idx_row, i % 3].imshow(
        RMSE.T, cmap='jet_r', interpolation='gaussian', aspect='auto',
        origin="lower", vmax=v_max, extent=[0, 1, 0, 1])

    axarr[idx_row, i % 3].set_xlabel(r'Tumor content (\%)')
    axarr[idx_row, i % 3].set_title(dict_methods[method])
    axarr[idx_row, i % 3].set_yticks(np.around(np.linspace(0, 1, num=5), 2))
    axarr[idx_row, i % 3].set_xticks(np.around(np.linspace(0, 1, num=5), 2))
    plt.rcParams["axes.grid"] = False
cbar = fig.colorbar(cax, ax=axarr.ravel().tolist(), pad=0.04, aspect=30)
cbar.ax.set_title('RMSE')
axarr[0, 0].set_ylabel('Noise (x1 s.d.)')
axarr[1, 0].set_ylabel('Noise (x1 s.d.)')
fig.show()

fig_dir = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
           "Manuscript/figures/")
save_fig = True

if save_fig:
    fig.savefig(
        fig_dir + "heatmaps_microarray_RMSE.pdf", bbox_inches="tight")
fig.show()

fig, axarr = plt.subplots(
    2, len(methods) // 2, sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)
MAE = np.load('results/MAE_%s.npy' % 'cibersort')
v_max = 0.25


for i, method in enumerate(methods):
    if i >= len(methods) // 2:
        idx_row = 1
    else:
        idx_row = 0
    MAE = np.load('results/MAE_%s.npy' % method)
    if np.sum(np.isnan(MAE)) > 0:
        is_nan = np.isnan(MAE)
        MAE[is_nan] = np.max(MAE[np.logical_not(is_nan)])
    MAE[MAE < 0] = 0.0
    cax = axarr[idx_row, i % 3].imshow(
        MAE.T, cmap='jet_r', interpolation='gaussian', aspect='auto',
        origin="lower", vmax=v_max, extent=[0, 1, 0, 1])

    axarr[idx_row, i % 3].set_xlabel(r'Tumor content (\%)')
    axarr[idx_row, i % 3].set_title(dict_methods[method])
    axarr[idx_row, i % 3].set_yticks(np.around(np.linspace(0, 1, num=5), 2))
    axarr[idx_row, i % 3].set_xticks(np.around(np.linspace(0, 1, num=5), 2))
    plt.rcParams["axes.grid"] = False
cbar = fig.colorbar(cax, ax=axarr.ravel().tolist(), pad=0.04, aspect=30)

# cbar.set_ticks([0.00, 0.25, 0.5, 0.75, 1.00])
cbar.ax.set_title('MAE')
axarr[0, 0].set_ylabel('Noise (x1 s.d.)')
axarr[1, 0].set_ylabel('Noise (x1 s.d.)')
fig.show()


# save_fig = True
save_fig = True

if save_fig:
    fig.savefig(
        fig_dir + "heatmaps_microarray_MAE.pdf", bbox_inches="tight")
fig.show()

fig, axarr = plt.subplots(
    2, len(methods) // 2, sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)
Corr = np.load('results/Corr_%s.npy' % 'cibersort')
v_max = 1.0


for i, method in enumerate(methods):
    if i >= len(methods) // 2:
        idx_row = 1
    else:
        idx_row = 0
    Corr = np.load('results/Corr_%s.npy' % method)
    if np.sum(np.isnan(Corr)) > 0:
        is_nan = np.isnan(Corr)
        Corr[is_nan] = np.max(Corr[np.logical_not(is_nan)])
    Corr[Corr < 0] = 0.0
    cax = axarr[idx_row, i % 3].imshow(
        Corr.T, cmap='jet', interpolation='gaussian', aspect='auto',
        origin="lower", vmax=v_max, vmin=0.0, extent=[0, 1, 0, 1])

    axarr[idx_row, i % 3].set_xlabel(r'Tumor content (\%)')
    axarr[idx_row, i % 3].set_title(dict_methods[method])
    axarr[idx_row, i % 3].set_yticks(np.around(np.linspace(0, 1, num=5), 2))
    axarr[idx_row, i % 3].set_xticks(np.around(np.linspace(0, 1, num=5), 2))
    plt.rcParams["axes.grid"] = False
cbar = fig.colorbar(cax, ax=axarr.ravel().tolist(), pad=0.04, aspect=30)

cbar.set_ticks([0.00, 0.25, 0.5, 0.75, 1.00])
cbar.ax.set_title('R')
axarr[0, 0].set_ylabel('Noise (x1 s.d.)')
axarr[1, 0].set_ylabel('Noise (x1 s.d.)')
fig.show()

save_fig = True

if save_fig:
    fig.savefig(
        fig_dir + "heatmaps_microarray_Corr.pdf", bbox_inches="tight")
fig.show()
