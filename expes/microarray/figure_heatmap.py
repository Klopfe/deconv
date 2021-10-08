import matplotlib.pyplot as plt
from numpy.linalg import norm
from sparse_ho.utils_plot import configure_plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import ticker

configure_plt()
current_palette = sns.color_palette("colorblind")

dict_methods = {}
dict_methods["ssvr"] = "Simplex SVR"
dict_methods["cibersort"] = "Cibersort"
dict_methods["SOLS"] = "Simplex OLS"

methods = ["ssvr", "cibersort", "SOLS"]

fig, axarr = plt.subplots(
    1, len(methods), sharex=False, sharey=True,
    figsize=[14, 4], constrained_layout=True)
RMSE = np.load('results/RMSE_%s.npy' %'cibersort')
v_max = RMSE.max()


for i, method in enumerate(methods):

    RMSE = np.load('results/RMSE_%s.npy' %method)
    RMSE[RMSE<0] = 0.0
    cax = axarr[i].imshow(RMSE.T, cmap='jet_r', interpolation='gaussian', aspect='auto', origin="lower", vmax=v_max, extent=[0, 1, 0, 1])
    
    axarr[i].set_xlabel('Tumor content (\%)')
    axarr[i].set_title(dict_methods[method])
    axarr[i].set_yticks(np.around(np.linspace(0, 1, num=5),2))
    axarr[i].set_xticks(np.around(np.linspace(0, 1, num=5),2))
    plt.rcParams["axes.grid"] = False
cbar = fig.colorbar(cax)
# cbar.set_ticks([0.00, 0.25, 0.5, 0.75, 1.00])
cbar.ax.set_title('RMSE')
axarr[0].set_ylabel('Noise (x1 s.d.)')
fig.show()


save_fig = True
# save_fig = False

if save_fig:
    fig_dir = "../../../../manuscript/thesis/prebuiltimages/"
    fig.savefig(
        fig_dir + "heatmaps_microarray_RMSE.pdf", bbox_inches="tight")
fig.show()

