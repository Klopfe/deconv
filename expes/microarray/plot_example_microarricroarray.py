from deconv.utils import mae, rmse
from sparse_ho.utils_plot import configure_plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

configure_plt()
current_palette = sns.color_palette("colorblind")

# plot figure 
dict_res = {}
dict_res["ssvr"] = np.load('results/estimated_ssvr.npy')
dict_res["cibersort"] = np.load('results/estimated_cibersort.npy')
dict_res["SOLS"] = np.load('results/estimated_SOLS.npy')

ground_truth = np.load('results/ground_truth_simu.npy')

dict_method = {}
dict_method['ssvr'] = "Simplex SVR"
dict_method['SOLS'] = "Simplex OLS"
dict_method['cibersort'] = "Cibersort" 


methods = ["ssvr", "SOLS", "cibersort"]
cell_types = ["Jurkat", "IM-9", "Raji", "THP1"]

fig, axarr = plt.subplots(
    3, 4, sharex=True, sharey=True,
    figsize=[20, 8], constrained_layout=True)


for i, method in enumerate(methods):
    for j, cell in enumerate(cell_types):
        axarr[i, j].scatter(ground_truth[j, :], dict_res[method][:, j], color=current_palette[j], alpha=0.4)
        axarr[i, j].plot((0, 1), (0, 1), color="black", marker='', linestyle='--', linewidth=1)
        axarr[i, j].set_xlim((0, 1))
        axarr[i, j].set_ylim((0, 1))
        corr_r = np.corrcoef(ground_truth[j, :], dict_res[method][:, j])
        MAE = mae(ground_truth[j, :], dict_res[method][:, j], axis=None)
        infos = "R = %1.2f" %corr_r[1, 0]
        infos += "\n"
        infos += "MAE = %1.2f" %MAE 
        axarr[i, j].text(0.05, 0.75, infos, fontsize=15)
        axarr[0, j].set_title(cell)
        axarr[-1, j].set_xlabel("Ground truth")
    string_ylabel = dict_method[method] + "\n" + "Estimated" 
    axarr[i, 0].set_ylabel(string_ylabel)


save_fig = False
# save_fig = False

if save_fig:
    fig_dir = "../../../../manuscript/thesis/prebuiltimages/"
    fig.savefig(
        fig_dir + "illustrative_microarray.pdf", bbox_inches="tight")
fig.show()


    
