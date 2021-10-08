from deconv.utils import mae, rmse
from sparse_ho.utils_plot import configure_plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

configure_plt()
current_palette = sns.color_palette("colorblind")

# plot figure 
dict_res = {}
dict_res["ssvr"] = np.load('ssvr.npy')
dict_res["cibersort"] = np.load('cibersort.npy')
dict_res["SOLS"] = np.load('SOLS.npy')

ground_truth = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/illumina_ground_truth.txt",header=0,index_col=0, delimiter="\t")
ground_truth = np.array(ground_truth) / 100

dict_method = {}
dict_method['ssvr'] = "Simplex SVR"
dict_method['SOLS'] = "Simplex Least Squares"
dict_method['cibersort'] = "Cibersort" 


methods = ["ssvr", "SOLS", "cibersort"]
cell_types = ['B cells naive', 'B cells memory', 'T cells CD8',
       'T cells CD4 naive', 'T cells CD4 memory resting',
       'T cells CD4 memory activated', 'T cells gamma delta', 'NK cells', 'Monocytes']

fig, axarr = plt.subplots(
    1, 3, sharex=True, sharey=True,
    figsize=[14, 4], constrained_layout=True)


for i, method in enumerate(methods):
    for j, cell in enumerate(cell_types):
        axarr[i].scatter(ground_truth[:, j], dict_res[method][j, :], color=current_palette[j], alpha=0.4)
        axarr[i].plot((0, 1), (0, 1), color="black", marker='', linestyle='--', linewidth=1)
        axarr[i].set_xlim((0, 1))
        axarr[i].set_ylim((0, 1))
    corr_r = np.corrcoef(ground_truth.flatten(), dict_res[method].flatten())
    MAE = mae(ground_truth.flatten(), dict_res[method].flatten(), axis=None)
    infos = "R = %1.2f" %corr_r[1, 0]
    infos += "\n"
    infos += "MAE = %1.2f" %MAE 
    axarr[i].text(0.05, 0.75, infos, fontsize=15)
    # axarr[-1, j].set_xlabel("Ground truth")
    string_ylabel = dict_method[method] + "\n" + "Estimated" 
    axarr[i].set_ylabel(string_ylabel)
fig.show()
    
