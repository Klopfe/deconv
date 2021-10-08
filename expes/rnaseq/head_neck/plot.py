from deconv.utils import mae, rmse
from sparse_ho.utils_plot import configure_plt, plot_legend_apart
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import pylab

configure_plt()
current_palette = sns.color_palette("colorblind")

# plot figure 
dict_res = {}
dict_res["ssvr"] = np.load('ssvr.npy')
dict_res["cibersort"] = np.load('cibersort.npy')
dict_res["SOLS"] = np.load('SOLS.npy')

ground_truth = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/head_and_neck/groundtruth.txt",header=0,index_col=0, delimiter="\t")
ground_truth = np.array(ground_truth)

dict_method = {}
dict_method['ssvr'] = "Simplex SVR"
dict_method['SOLS'] = "Simplex Least Squares"
dict_method['cibersort'] = "Cibersort" 


methods = ["ssvr", "cibersort"]
# cell_types = ["Malignant cells", "Endothelial cells", "Fibroblast", "CD8 T-cells","NK cells", "Macrophages", "CD4 T-cells",  "B-cells"]
cell_types = ["B-cells", "CD8 T-cells", "CD4 T-cells", "Macrophages", "Dendritic cells", "Mast cells", "Myocytes", "Fibroblast" , "Endothelial cells", "Malignant cells"]

fig, axarr = plt.subplots(
    1, len(methods), sharex=True, sharey=True,
    figsize=[14, 7], constrained_layout=True)


for i, method in enumerate(methods):
    for j, cell in enumerate(cell_types):
        axarr[i].scatter(ground_truth[:, j], dict_res[method][:, j], color=current_palette[j], alpha=0.9, label=cell_types[j])
        axarr[i].plot((0, 0.75), (0, 0.75), color="black", marker='', linestyle='--', linewidth=1)
        axarr[i].set_xlim((0, 0.75))
        axarr[i].set_ylim((0, 0.75))
    corr_r = np.corrcoef(ground_truth.flatten(), dict_res[method].flatten())
    MAE = mae(ground_truth.flatten(), dict_res[method].flatten(), axis=None)
    infos = "R = %1.2f" %corr_r[1, 0]
    infos += "\n"
    infos += "MAE = %1.2f" %MAE 
    axarr[i].text(0.05, 0.55, infos, fontsize=15, weight="bold")
    # axarr[-1, j].set_xlabel("Ground truth")
    string_ylabel = "Estimated proportions" 
    axarr[i].set_xlabel("Ground Truth")
    axarr[i].set_title(dict_method[method])
axarr[0].set_ylabel(string_ylabel)
fig.tight_layout()
fig.savefig('HNSCC.pdf')
fig.show()


handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in current_palette[0:8]]
figlegend = pylab.figure(figsize=(14, 3.5))
figlegend.legend(handlelist, cell_types, loc='center', ncol=4)
figlegend.savefig("legend_HNSCC.pdf")