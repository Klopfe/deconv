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
dict_res["ssvr"] = np.load('estimated_ssvr_GSE65135.npy')
dict_res["cibersort"] = np.load('estimated_cibersort_GSE65135.npy')
dict_res["SOLS"] = np.load('estimated_SOLS_GSE65135.npy')

ground_truth = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/Ground_truth.txt",header=0,index_col=0, delimiter="\t")
ground_truth = np.array(ground_truth)

dict_method = {}
dict_method['ssvr'] = "Simplex SVR"
dict_method['SOLS'] = "Simplex Least Squares"
dict_method['cibersort'] = "Cibersort" 


methods = ["ssvr", "SOLS", "cibersort"]
cell_types = ["CD4 T-cells", "CD8 T-cells", "B-cells"]

fig, axarr = plt.subplots(
    3, 3, sharex=True, sharey=True,
    figsize=[14, 4], constrained_layout=True)


for i, method in enumerate(methods):
    for j, cell in enumerate(cell_types):
        axarr[i, j].scatter(ground_truth[:, j], dict_res[method][j, :], color=current_palette[j], alpha=0.4)
        axarr[i, j].plot((0, 1), (0, 1), color="black", marker='', linestyle='--', linewidth=1)
        axarr[i, j].set_xlim((0, 1))
        axarr[i, j].set_ylim((0, 1))
        corr_r = np.corrcoef(ground_truth[:,j], dict_res[method][j, :])
        MAE = mae(ground_truth[:, j], dict_res[method][j, :], axis=None)
        infos = "R = %1.2f" %corr_r[1, 0]
        infos += "\n"
        infos += "MAE = %1.2f" %MAE 
        axarr[i, j].text(0.05, 0.75, infos, fontsize=15)
        axarr[0, j].set_title(cell)
        axarr[-1, j].set_xlabel("Ground truth")
    string_ylabel = dict_method[method] + "\n" + "Estimated" 
    axarr[i, 0].set_ylabel(string_ylabel)
fig.show()
    
