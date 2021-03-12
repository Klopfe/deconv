import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from joblib import Parallel, delayed
from itertools import product
import pandas as pd
from deconv.deconv_methods import *
from deconv.utils import semi_synth_data, mae, rmse, quantile_normalize

dict_method = {}
dict_method["cibersort"] = cibersort
dict_method["ssvr"] = deconv_ssvr
dict_method["SOLS"] = SOLS


X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE11103/GSE11103_X.txt", header=0, index_col=0, delimiter="\t")
ref = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE11103/GSE11103_X_complete.txt",header=0,index_col=0, delimiter="\t")
tumor_content = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/tumor_content/tumor_content.txt", header=0, index_col=0, delimiter="\t")

ref = ref[ref.index.isin(tumor_content.index)]
tumor_content = tumor_content[tumor_content.index.isin(ref.index)]

ref = ref[ref.index.isin(X.index)]
tumor_content = tumor_content[tumor_content.index.isin(X.index)]
X.sort_index(inplace=True)

genes_to_keep = ref.index.isin(X.index)
X = np.array(X)
X = np.delete(X, 4, axis=1)

ref.sort_index(inplace=True)
ref = np.array(ref)

tumor_content.sort_index(inplace=True)
tumor_content = np.array(tumor_content)
tumor_content = np.mean(tumor_content, axis=1)


phenotype = np.array(["Jurkat", "Jurkat", "Jurkat", "IM-9", "IM-9", "IM-9", "Raji", "Raji", "Raji", "THP1", "THP1", "THP1"])
n_try = 25
size_grid = 30


# Predefined grid for noise level and tumor content
sigmas = np.linspace(0, 11.6, num=size_grid)
tumor_level = np.linspace(0, 1.0, num=size_grid)

methods = ["SOLS", "cibersort", "ssvr"]

def fun_to_run(method):
# Initialization of the performance matrices

    RMSE = np.zeros((size_grid, size_grid))
    MAE = np.zeros((size_grid, size_grid))
    Corr = np.zeros((size_grid, size_grid))

    # seed the process for repeatability
    rng = check_random_state(42)

    data, ground_truth = semi_synth_data(ref, n_try, phenotype, sigma = 0.0, noise_type="log_gaussian")

    for i in range(size_grid):
        # Spiked in tumor content
        data_tumor = (1 - tumor_level[i]) * data + tumor_level[i] * np.array([tumor_content,]*n_try).transpose()
        for j in range(size_grid):
            # noise in the data
            if sigmas[j] != 0:
                data_noised = data_tumor + 2 ** (rng.randn(*data_tumor.shape) * sigmas[j])
            else:
                data_noised = data_tumor
            estimated = dict_method[method](X, data_noised)

            RMSE[i, j] = np.mean(rmse(estimated, ground_truth.T))
            MAE[i, j] = np.mean(mae(estimated, ground_truth.T))
            Corr[i, j] = np.corrcoef(estimated.flatten(), ground_truth.T.flatten())[0, 1]


    np.save('results/RMSE_%s.npy' %method, RMSE)
    np.save('results/MAE_%s.npy' %method, MAE)
    np.save('results/Corr_%s.npy' %method, Corr)

print("Enter sequential")
n_jobs=1
backend = 'loky'
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(fun_to_run)(
        method)
    for method in methods)
print('OK finished parallel')