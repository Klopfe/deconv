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
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri

X = pd.read_csv("~/deconvolution_SSVR/Data/microarray/abbas/Signature_tumor.txt", header=0, index_col=0, delimiter="\t")
ref = pd.read_csv("~/deconvolution_SSVR/Data/microarray/abbas/Reference_tumor.txt",header=0,index_col=0, delimiter="\t")

tumor_content = pd.read_csv("~/deconvolution_SSVR/Data/microarray/abbas/tumor_content.txt", header=0, index_col=0, delimiter="\t")
phenotype = pd.read_csv("~/deconvolution_SSVR/Data/microarray/abbas/Phenotype_tumor.txt", header=0, index_col=0, delimiter="\t")

ref = ref[ref.index.isin(tumor_content.index)]
tumor_content = tumor_content[tumor_content.index.isin(ref.index)]
ref.sort_index(inplace=True)
tumor_content.sort_index(inplace=True)

# ref = ref[ref.index.isin(X.index)]
# tumor_content = tumor_content[tumor_content.index.isin(X.index)]
X.sort_index(inplace=True)

genes_to_keep = ref.index.isin(X.index)
X_frame = X.copy()
X = np.array(X)

ref_frame = ref.copy()
ref = np.array(ref)
ref = np.delete(ref, [12, 13], axis=1)

tumor_content = np.array(tumor_content)
tumor_content = np.mean(tumor_content, axis=1)

cell_types = np.array(["Jurkat", "Jurkat", "Jurkat", "IM-9", "IM-9", "IM-9", "Raji", "Raji", "Raji", "THP1", "THP1", "THP1"])
n_try = 10
size_grid = 30

# Predefined grid for noise level and tumor content
sigmas = np.linspace(0, 11.6, num=size_grid)
tumor_level = np.linspace(0, 1, num=size_grid)

methods = ["EPIC"]
# methods = ["hspe", "EPIC"]
# methods = ["ssvr", "hspe", "EPIC"]


def fun_to_run(method):
# Initialization of the performance matrices

    RMSE = np.zeros((size_grid, size_grid))
    MAE = np.zeros((size_grid, size_grid))
    Corr = np.zeros((size_grid, size_grid))

    # seed the process for repeatability
    rng = check_random_state(1)

    data, ground_truth = semi_synth_data(ref, n_try, cell_types, sigma = 0.0, noise_type="log_gaussian")

    for i in range(size_grid):
        # Spiked in tumor content
        data_tumor = (1 - tumor_level[i]) * data + tumor_level[i] * np.array([tumor_content,]*n_try).transpose()
        for j in range(size_grid):
            # noise in the data
            if sigmas[j] != 0:
                data_noised = data_tumor + 2 ** (rng.randn(*data_tumor.shape) * sigmas[j])
            else:
                data_noised = data_tumor

            
            if method == "ssvr":
                data_noised = data_noised[genes_to_keep, :]
                estimated = deconv_ssvr(X, data_noised)
    
            elif method == "cibersort":
                data_noised = data_noised[genes_to_keep, :]
                estimated = cibersort(X, data_noised)
            
            elif method == "sols":
                data_noised = data_noised[genes_to_keep, :]
                estimated = SOLS(X, data_noised)

            elif method == "fardeep":
                data_noised = data_noised[genes_to_keep, :]
                robjects.r['setwd']("~/deconvolution_SSVR/Analysis/deconv/expes/microarray/")
                robjects.r.source("~/deconvolution_SSVR/Analysis/deconv/deconv/competitors/FARDEEP.R")
                numpy2ri.activate()
                results = robjects.r('fardeep_perso')(X, data_noised)
                estimated = np.array(results)
                numpy2ri.deactivate()

            elif method == "EPIC":
                data_noised_frame = pd.DataFrame(data_noised)
                data_noised_frame.index = ref_frame.index
                robjects.r['setwd']("~/deconvolution_SSVR/Analysis/deconv/expes/microarray/")
                robjects.r.source("~/deconvolution_SSVR/Analysis/deconv/deconv/competitors/EPIC.R")
                pandas2ri.activate()
                results = robjects.r('EPIC_perso')(ref_frame, X_frame, data_noised_frame, phenotype)
                estimated = np.array(results)
                pandas2ri.deactivate()

            elif method == "hspe":
                data_noised_frame = pd.DataFrame(data_noised)
                data_noised_frame.index = ref_frame.index
                robjects.r['setwd']("~/deconvolution_SSVR/Analysis/deconv/expes/microarray/")
                robjects.r.source("~/deconvolution_SSVR/Analysis/deconv/deconv/competitors/hspe.R")
                pandas2ri.activate()
                results = robjects.r('hspe_perso')(ref_frame, X_frame, data_noised_frame, phenotype)
                estimated = np.array(results)
                pandas2ri.deactivate()
            
            estimated = np.delete(estimated, 4, axis=1)
            estimated = estimated.T / np.sum(estimated, axis=1)
            estimated = estimated.T
            RMSE[i, j] = np.mean(rmse(estimated, ground_truth.T, axis=1))
            MAE[i, j] = np.mean(mae(estimated, ground_truth.T, axis=1))
            Corr[i, j] = np.corrcoef((estimated.flatten(), ground_truth.T.flatten()))[0, 1]

    np.save('results/RMSE_%s.npy' %method, RMSE)
    np.save('results/MAE_%s.npy' %method, MAE)
    np.save('results/Corr_%s.npy' %method, Corr)

print("Enter sequential")
n_jobs = len(methods)
backend = 'loky'
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(fun_to_run)(
        method)
    for method in methods)
print('OK finished parallel')