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

X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/LM22.txt", header=0, index_col=0, delimiter="\t")
# sig_lvl3 = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/sig_lvl3.txt", header=0, index_col=0, delimiter="\t")

cells = X.columns
# cells_lvl3 = sig_lvl3.columns

Y = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/mixture_corrected.txt",header=0,index_col=0, delimiter="\t")

X = X[X.index.isin(Y.index)]
X.sort_index(inplace=True)
Y.sort_index(inplace=True)
genes_to_keep = Y.index.isin(X.index)
X = np.array(X)
Y = np.array(Y)
Y = Y[genes_to_keep, :]

estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)
estimated_ssvr = deconv_ssvr(X, Y, sum_to_one=True)

cibersort = np.array([estimated_cibersort[:, 0], estimated_cibersort[:, 1], estimated_cibersort[:, 3], estimated_cibersort[:, 4],
estimated_cibersort[:, 5], estimated_cibersort[:, 6], estimated_cibersort[:, 9], estimated_cibersort[:, 10] + estimated_cibersort[:, 11], estimated_cibersort[:, 12]])

SOLS = np.array([estimated_SOLS[:, 0], estimated_SOLS[:, 1], estimated_SOLS[:, 3], estimated_SOLS[:, 4],
estimated_SOLS[:, 5], estimated_SOLS[:, 6], estimated_SOLS[:, 9], estimated_SOLS[:, 10] + estimated_SOLS[:, 11], estimated_SOLS[:, 12]])
ssvr = np.array([estimated_ssvr[:, 0], estimated_ssvr[:, 1], estimated_ssvr[:, 3], estimated_ssvr[:, 4],
estimated_ssvr[:, 5], estimated_ssvr[:, 6], estimated_ssvr[:, 9], estimated_ssvr[:, 10] + estimated_ssvr[:, 11], estimated_ssvr[:, 12]])




# cibersort =  cibersort.T / np.sum(cibersort, axis=1)
# cibersort = cibersort.T

# ssvr =  ssvr.T / np.sum(ssvr, axis=1)
# ssvr = ssvr.T

# SOLS =  SOLS.T / np.sum(SOLS, axis=1)
# SOLS = SOLS.T

np.save("cibersort.npy", cibersort)
np.save("ssvr.npy", ssvr)
np.save("SOLS.npy", SOLS)