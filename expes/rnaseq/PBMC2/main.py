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

X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/PBMC2/Signature_scaden.txt", header=0, index_col=0, delimiter="\t")
cells = X.columns

Y = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/PBMC2/mixtures_PBMC2.csv",header=0,index_col=0)

X = X[X.index.isin(Y.index)]
X.sort_index(inplace=True)
Y.sort_index(inplace=True)
genes_to_keep = Y.index.isin(X.index)
X = np.array(X)
Y = np.array(Y)
Y = Y[genes_to_keep, :]

estimated_ssvr = deconv_ssvr(X, Y, sum_to_one=True)
estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)

# getting rid of the NKT cells that were not quantified in the FACS ground truth

# estimated_cibersort = np.delete(estimated_cibersort, 3, axis=1)
# estimated_SOLS = np.delete(estimated_SOLS, 3, axis=1)
# estimated_ssvr = np.delete(estimated_ssvr, 3, axis=1)

# estimated_cibersort = (estimated_cibersort.T / np.sum(estimated_cibersort, axis=1)).T
# estimated_SOLS = (estimated_SOLS.T / np.sum(estimated_SOLS, axis=1)).T
# estimated_ssvr = (estimated_ssvr.T / np.sum(estimated_ssvr, axis=1)).T

np.save("cibersort.npy", estimated_cibersort)
np.save("ssvr.npy", estimated_ssvr)
np.save("SOLS.npy", estimated_SOLS)