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

X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/head_and_neck/signature.txt", header=0, index_col=0, delimiter="\t")
cells = X.columns
Y = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/head_and_neck/mixture.txt",header=0,index_col=0, delimiter="\t")
X.sort_index(inplace=True)
Y.sort_index(inplace=True)


X = X[X.index.isin(Y.index)]

genes_to_keep = Y.index.isin(X.index)
X = np.array(X)
Y = np.array(Y)
Y = Y[genes_to_keep, :]


estimated_ssvr = deconv_ssvr(X, Y, sum_to_one=False)
estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)



np.save("cibersort.npy", estimated_cibersort)
np.save("ssvr.npy", estimated_ssvr)
np.save("SOLS.npy", estimated_SOLS)