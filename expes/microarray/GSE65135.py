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
from deconv.utils import semi_synth_data, mae, rmse

X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/LM22.txt", header=0, index_col=0, delimiter="\t")
cells = X.columns
Y = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE65135/GSE65135_Y.txt",header=0,index_col=0, delimiter="\t")
Y = Y[Y.index.isin(X.index)]
X = X[X.index.isin(Y.index)]
X.sort_index(inplace=True)
Y.sort_index(inplace=True)
X = np.array(X)
Y = np.array(Y)


estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)
estimated_ssvr = deconv_ssvr(X, Y)

estimated_cibersort = np.array((np.sum(estimated_cibersort[:, 4:9], axis=1), estimated_cibersort[:, 3], estimated_cibersort[:, 0] + estimated_cibersort[:, 1], ))
estimated_ssvr = np.array((np.sum(estimated_ssvr[:, 4:9], axis=1), estimated_ssvr[:, 3], estimated_ssvr[:, 0] + estimated_ssvr[:, 1], ))
estimated_SOLS = np.array((np.sum(estimated_SOLS[:, 4:9], axis=1), estimated_SOLS[:, 3], estimated_SOLS[:, 0] + estimated_SOLS[:, 1], ))

np.save("estimated_cibersort_GSE65135.npy", estimated_cibersort)
np.save("estimated_ssvr_GSE65135.npy", estimated_ssvr)
np.save("estimated_SOLS_GSE65135.npy", estimated_SOLS)