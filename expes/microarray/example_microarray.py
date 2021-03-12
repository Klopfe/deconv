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

X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE11103/GSE11103_X.txt", header=0, index_col=0, delimiter="\t")
ref = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE11103/GSE11103_X_complete.txt",header=0,index_col=0, delimiter="\t")
ref = ref[ref.index.isin(X.index)]
X.sort_index(inplace=True)
X = np.array(X)
X = np.delete(X, 4, axis=1)
ref.sort_index(inplace=True)
phenotype = np.array(["Jurkat", "Jurkat", "Jurkat", "IM-9", "IM-9", "IM-9", "Raji", "Raji", "Raji", "THP1", "THP1", "THP1"])
n_try = 100
ref = np.array(ref)

sigma = 8.5

# generate pseudo simulated data
Y, ground_truth = semi_synth_data(ref, n_try, phenotype, sigma = sigma, noise_type="log_gaussian")
estimated_ssvr = deconv_ssvr(X, Y)
estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)

np.save("estimated_cibersort.npy", estimated_cibersort)
np.save("estimated_ssvr.npy", estimated_ssvr)
np.save("estimated_SOLS.npy", estimated_SOLS)
np.save("ground_truth_simu.npy", ground_truth)


