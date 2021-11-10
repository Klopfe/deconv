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


X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/LM22/LM22.txt", header=0, index_col=0, delimiter="\t")
ref = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/LM22/LM22_ref.txt",header=0,index_col=0, delimiter="\t")
phenotype = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/LM22/LM22_phenotype.txt", header=None, index_col=0, delimiter="\t")

genes_to_keep = ref.index.isin(X.index)
cells = X.columns
cells = list(cells)

ref.sort_index(inplace=True)
ref = np.array(ref)
phenotype = np.array(phenotype)

pheno = []
for i in range(ref.shape[1]):
    if np.sum(phenotype[:, i] == 1.0) == 1:
        index = np.arange(0, len(cells))[phenotype[:, i] == 1.0]
        pheno.append(cells[int(index)])
    else:
        pheno.append("unknow")

pheno = np.array(pheno)
index = np.logical_not(pheno == "unknow")

ref = ref[:, index]
pheno = pheno[index]
n_try = 25


methods = ["SOLS", "cibersort", "ssvr"]
Y, ground_truth = semi_synth_data(ref, n_try, pheno, sigma = 2.0, noise_type="log_gaussian")


Y = Y[genes_to_keep, :]
X.sort_index(inplace=True)
X = np.array(X)


estimated_ssvr = deconv_ssvr(X, Y)
estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)