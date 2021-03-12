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
from deconv.utils import *

X = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE11103/GSE11103_X.txt", header=0, index_col=0, delimiter="\t")
ref = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/GSE11103/GSE11103_X_complete.txt",header=0,index_col=0, delimiter="\t")
ref = ref[ref.index.isin(X.index)]
X.sort_index(inplace=True)
X = np.array(X)
X = np.delete(X, 4, axis=1)
ref.sort_index(inplace=True)
phenotype = np.array(["Jurkat", "Jurkat", "Jurkat", "IM-9", "IM-9", "IM-9", "Raji", "Raji", "Raji", "THP1", "THP1", "THP1"])
n_try = 100
num = 30
ref = np.array(ref)

dict_method = {}
dict_method["cibersort"] = cibersort
dict_method["ssvr"] = deconv_ssvr
dict_method["SOLS"] = SOLS

list_noise = ["log_gaussian", "laplacian"]

dict_noise_level = {}
dict_noise_level["log_gaussian"] = np.linspace(0, 11.6, num=num)
dict_noise_level["laplacian"] = np.linspace(1, 30, num=num)


methods = ["cibersort", "SOLS", "ssvr"]

def increased_noise(method, noise_type='log_gaussian'):
# different noise level following the method in Newman et al. 2015
    list_rmse = []
    list_mae = []

    for i in range(len(dict_noise_level[noise_type])):
        rmse_temp = []
        mae_temp = []

        if noise_type == "laplacian":
            Y, ground_truth = semi_synth_data(ref, n_try, phenotype, sigma=None, snr=dict_noise_level[noise_type][i], noise_type='laplacian')
        elif noise_type == "log_gaussian":
            Y, ground_truth = semi_synth_data(ref, n_try, phenotype, sigma=dict_noise_level[noise_type][i], noise_type="log_gaussian")

        estimated = dict_method[method](X, Y)

        mae_temp = mae(estimated, ground_truth.T)
        rmse_temp = rmse(estimated, ground_truth.T)

        mae_temp = np.sort(mae_temp)
        rmse_temp = np.sort(rmse_temp)

        low_CI = int(n_try * 0.05)
        high_CI = int(n_try * 0.95)
        # low_CI = 0
        # high_CI = n_try - 1

        list_rmse.append(np.array([rmse_temp[low_CI], np.median(rmse_temp), rmse_temp[high_CI]]))
        list_mae.append(np.array([mae_temp[low_CI], np.median(mae_temp), mae_temp[high_CI]]))
    
    return (method, np.array(list_rmse), np.array(list_mae), dict_noise_level[noise_type], noise_type)


print("Enter sequential")
n_jobs=1
backend = 'loky'
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(increased_noise)(
        method, noise)
    for method, noise in product(methods, list_noise))
print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = [
    'method', 'rmse', 'mae', 'noise_level', 'noise_type']
df.to_pickle("res_simu_microarray.pkl" )
