import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import norm
from joblib import Parallel, delayed
from itertools import product
import pandas as pd
from deconv.deconv_methods import *
from deconv.utils import semi_synth_data, mae, rmse, quantile_normalize
import scipy
from sklearn.utils import check_random_state
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from sparse_ho.utils_plot import configure_plt, plot_legend_apart


def semi_synth_data(ref_samples, n_samples, sparse=False, random_state=42):
    n_cells = ref_samples.shape[1]
    
    rng = check_random_state(random_state)

    if sparse:
        ground_truth = scipy.sparse.random(n_cells, n_samples, density=0.7, data_rvs=np.random.rand)
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)
    else:
        ground_truth = rng.uniform(size=(n_cells, n_samples))
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)

    data = np.zeros((ref_samples.shape[0], n_samples))
    data = ref_samples @ ground_truth

    return np.array(data), np.array(ground_truth)


Sig = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/LM6/LM6.txt", header=0, index_col=0, delimiter="\t")
cells = Sig.columns

Sig.sort_index(inplace=True)
Sig = np.array(Sig)

percentage = np.linspace(0, 1, num=11)


def simulate_rna(Sig, method, noise='Gaussian', percent=0.5, rep=2, random_state=42):
    rng = check_random_state(random_state)
    # Log Normal Noise

    Mix, Weights = semi_synth_data(Sig, rep, random_state=random_state)

    if noise == "Gaussian":
        Mix_noised = 2 ** (np.log2(Mix + 1) + rng.randn(*Mix.shape) * percent * rep)
        
    elif noise == "logGaussian":
        Mix_noised = Mix + 2 ** (rng.randn(*Mix.shape) * percent * rep)

        
    if method == "SSVR":
        estimated = deconv_ssvr(Sig, Mix_noised, sum_to_one=True)
    elif method == "Cibersort":
        estimated = cibersort(Sig, Mix_noised)
    elif method == "SOLS":
        estimated = SOLS(Sig, Mix_noised)

    elif method == "DeconRNAseq":
        robjects.r['setwd']("/Users/qklopfenstein/Documents/these/deconv/deconv/expes/rnaseq/simulated_rna")
        robjects.r.source("/Users/qklopfenstein/Documents/these/deconv/deconv/expes/rnaseq/DeconRNAseq.R")
        numpy2ri.activate()
        results = robjects.r('decon_perso')(Mix_noised, Sig)
        estimated = np.array(results)
        numpy2ri.deactivate()
    
    elif method == "FARDEEP":
        robjects.r['setwd']("/Users/qklopfenstein/Documents/these/deconv/deconv/expes/rnaseq/simulated_rna")
        robjects.r.source("/Users/qklopfenstein/Documents/these/deconv/deconv/expes/rnaseq/FARDEEP.R")
        numpy2ri.activate()
        results = robjects.r('fardeep_perso')(Mix_noised, Sig)
        estimated = np.array(results)
        numpy2ri.deactivate()


    return(method, noise, estimated, percent, Weights)


n_jobs = 2
list_noise = ["Gaussian", "logGaussian"]
list_method = ["SSVR", "Cibersort", "SOLS", "DeconRNAseq", "FARDEEP"]
# list_method = ["FARDEEP"]

print('Begin parallel')
results = Parallel(n_jobs=n_jobs, verbose=100, backend='loky')(delayed(simulate_rna)(
        Sig, method, noise, percent, rep=10, random_state=int(10 * percent))
    for method, noise, percent in product(list_method, list_noise, percentage))
print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = ['method', 'noise', 'estimated', 'percentage', 'ground truth']
df.to_pickle("%s.pkl" % "results_rnasimulated")




        # cells_temp_ssvr = []
        # cells_temp_cibersort = []
        # cells_temp_SOLS = []

        # for j in range(cells.shape[0]):
        #     cells_temp_ssvr.append(rmse(Weights.T[:, j], estimated_ssvr[:, j]))
        #     cells_temp_cibersort.append(rmse(Weights.T[:, j], estimated_cibersort[:, j])) 
        #     cells_temp_SOLS.append(rmse(Weights.T[:, j], estimated_SOLS[:, j])) 

        # cells_ssvr.append(cells_temp_ssvr)
        # cells_cibersort.append(cells_temp_cibersort)
        # cells_SOLS.append(cells_temp_SOLS)

        # res_ssvr.append(rmse(Weights.T, estimated_ssvr, rep=5))
        # res_cibersort.append(rmse(Weights.T, estimated_cibersort, rep=5))
        # res_SOLS.append(rmse(Weights.T, estimated_SOLS, rep=5))
