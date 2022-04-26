import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from deconv.deconv_methods import SOLS, deconv_ssvr, cibersort
from deconv.utils import clean_mixture_signature
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from deconv.utils import rmse, mae

# Give a list of datasets names for the experiment
list_datasets = ["abbas", "becht", "gong", "kuhn", "newman_fl",
                 "newman_pbmc", "shen", "shi"]
# Give a list of competitors to perform deconvolution
list_competitors = ["ssvr", "cibersort", "sols", "fardeep",  "hspe", "EPIC"]


path_to_dir = "~/deconvolution_SSVR/" + \
    "Data/microarray/"
dir_to_competitors = "~/deconvolution_SSVR" + \
    "/Analysis/deconv/deconv/competitors/"


def parallel_function(dataset, competitor):
    signature = pd.read_csv(
        path_to_dir + dataset + "/Signature.txt", header=0,
        index_col=0, delimiter="\t")
    mixture = pd.read_csv(
        path_to_dir + dataset + "/Mixture.txt", header=0,
        index_col=0, delimiter="\t")
    reference = pd.read_csv(
        path_to_dir + dataset + "/Reference.txt", header=0,
        index_col=0, delimiter="\t")
    phenotype = pd.read_csv(
        path_to_dir + dataset + "/Phenotype.txt", header=0,
        index_col=0, delimiter="\t")
    ground_truth = pd.read_csv(
        path_to_dir + dataset + "/Ground_truth.txt", header=0,
        index_col=0, delimiter="\t")

    X, Y = clean_mixture_signature(mixture, signature)
    M = np.linspace(10, X.shape[0], 15, dtype=int)
    repeat = 25

    results_rmse = np.zeros((len(M), repeat))
    results_mae = np.zeros((len(M), repeat))
    X_copy = X.copy()
    Y_copy = Y.copy()
    for i in range(len(M)):
        print('This is i:', i)
        for k in range(repeat):
            X = X_copy
            Y = Y_copy
            rstate = np.random.RandomState(k)
            idx = rstate.choice(X.shape[0], size=M[i], replace=False)
            X = X[idx, :]
            Y = Y[idx, :]
            if competitor == "ssvr":
                estimated_prop = deconv_ssvr(X, Y, sum_to_one=True)

            elif competitor == "cibersort":
                estimated_prop = cibersort(X, Y)
            elif competitor == "sols":
                estimated_prop = SOLS(X, Y)

            elif competitor == "fardeep":
                robjects.r['setwd'](path_to_dir)
                robjects.r.source(dir_to_competitors + "FARDEEP.R")
                pandas2ri.activate()
                results = robjects.r('fardeep_perso')(
                    signature.iloc[idx], mixture)
                estimated_prop = np.array(results)
                pandas2ri.deactivate()

            elif competitor == "EPIC":
                robjects.r['setwd'](path_to_dir)
                robjects.r.source(dir_to_competitors + "EPIC.R")
                pandas2ri.activate()
                results = robjects.r('EPIC_perso')(
                    signature.iloc[idx], signature.iloc[idx], mixture, phenotype)
                estimated_prop = np.array(results)
                pandas2ri.deactivate()

            elif competitor == "hspe":
                robjects.r['setwd'](path_to_dir)
                robjects.r.source(dir_to_competitors + "hspe.R")
                pandas2ri.activate()
                results = robjects.r('hspe_perso')(
                    reference, signature.iloc[idx], mixture, phenotype, markers=True)
                estimated_prop = np.array(results)
                pandas2ri.deactivate()

            results_rmse[i, k] = np.mean(rmse(estimated_prop, ground_truth, axis=1))
            results_mae[i, k] = np.mean(mae(estimated_prop, ground_truth, axis=1))

    return(dataset, competitor, M, results_rmse, results_mae)


n_jobs = 6
print('Begin parallel')
results = Parallel(n_jobs=n_jobs, verbose=100, backend='loky')(
    delayed(parallel_function)(dataset, competitor)
    for dataset, competitor in product(list_datasets, list_competitors))
print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = ['dataset', 'competitor', 'number genes', 'RMSE', 'MAE']

df.to_pickle("%s.pkl" % "results")
