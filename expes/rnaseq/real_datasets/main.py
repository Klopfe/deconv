import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from deconv.deconv_methods import deconv_ssvr, SOLS
from deconv.utils import clean_mixture_signature
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri


# Give a list of datasets names for the experiment
list_datasets = ["linsley", "liu", "parsons", "PBMC", "racle"]

# Give a list of competitors to perform deconvolution
list_competitors = ["ssvr", "cibersort", "sols", "hspe", "EPIC", "deconrnaseq"]

path_to_dir = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
               "Data/rnaseq/")
dir_to_competitors = ("/Users/quentin.klopfenstein/Documents/",
                      "deconvolution_SSVR/Analysis/deconv/expes/rnaseq/")


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
    X, Y = clean_mixture_signature(mixture, signature)

    if competitor == "ssvr":
        estimated_prop = deconv_ssvr(X, Y)

    elif competitor == "cibersort":
        robjects.r['setwd'](path_to_dir)
        robjects.r.source(dir_to_competitors + "CIBERSORT.R")
        pandas2ri.activate()
        results = robjects.r('CIBERSORT')(signature, mixture)
        estimated_prop = np.array(results)
        pandas2ri.deactivate()

    elif competitor == "sols":
        estimated_prop = SOLS(X, Y)

    elif competitor == "fardeep":
        robjects.r['setwd'](path_to_dir)
        robjects.r.source(dir_to_competitors + "FARDEEP.R")
        numpy2ri.activate()
        results = robjects.r('fardeep_perso')(X, Y)
        estimated_prop = np.array(results)
        numpy2ri.deactivate()

    elif competitor == "deconrnaseq":
        robjects.r['setwd'](path_to_dir)
        robjects.r.source(dir_to_competitors + "DeconRNAseq.R")
        numpy2ri.activate()
        results = robjects.r('decon_perso')(X, Y)
        estimated_prop = np.array(results)
        numpy2ri.deactivate()

    elif competitor == "EPIC":
        robjects.r['setwd'](path_to_dir)
        robjects.r.source(dir_to_competitors + "EPIC.R")
        pandas2ri.activate()
        results = robjects.r('EPIC_perso')(
            reference, signature, mixture, phenotype)
        estimated_prop = np.array(results)
        pandas2ri.deactivate()

    elif competitor == "hspe":
        robjects.r['setwd'](path_to_dir)
        robjects.r.source(dir_to_competitors + "hspe.R")
        pandas2ri.activate()
        results = robjects.r('hspe_perso')(
            reference, signature, mixture, phenotype)
        estimated_prop = np.array(results)
        pandas2ri.deactivate()

    return(dataset, competitor, estimated_prop)


n_jobs = 4
print('Begin parallel')
results = Parallel(n_jobs=n_jobs, verbose=100, backend='loky')(
    delayed(parallel_function)(dataset, competitor)
    for dataset, competitor in product(list_datasets, list_competitors))
print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = ['dataset', 'competitor', 'estimated_proportions']

df.to_pickle("%s.pkl" % "Results_rnaseq_real")
