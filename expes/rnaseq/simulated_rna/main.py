import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from itertools import product
from deconv.deconv_methods import deconv_ssvr, SOLS
import scipy
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from joblib import Parallel, delayed


def semi_synth_data(ref_samples, n_samples, sparse=False, random_state=42):
    n_cells = ref_samples.shape[1]

    rng = check_random_state(random_state)

    if sparse:
        ground_truth = scipy.sparse.random(
            n_cells, n_samples, density=0.7, data_rvs=np.random.rand)
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)
    else:
        ground_truth = rng.uniform(size=(n_cells, n_samples))
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)

    data = np.zeros((ref_samples.shape[0], n_samples))
    data = ref_samples @ ground_truth

    return np.array(data), np.array(ground_truth)


path_to_sig = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
               "Data/rnaseq/LM6/LM6.txt")

working_dir = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
               "Analysis/deconv/expes/rnaseq/simulated_rna")

dir_competitors = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
                   "Analysis/deconv/deconv/competitors/")
Sig = pd.read_csv(path_to_sig, header=0, index_col=0, delimiter="\t")
cells = Sig.columns

Sig.sort_index(inplace=True)
Sig = np.array(Sig)

percentage = np.linspace(0, 1, num=11)


def simulate_rna(
        Sig, method, noise='Gaussian', percent=0.5, rep=2, random_state=42):
    rng = check_random_state(random_state)
    # Log Normal Noise

    Mix, Weights = semi_synth_data(Sig, rep, random_state=random_state)

    if noise == "Gaussian":
        Mix_noised = 2 ** (
            np.log2(Mix + 1) + rng.randn(*Mix.shape) * percent * rep)

    elif noise == "logGaussian":
        Mix_noised = Mix + 2 ** (rng.randn(*Mix.shape) * percent * rep)

    phenotype = np.ones((Sig.shape[1], Sig.shape[1])) + 1
    np.fill_diagonal(phenotype, 1)

    if method == "SSVR":
        estimated = deconv_ssvr(Sig, Mix_noised, sum_to_one=True)

    elif method == "Cibersort":
        Sig = pd.read_csv(path_to_sig, header=0, index_col=0, delimiter="\t")
        Sig.sort_index(inplace=True)
        Mix_noised_frame = pd.DataFrame(Mix_noised)
        Mix_noised_frame.index = Sig.index
        robjects.r['setwd'](working_dir)
        robjects.r.source(dir_competitors + "CIBERSORT.R")
        pandas2ri.activate()
        results = robjects.r('CIBERSORT')(Sig, Mix_noised_frame)
        estimated = np.array(results)
        pandas2ri.deactivate()
    elif method == "SOLS":
        estimated = SOLS(Sig, Mix_noised)

    elif method == "DeconRNAseq":
        robjects.r['setwd'](working_dir)
        robjects.r.source(dir_competitors + "DeconRNAseq.R")
        numpy2ri.activate()
        results = robjects.r('decon_perso')(Sig, Mix_noised)
        estimated = np.array(results)
        numpy2ri.deactivate()

    elif method == "FARDEEP":
        robjects.r['setwd'](working_dir)
        robjects.r.source(dir_competitors + "FARDEEP.R")
        numpy2ri.activate()
        results = robjects.r('fardeep_perso')(Sig, Mix_noised)
        estimated = np.array(results)
        numpy2ri.deactivate()

    elif method == "EPIC":
        Sig = pd.read_csv(path_to_sig, header=0, index_col=0, delimiter="\t")
        Sig.sort_index(inplace=True)
        Mix_noised_frame = pd.DataFrame(Mix_noised)
        Mix_noised_frame.index = Sig.index
        robjects.r['setwd'](working_dir)
        robjects.r.source(dir_competitors + "EPIC.R")
        pandas2ri.activate()
        results = robjects.r('EPIC_perso')(
            Sig, Sig, Mix_noised_frame, phenotype)
        estimated = np.array(results)
        pandas2ri.deactivate()

    elif method == "hspe":
        Sig = pd.read_csv(path_to_sig, header=0, index_col=0, delimiter="\t")
        Sig.sort_index(inplace=True)
        Mix_noised_frame = pd.DataFrame(Mix_noised)
        Mix_noised_frame.index = Sig.index
        robjects.r['setwd'](working_dir)
        robjects.r.source(dir_competitors + "hspe.R")
        pandas2ri.activate()
        results = robjects.r('hspe_perso')(
            Sig, Sig, Mix_noised_frame, phenotype)
        estimated = np.array(results)
        pandas2ri.deactivate()

    return(method, noise, estimated, percent, Weights)


n_jobs = 4
list_noise = ["Gaussian", "logGaussian"]
list_method = ["SSVR", "Cibersort", "SOLS",
               "DeconRNAseq", "FARDEEP", "EPIC", "hspe"]

print('Begin parallel')
results = Parallel(n_jobs=n_jobs, verbose=100, backend='loky')(
    delayed(simulate_rna)(
        Sig, method, noise, percent, rep=10, random_state=int(10 * percent))
    for method, noise, percent in product(list_method, list_noise, percentage))
print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = ['method', 'noise', 'estimated', 'percentage', 'ground truth']
df.to_pickle("%s.pkl" % "results_rnasimulated")
