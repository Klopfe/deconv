import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from numpy.linalg import norm

def rmse(X, Y, axis=1):
    if axis is None:
        return np.mean((X - Y) ** 2)
    else:
        return np.mean((X - Y) ** 2, axis=axis)

def mae(X, Y, axis=1):
    return np.mean(np.abs(X - Y), axis=axis)


def semi_synth_data(ref_samples, n_samples, phenotype, sigma, random_state=23,
    noise_type="log_gaussian", proportions=None, snr=None):
    rng = check_random_state(random_state)
    n_cells = len(np.unique(phenotype))

    if proportions is None:
        ground_truth = rng.uniform(size=(n_cells, n_samples))
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)
    else:
        ground_truth = np.tile(proportions, (n_samples, 1))
        ground_truth = ground_truth.T
    data = np.zeros((ref_samples.shape[0], n_samples))
    for i in range(n_cells):
        cell = pd.unique(phenotype)[i]
        ref_temp = ref_samples[:, phenotype==cell]
        local_ground_truth = rng.uniform(size=(np.sum(phenotype==cell), n_samples))
        local_ground_truth = local_ground_truth / np.sum(local_ground_truth, axis=0) * ground_truth[i, :]
        data = data + ref_temp @ local_ground_truth
    if noise_type == "log_gaussian":
        noise = rng.randn(*data.shape)
        data += 2 ** (noise * sigma)
    elif noise_type == "gaussian":
        sigma_star = np.sqrt(np.mean(data.T @ data) / (10 ** (snr / 10)))
        noise = rng.randn(*data.shape)
        data += (noise * sigma_star)
    elif noise_type == "laplacian":
        scale = np.sqrt(np.var(data) / (snr * 2))
        noise = rng.laplace(loc=0.0, scale=scale, size=((data.shape[0], data.shape[1])))
        data += noise
    return data, ground_truth