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


from sparse_ho.utils_plot import configure_plt, plot_legend_apart
import seaborn as sns

configure_plt()
current_palette = sns.color_palette("colorblind")

list_methods = ["SSVR", "Cibersort", "SOLS"]

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def semi_synth_data(ref_samples, n_samples, sparse=False):
    n_cells = ref_samples.shape[1]

    if sparse:
        ground_truth = scipy.sparse.random(n_cells, n_samples, density=0.7, data_rvs=np.random.rand)
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)
    else:
        ground_truth = np.random.uniform(size=(n_cells, n_samples))
        ground_truth = ground_truth / np.sum(ground_truth, axis=0)

    data = np.zeros((ref_samples.shape[0], n_samples))
    data = ref_samples @ ground_truth

    return np.array(data), np.array(ground_truth)

def mad(x, y, rep=None):
    if rep is not None:
        res = np.zeros(rep)
        for i in range(rep):
            res[i] = np.mean(np.abs(x[i, :] - y[i, :]))
        
        return res
    else:
        return np.mean(np.abs(x - y))

def rmse(x, y, rep=None):
    if rep is not None:
        res = np.zeros(rep)
        for i in range(rep):
            res[i] = np.sqrt(np.mean((x[i, :] - y[i, :]) ** 2))
        
        return res
    else:
        return np.sqrt(np.mean((x - y) ** 2))


Sig = pd.read_csv("/Users/qklopfenstein/Documents/these/datasets/deconvolution/LM6/LM6.txt", header=0, index_col=0, delimiter="\t")
cells = Sig.columns

Sig.sort_index(inplace=True)
Sig = np.array(Sig)

percentage = np.linspace(0, 1, num=11)

# Log Normal Noise

res_ssvr = []
res_cibersort = [] 
res_SOLS = []

cells_ssvr = []
cells_cibersort = [] 
cells_SOLS = []
rep = 10
Mix, Weights = semi_synth_data(Sig, rep)

for percent in percentage:
    Mix_noised = 2 ** (np.log2(Mix + 1) + np.random.randn(*Mix.shape) * percent * 10)
    # Mix_noised = Mix + 2 ** (np.random.randn(*Mix.shape) * percent * 10)
    estimated_ssvr = deconv_ssvr(Sig, Mix_noised, sum_to_one=True)
    estimated_cibersort = cibersort(Sig, Mix_noised)
    estimated_SOLS = SOLS(Sig, Mix_noised)
    
    cells_temp_ssvr = []
    cells_temp_cibersort = []
    cells_temp_SOLS = []

    for j in range(cells.shape[0]):
        cells_temp_ssvr.append(rmse(Weights.T[:, j], estimated_ssvr[:, j]))
        cells_temp_cibersort.append(rmse(Weights.T[:, j], estimated_cibersort[:, j])) 
        cells_temp_SOLS.append(rmse(Weights.T[:, j], estimated_SOLS[:, j])) 

    cells_ssvr.append(cells_temp_ssvr)
    cells_cibersort.append(cells_temp_cibersort)
    cells_SOLS.append(cells_temp_SOLS)

    res_ssvr.append(rmse(Weights.T, estimated_ssvr, rep=5))
    res_cibersort.append(rmse(Weights.T, estimated_cibersort, rep=5))
    res_SOLS.append(rmse(Weights.T, estimated_SOLS, rep=5))


cells_mean_ssvr = np.mean(np.array(cells_ssvr), axis=0)
cells_mean_cibersort = np.mean(np.array(cells_cibersort), axis=0)
cells_mean_SOLS = np.mean(np.array(cells_SOLS), axis=0)
results = np.column_stack([cells_mean_ssvr, cells_mean_cibersort, cells_mean_SOLS])

medianprops = dict(color='black')

data = np.column_stack([np.array(res_ssvr).flatten(), np.array(res_cibersort).flatten(), np.array(res_SOLS).flatten()])
fig1, ax1 = plt.subplots(1,2, figsize=(20, 8))



ax1[0].set_title('Gaussian Noise')
ax1[0].set_xticklabels(list_methods)
ax1[0].set_ylabel("RMSE")
bplot1 = ax1[0].boxplot(data, patch_artist=True, medianprops=medianprops)

# fill with colors
colors = [current_palette[0], current_palette[1], current_palette[2]]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)


im, cbar = heatmap(results.T, list_methods, np.array(cells), ax=ax1[1],
                   cmap="PuOr", cbarlabel="RMSE")
texts = annotate_heatmap(im, valfmt="{x:.2f}", size=10, threshold=0.3)
fig1.tight_layout()
plt.savefig("simulated_rna_gaussian.pdf")
plt.show()
# ax1[1].imshow(Weights)

# plt.show()