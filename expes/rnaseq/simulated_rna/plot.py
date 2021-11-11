import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from deconv.utils import mae


def configure_plt():
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette("colorblind")
    sns.set_context("poster")
    sns.set_style("ticks")


configure_plt()
fontsize = 48


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


def rmse(x, y, rep=None):
    if rep is not None:
        res = np.zeros(rep)
        for i in range(rep):
            res[i] = np.sqrt(np.mean((x[i, :] - y[i, :]) ** 2))

        return res
    else:
        return np.sqrt(np.mean((x - y) ** 2))


save_fig = True
fontsize = 16
medianprops = dict(color='black')

current_palette = sns.color_palette("colorblind")
plt.close('all')


list_methods = ["SSVR", "Cibersort", "SOLS",
                "FARDEEP", "EPIC", "hspe", "DeconRNAseq"]
list_labels = ["AC-SVR", 'Cibersort', 'SOLS',
               'FARDEEP', "HSPE", "EPIC", "DeconRNA"]

list_cells = ["B-cells", "CD8 T-cells", "CD4 T-cells",
              "NK cells", "Monocytes", "Neutrophilis"]
df = pandas.read_pickle("results_rnasimulated.pkl")

list_noise = ['Gaussian', "logGaussian"]
dict_noise = {}
dict_noise["Gaussian"] = "Gaussian noise"
dict_noise["logGaussian"] = "log-Gaussian noise"

dict_threshold = {}
dict_threshold["Gaussian"] = 0.3
dict_threshold["logGaussian"] = 0.17

fig, axarr = plt.subplots(
    2, 2, sharex=False, sharey=False, figsize=[20, 20])
for idx, noise in enumerate(list_noise):

    df_data = df[df['noise'] == noise]
    results = np.zeros((110, len(list_methods)))
    results_cells = np.zeros((len(list_cells), len(list_methods)))

    for idx2, method in enumerate(list_methods):
        df_method = df_data[df_data['method'] == method]
        temp = []
        temp2 = []
        for i in df_method.index:
            estimated = df_method['estimated'][i]
            temp.append(
                rmse(estimated, df_method['ground truth'][i].T, rep=10))
            temp2.append(
                rmse(estimated.T, df_method['ground truth'][i], rep=6))
        results[:, idx2] = np.array(temp).flatten()
        results_cells[:, idx2] = np.mean(np.array(temp2), axis=0)

    axarr[idx, 0].set_title(dict_noise[noise])
    axarr[idx, 0].set_xticklabels(list_labels)
    axarr[idx, 0].set_ylabel("RMSE")
    bplot1 = axarr[idx, 0].boxplot(
        results, patch_artist=True, medianprops=medianprops)

    # fill with colors
    colors = [current_palette[i] for i in range(len(list_methods))]

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    im, cbar = heatmap(
        results_cells.T, list_labels, np.array(list_cells), ax=axarr[idx, 1],
        cmap="PuOr", cbarlabel="RMSE")
    texts = annotate_heatmap(
        im, valfmt="{x:.2f}", size=16, threshold=dict_threshold[noise])
    # fig2.tight_layout()
    # fig2.show()

fig.tight_layout()

dir_to_save = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
               "Manuscript/figures/")
if save_fig:
    fig.savefig(dir_to_save + "simulated_rna_rmse.pdf")
fig.show()


fig2, axarr2 = plt.subplots(
    2, 2, sharex=False, sharey=False, figsize=[20, 20])


for idx, noise in enumerate(list_noise):

    df_data = df[df['noise'] == noise]
    results = np.zeros((110, len(list_methods)))
    results_cells = np.zeros((len(list_cells), len(list_methods)))

    for idx2, method in enumerate(list_methods):
        df_method = df_data[df_data['method'] == method]
        temp = []
        temp2 = []
        for i in df_method.index:
            estimated = df_method['estimated'][i]
            temp.append(mae(estimated, df_method['ground truth'][i].T, rep=10))
            temp2.append(mae(estimated.T, df_method['ground truth'][i], rep=6))
        results[:, idx2] = np.array(temp).flatten()
        results_cells[:, idx2] = np.mean(np.array(temp2), axis=0)

    axarr2[idx, 0].set_title(dict_noise[noise])
    axarr2[idx, 0].set_xticklabels(list_labels)
    axarr2[idx, 0].set_ylabel("MAE")
    bplot1 = axarr2[idx, 0].boxplot(
        results, patch_artist=True, medianprops=medianprops)

    # fill with colors
    colors = [current_palette[i] for i in range(len(list_methods))]

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    im, cbar = heatmap(
        results_cells.T, list_labels, np.array(list_cells), ax=axarr2[idx, 1],
        cmap="PuOr", cbarlabel="MAE")
    texts = annotate_heatmap(
        im, valfmt="{x:.2f}", size=16, threshold=dict_threshold[noise])
    # fig2.tight_layout()
    # fig2.show()

fig2.tight_layout()
if save_fig:
    fig2.savefig(dir_to_save + "simulated_rna_mae.pdf")
fig2.show()
