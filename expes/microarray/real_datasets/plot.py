import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from deconv.utils import rmse, mae
import pylab


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


def my_corr(x, y, rep=None):
    if rep is not None:
        res = np.zeros(rep)
        for i in range(rep):
            res[i] = np.corrcoef(x[i, :], y[i, :])[0, 1]

        return res
    else:
        return np.corrcoef(x.flatten(), y.flatten())[0, 1]


configure_plt()
save_fig = True

fontsize = 16
medianprops = dict(color='black')

current_palette = sns.color_palette("colorblind")
plt.close('all')

list_competitors = ["ssvr", "cibersort", "sols", "fardeep", "hspe", "EPIC"]
list_datasets = ["abbas", "becht", "gong", "kuhn",
                 "newman_pbmc", "shen", "shi"]

df = pandas.read_pickle("Results_microarray_real.pkl")

fig, axarr = plt.subplots(
    1, 1, sharex=False, sharey=False, figsize=[15, 10])

path_to_dir = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
               "Data/microarray/")

dict_ground_truth = {}
dict_ground_truth["abbas"] = np.array(
    pandas.read_csv(path_to_dir + "abbas" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["becht"] = np.array(
    pandas.read_csv(path_to_dir + "becht" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["gong"] = np.array(
    pandas.read_csv(path_to_dir + "gong" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["kuhn"] = np.array(
    pandas.read_csv(path_to_dir + "kuhn" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["newman_fl"] = np.array(
    pandas.read_csv(path_to_dir + "newman_fl" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["newman_pbmc"] = np.array(
    pandas.read_csv(path_to_dir + "newman_pbmc" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["shen"] = np.array(
    pandas.read_csv(path_to_dir + "shen" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["shi"] = np.array(
    pandas.read_csv(path_to_dir + "shi" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))
dict_ground_truth["siegert"] = np.array(
    pandas.read_csv(path_to_dir + "siegert" + "/Ground_truth.txt",
                    header=0, index_col=0, delimiter="\t"))

dict_rmse = {}

for idx, competitor in enumerate(list_competitors):

    df_competitor = df[df['competitor'] == competitor]
    dict_rmse[competitor] = []
    for idx2, dataset in enumerate(list_datasets):
        print(competitor)
        df_dataset = df_competitor[df_competitor['dataset'] == dataset]
        estimated_prop = df_dataset['estimated_proportions']
        estimated_prop.index = [0]
        estimated_prop = estimated_prop[0]
        n_samples, n_cells = estimated_prop.shape

        dict_rmse[competitor].append(
            rmse(estimated_prop, dict_ground_truth[dataset], axis=1))


results = np.zeros((
    len(np.concatenate(dict_rmse["ssvr"], axis=0)), len(list_competitors)))
for idx, competitor in enumerate(list_competitors):
    results[:, idx] = np.concatenate(dict_rmse[competitor], axis=0)


list_methods = ["AC-SVR", 'Cibersort', 'SOLS', 'FARDEEP', "HSPE", "EPIC"]
axarr.set_xticklabels(list_methods)
axarr.set_ylabel("RMSE")

bplot1 = axarr.boxplot(results, patch_artist=True, medianprops=medianprops)

# fill with colors
colors = [current_palette[i] for i in range(len(list_methods))]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

list_markers = [".", "^", "s", "*", "D", "H", "P"]
for i, competitor in enumerate(list_competitors):
    for j in range(len(list_datasets)):
        y = dict_rmse[competitor][j]
        x = np.random.normal(0.5+i, 0.04, size=len(y))
        axarr.scatter(x, y, alpha=0.7, color=colors[i], marker=list_markers[j])

fig_dir = ("/Users/quentin.klopfenstein/Documents/deconvolution_SSVR/",
           "Manuscript/figures/")

fig.tight_layout()
if save_fig:
    fig.savefig(fig_dir + "real_microarray.pdf")
fig.show()

list_datasets_legend = ["Abbas", "Becht", "Gong", "Kuhn",
                        "Newman PBMC", "Shen", "Shi"]

handlelist = [plt.plot(
    [], marker=marker, ls="", color="black")[0] for marker in list_markers]
figlegend = pylab.figure(figsize=(15, 10))
figlegend.legend(handlelist, list_datasets_legend, loc='center', ncol=4)
figlegend.savefig(fig_dir + "legend_real_microarray.pdf")

# Supplementary Figures related to each microarray datasets

for i, dataset in enumerate(list_datasets):
    df_dataset = df[df['dataset'] == dataset]
    fig, axarr = plt.subplots(
        2, 2, sharex=False, sharey=False, figsize=[20, 20])

    dict_rmse = {}
    dict_mae = {}
    dict_corr = {}
    for idx2, competitor in enumerate(list_competitors):
        dict_rmse[competitor] = []
        dict_mae[competitor] = []
        dict_corr[competitor] = []

        df_competitor = df_dataset[df_dataset['competitor'] == competitor]
        estimated_prop = df_competitor['estimated_proportions']
        estimated_prop.index = [0]
        estimated_prop = estimated_prop[0]
        n_samples, n_cells = estimated_prop.shape

        dict_rmse[competitor].append(
            rmse(estimated_prop, dict_ground_truth[dataset], axis=1))
        dict_mae[competitor].append(
            mae(estimated_prop, dict_ground_truth[dataset], axis=1))
        dict_corr[competitor].append(
            my_corr(estimated_prop, dict_ground_truth[dataset], rep=n_samples))
    results_rmse = np.zeros((
        len(np.concatenate(dict_rmse["ssvr"], axis=0)), len(list_competitors)))
    results_mae = np.zeros((
        len(np.concatenate(dict_rmse["ssvr"], axis=0)), len(list_competitors)))
    results_corr = np.zeros((
        len(np.concatenate(dict_rmse["ssvr"], axis=0)), len(list_competitors)))

    for idx, competitor in enumerate(list_competitors):
        results_rmse[:, idx] = np.concatenate(dict_rmse[competitor], axis=0)
        results_mae[:, idx] = np.concatenate(dict_mae[competitor], axis=0)
        results_corr[:, idx] = np.concatenate(dict_corr[competitor], axis=0)

    list_methods = ["AC-SVR", 'Cibersort', 'SOLS', 'FARDEEP', "HSPE", "EPIC"]
    axarr[0, 0].set_xticklabels(list_methods)
    axarr[0, 0].set_ylabel("RMSE")
    bplot1 = axarr[0, 0].boxplot(
        results_rmse, patch_artist=True, medianprops=medianprops)

    axarr[0, 1].set_xticklabels(list_methods)
    axarr[0, 1].set_ylabel("Correlation")
    bplot2 = axarr[0, 1].boxplot(
        results_corr, patch_artist=True, medianprops=medianprops)

    axarr[1, 0].set_xticklabels(list_methods)
    axarr[1, 0].set_ylabel("mae")
    bplot3 = axarr[1, 0].boxplot(
        results_mae, patch_artist=True, medianprops=medianprops)

    colors = [current_palette[i] for i in range(len(list_methods))]

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    for j, competitor in enumerate(list_competitors):
        df_competitor = df_dataset[df_dataset['competitor'] == competitor]
        estimated_prop = df_competitor['estimated_proportions']
        estimated_prop.index = [0]
        estimated_prop = estimated_prop[0]
        axarr[1, 1].scatter(
            dict_ground_truth[dataset].flatten(), estimated_prop.flatten(),
            color=colors[j], alpha=0.5)
        axarr[1, 1].plot(
            (0, 1.0), (0, 1.0), color="black", marker='',
            linestyle='--', linewidth=1)
        axarr[1, 1].set_xlim((0, 1.0))
        axarr[1, 1].set_ylim((0, 1.0))
        axarr[1, 1].set_xlabel("Ground truth")
        axarr[1, 1].set_ylabel("Estimated")

    fig.suptitle("%s" % list_datasets_legend[i])
    fig.tight_layout()
    if save_fig:
        fig.savefig(fig_dir + "%s_supplementary.pdf" % dataset)
    fig.show()
