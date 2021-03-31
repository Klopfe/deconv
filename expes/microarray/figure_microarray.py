import matplotlib.pyplot as plt
from numpy.linalg import norm
from sparse_ho.utils_plot import configure_plt
import numpy as np
import seaborn as sns
import pandas as pd

configure_plt()
current_palette = sns.color_palette("colorblind")

dict_color = {}
dict_color['ssvr'] = current_palette[0]
dict_color['SOLS'] = current_palette[1]
dict_color['cibersort'] = current_palette[2]

dict_method = {}
dict_method['ssvr'] = "Simplex SVR"
dict_method['SOLS'] = "Simplex Least Squares"
dict_method['cibersort'] = "Cibersort" 

df_data = pd.read_pickle("res_simu_microarray.pkl")
methods = ["cibersort", "SOLS", "ssvr"]

fig, axarr = plt.subplots(
    1, 1, sharex=False, sharey=False,
    figsize=[14, 4], constrained_layout=True)


# df_noise = df_data[df_data['noise_type'] == "log_gaussian"]
# for i, method in enumerate(methods):
#     df_temp = df_noise[df_noise['method'] == method]
#     import ipdb; ipdb.set_trace()
#     axarr[0].plot(df_temp['noise_level'][i * 2], df_temp['mae'][i * 2][:, 1], color=dict_color[method], marker='', label=dict_method[method])
#     axarr[0].fill_between(df_temp['noise_level'][i * 2], df_temp['mae'][i * 2][:, 0], df_temp['mae'][i * 2][:, 2], color=dict_color[method], alpha=.1)
# axarr[0].legend()

for i, method in enumerate(methods):
    df_temp = df_data[df_data['method'] == method]
    axarr.plot(df_temp['noise_level'][i], df_temp['mae'][i][:, 1], color=dict_color[method], marker='', label=dict_method[method])
    axarr.fill_between(df_temp['noise_level'][i], df_temp['mae'][i][:, 0], df_temp['mae'][i][:, 2], color=dict_color[method], alpha=.1)

axarr.legend()
fig.show()
