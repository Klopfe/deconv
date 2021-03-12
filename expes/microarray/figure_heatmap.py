import matplotlib.pyplot as plt
from numpy.linalg import norm
from sparse_ho.utils_plot import configure_plt
import numpy as np
import seaborn as sns
import pandas as pd

configure_plt()
current_palette = sns.color_palette("colorblind")

RMSE = np.load('results/RMSE_cibersort.npy')


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(RMSE.T, cmap='jet_r', interpolation='gaussian', aspect='auto', origin="lower")
fig.colorbar(cax)
plt.ylabel('noise')
plt.xlabel('tumor content')
plt.rcParams["axes.grid"] = False
fig.show()

