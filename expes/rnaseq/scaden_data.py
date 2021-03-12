import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from joblib import Parallel, delayed
from itertools import product
import pandas as pd
from deconv.deconv_methods import *
from deconv.utils import semi_synth_data, mae, rmse

X = pd.read_csv("cibersortX.txt", header=0, index_col=0, delimiter="\t")
cell_type = ["Mono", "NK", "CD8", "CD4", "B.cells"]
clustered_cells = X.columns

new_names = []
for i in range(len(clustered_cells)):
    for j in range(len(cell_type)):
        if cell_type[j] in clustered_cells[i]:
            new_names.append(cell_type[j])


