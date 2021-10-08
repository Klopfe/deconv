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
# import scanpy

# X = pd.read_csv("path_to_scaden_signature.txt", header=0, index_col=0, delimiter="\t")
# anndata = scanpy.read_h5ad('/work/imb/qu2140kl/pbmc_data.h5ad')
# Y = pd.DataFrame(anndata.X.T, index=anndata.var.index)

# Y = Y[Y.index.isin(X.index)]
# X.sort_index(inplace=True)
# X = np.array(X)
# Y.sort_index(inplace=True)
# Y = np.array(Y)

# CD4, Mono, CD8, B, NK, unknown
X = np.load("X.npy")
Y = np.load("Y.npy")

ground_truth10 = pd.read_csv("ground_truth_10.txt", header=0, index_col=0, delimiter="\t")



# generate pseudo simulated data
estimated_ssvr = deconv_ssvr(X, Y, rnaseq=True)
estimated_cibersort = cibersort(X, Y)
estimated_SOLS = SOLS(X, Y)

np.save("estimated_cibersort.npy", estimated_cibersort)
np.save("estimated_ssvr.npy", estimated_ssvr)
np.save("estimated_SOLS.npy", estimated_SOLS)






# # Prepare file for signature construction in CibersortX
# X = pd.read_csv("cibersortX.txt", header=0, index_col=0, delimiter="\t")
# cell_type = ["Mono", "NK", "CD8", "CD4", "B.cells"]
# clustered_cells = X.columns

# new_names = []
# for i in range(len(clustered_cells)):
#     boolean = False
#     for j in range(len(cell_type)):
#         if cell_type[j] in clustered_cells[i]:
#             new_names.append(cell_type[j])
#             boolean = True
#     if boolean == False:
#         new_names.append("unknown")

# X.columns = new_names

# X.to_csv(r'true_cibersortX.txt', sep='\t')
# with open("phenotype.txt", "w") as output:
#     output.write(str(new_names))