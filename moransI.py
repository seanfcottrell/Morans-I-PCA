import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

adata = sc.read_visium('DD073R_A1_processed')
adata.var_names_make_unique()

adata.obs['dummy'] = 'spot' # add a dummy

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, color = 'dummy')

# pre processing
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=4000)

# Morans I
def cal_Morans_I(x,W):
    N = np.sum(W)
    n = W.shape[0]
    x_dot_x = x.T @ x
    #I = Moran(x,W)
    I = (x.T @ W @ x) * (1/N) * (n / (x.T @ x + 1e-10))
    return I

# Z Score
# permutation test
def permute_moran(x, W):
    permuted_indexes = np.random.permutation(np.arange(len(x)))
    permuted_x = x[permuted_indexes]
    #moran = Moran(permuted_x, W)
    moran = cal_Morans_I(permuted_x,W)
    return moran

def cal_z_score(x, W):
    n = x.shape[0]
    #cal Morans I
    #I = Moran(x,W)
    I = cal_Morans_I(x,W)
    #E(I)
    E_I = -1/(n-1)
    #estimate E(I^2) by taking mean of I values over multiple Monte Carlo simulations
    permutations = 2000
    I_squares = [permute_moran(x, W)**2 for _ in range(permutations)]
    E_I2 = np.mean(I_squares)
    #cal V(I)
    V_I = E_I2 - E_I**2
    #cal z score
    z = (I - E_I) / V_I
    return z, I_squares

# Statistical testing
def cal_p_val(z):
    # we expect to find positive spatial auto correlation among genes
    # --> one-tailed test
    p_value = norm.cdf(z) if z < 0 else (1 - norm.cdf(z))
    if p_value < 0.05 and z > 0:
        print('Reject Null Hypothesis --> Statistically Significant (postive) Spatial Autocorrelation among this gene')
    if p_value < 0.05 and z < 0:
        print('Reject Null Hypothesis --> Statistically Significant (negative) Spatial Autocorrelation among this gene')
    if p_value >= 0.05:
        print('Cannot Reject Null Hypothesis --> Similar gene expression values possibly are randomly clustered')
    return p_value

# Filter the data for highly variable genes
adata = adata[:, adata.var['highly_variable']]

gene_names = adata.var_names

sorted_genes = adata.var.sort_values(by='dispersions_norm', ascending=False).index
# Print the top 15 variable genes
for gene in sorted_genes[:15]:
    print(gene)

# plot different genes of interest
gene_of_interest = 'PRH1' 
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, color=gene_of_interest,cmap = 'hot', size = 1.65)

import scipy
gene_of_interest = 'CCL21'
X_spatial = adata.obsm["spatial"]
x = adata[:, adata.var_names == gene_of_interest].X
if isinstance(x, scipy.sparse.spmatrix):
    x = x.toarray().flatten()
else:
    x = x.flatten()

G = cal_weighted_adj(X_spatial, 10)
# Morans I
I = cal_Morans_I(x,G)
print(f"Morans I = {I} for Gene {gene_of_interest}")
# z Score
z_score_I = cal_z_score(x, G)
z_score = z_score_I[0]
I_squares = z_score_I[1]
print(f"Z Score = {z_score}")
# Create a histogram
plt.hist(I_squares, bins=20, edgecolor='black')
plt.axvline(x=I, color='black', linestyle='--', label='Observed Morans I')

# Adding titles and labels
plt.title('Permutation Simulations')
plt.xlabel('Morans I')
plt.ylabel('Frequency')
plt.legend()
# Show the plot
plt.show()
#p values
p_val = cal_p_val(z_score)
print(f"P Value = {p_val}")
