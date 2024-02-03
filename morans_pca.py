# Graph Structure
def gaussian_kernel(dist, t):
    '''
    gaussian kernel function for weighted edges
    '''
    return np.exp(-(dist**2 / t))

def Eu_dis(x):
    #print('Coords Shape:', x.shape)
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: number of samples
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.asarray(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x @ x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    dist_mat = np.asarray(dist_mat)
    return dist_mat

def cal_weighted_adj(data, n_neighbors):
    '''
    Calculate weighted adjacency matrix based on KNN
    For each row of X, put an edge between nodes i and j
    If nodes are among the n_neighbors nearest neighbors of each other
    according to Euclidean distance
    '''
    data = np.array(data)
    dist = Eu_dis(data)
    #print('Distances:',dist)
    #print('dist shape:', dist.shape)
    n = dist.shape[0]
    t = np.mean(dist**2)
    gk_dist = gaussian_kernel(dist, t)
    #print('GKs:', gk_dist)
    #print(gk_dist)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors] 
        len_index_L = len(index_L)
        #print(len_index_L)
        for j in range(len_index_L):
            #print(gk_dist[i, index_L[j]])
            W_L[i, index_L[j]] = gk_dist[i, index_L[j]] #weighted edges
            #print('entries:', np.sum(W_L, axis=1))
    W_L = np.maximum(W_L, W_L.T)
    #print('entries:', np.sum(W_L, axis=1))
    return W_L

def cal_laplace(adj):
    N = adj.shape[0]
    D = np.zeros_like(adj)
    for i in range(N):
        D[i, i] = np.sum(adj[i]) # Degree Matrix
    L = D - adj  # Laplacian
    return L

def get_mean_and_variance_rgb(histology_data, x, y):
    # Ensure the box is within the image boundaries
    start_x = max(x - 25, 0)
    end_x = min(x + 25, histology_data.shape[1] - 1)
    start_y = max(y - 25, 0)
    end_y = min(y + 25, histology_data.shape[0] - 1)

    # Extract the box
    box = histology_data[start_y:end_y+1, start_x:end_x+1]

    # Calculate the mean and variance for RGB values
    mean_red, var_red = np.mean(box[:, :, 0]), np.var(box[:, :, 0])
    mean_green, var_green = np.mean(box[:, :, 1]), np.var(box[:, :, 1])
    mean_blue, var_blue = np.mean(box[:, :, 2]), np.var(box[:, :, 2])
    
    mean_red = np.nan_to_num(mean_red, nan=1e-10)
    mean_green = np.nan_to_num(mean_green, nan=1e-10)
    mean_blue = np.nan_to_num(mean_blue, nan=1e-10)
    var_red = np.nan_to_num(var_red, nan=1e-10)
    var_green = np.nan_to_num(var_green, nan=1e-10)
    var_blue = np.nan_to_num(var_blue, nan=1e-10)

    return (mean_red, mean_green, mean_blue), (var_red, var_green, var_blue)


def hist_spatial_graph(spatial_coords, histology_data, num_neighbors, s):
    #define grid around pixel and take average / variance of RGB channel
    mean_red = []
    mean_blue = []
    mean_green = []
    variance_red = []
    variance_blue = []
    variance_green = []
    z_list = []
    for x, y in spatial_coords:
        means, variances = get_mean_and_variance_rgb(histology_data, int(x), int(y))
        mean_red.append(means[0])
        mean_green.append(means[1])
        mean_blue.append(means[2])
        variance_red.append(variances[0])
        variance_green.append(variances[1])
        variance_blue.append(variances[2])
    for i in range(len(mean_red)):
        total_variance = variance_red[i] + variance_green[i] + variance_blue[i]
        z = (mean_red[i] * variance_red[i] + mean_green[i] * variance_green[i] + mean_blue[i] * variance_blue[i]) / total_variance
        z_list.append(z)
    #rescale z feature
    mu_z = np.mean(z_list)
    sigma_z = np.std(z_list)
    sigma_x = np.std(spatial_coords[:, 0])
    sigma_y = np.std(spatial_coords[:, 1])
    sigma = max(sigma_y,sigma_x)
    for i in range(len(z_list)):
        z_list[i] = (z_list[i]-mu_z) / sigma_z * sigma * s
    #attach z features to spatial coords to make 3d coordinates
    z_list = np.asarray(z_list)
    z_list = z_list.reshape(-1,1)
    spatial_coords = np.concatenate((spatial_coords, z_list), axis=1)
    #distance calculation
    dist = Eu_dis(spatial_coords)
    #graph weighting
    n = dist.shape[0]
    t = np.mean(dist**2)
    gk_dist = gaussian_kernel(dist, t)
    #construct graph
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + num_neighbors] 
        len_index_L = len(index_L)
        for j in range(len_index_L):
            W_L[i, index_L[j]] = gk_dist[i, index_L[j]] #weighted edges
    W_L = np.maximum(W_L, W_L.T)
    return W_L

def sPCA_Algorithm(X, W, k, n):
    # Spatially weighted covariance matrix
    Z = (1 / (2*n) * X.T @ (W+W.T) @ X)
    #print(Z.shape)
    # cal Q (Projected Data Matrix)
    Z_eigVals, Z_eigVects = np.linalg.eig(np.asarray(Z))
    eigValIndice = np.argsort(Z_eigVals)[::-1]
    n_eigValIndice = eigValIndice[0:k]
    n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
    # Optimal Q given by eigenvectors corresponding
    # to largest k eigenvectors
    # We wish to emphasize global structure because we assume neighboring 
    # spots have positively correlated gene expression profiles 
    V = np.array(n_Z_eigVect)
    return V

def LPCA_Algorithm(X, W, k, n):
    # Spatially weighted covariance matrix
    Z = (1 / (2*n) * X.T @ (W) @ X)
    #print(Z.shape)
    # cal Q (Projected Data Matrix)
    Z_eigVals, Z_eigVects = np.linalg.eig(np.asarray(Z))
    eigValIndice = np.argsort(Z_eigVals)[::-1]
    n_eigValIndice = eigValIndice[0:k]
    n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
    # Optimal Q given by eigenvectors corresponding
    # to largest k eigenvectors
    # We wish to emphasize global structure because we assume neighboring 
    # spots have positively correlated gene expression profiles 
    V = np.array(n_Z_eigVect)
    return V

def cal_persistent_laplace(W_L, num_filtrations):
    n = W_L.shape[0]
    np.fill_diagonal(W_L,0)

    L = cal_laplace(W_L)
    #print("Laplace: ", L)

    np.fill_diagonal(L, 1e8) #Make sure diagonal is excluded from maximal and minimal value consideration
    min_l = np.min(L[np.nonzero(L)]) #Establish Min Value
    #print("min: ", min_l)
    np.fill_diagonal(L, -1e8)
    max_l = np.max(L[np.nonzero(L)]) #Establish Max Value
    #print("max: ", max_l)

    d = max_l - min_l
    #print("d: ", d)

    L = cal_laplace(W_L)
    #print('Laplace:', L)
    zetas = []
    for i in range(1,num_filtrations+2):
        zetas.append(1/i)
    zetas = np.asarray(zetas)
    #zetas = zetas[::-1]
    PL = np.zeros((num_filtrations+1,n,n))
    for k in range(1,num_filtrations+1):
        PL[k,:,:] = np.where(L < (k/num_filtrations*d + min_l), L, 0) 
        #PL[k,:,:] = np.where(L < (k/num_filtrations*d + min_l), 1, 0)
        #print('Filtered Adj:', PL[k,:,:])
        print("Threshold for Filtration ", k, ": ", k/num_filtrations*d + min_l)
        np.fill_diagonal(PL[k,:,:],0)
        #row_sums = np.sum(PL[k,:,:], axis=1)
        #np.fill_diagonal(PL[k,:,:], row_sums)
        PL[k,:,:] = cal_laplace(PL[k,:,:])
        #print(PL[k,:,:])

    P_L = np.sum(zetas[:, np.newaxis, np.newaxis] * PL, axis=0)*(-1)
    #P_L = np.sum(zetas[:, np.newaxis, np.newaxis] * PL, axis=0)
    #print('PL:', P_L)
    #P_L2 = np.copy(P_L)
    #np.fill_diagonal(P_L2,0)
    #P_L2 = -2*P_L2
    #P_L = P_L - P_L2
    return P_L

def cal_persistence(W_L, num_filtrations):
    n = W_L.shape[0]
    np.fill_diagonal(W_L,0)

    L = cal_laplace(W_L)
    #print("Laplace: ", L)

    np.fill_diagonal(L, 1e8) #Make sure diagonal is excluded from maximal and minimal value consideration
    min_l = np.min(L[np.nonzero(L)]) #Establish Min Value
    #print("min: ", min_l)
    np.fill_diagonal(L, -1e8)
    max_l = np.max(L[np.nonzero(L)]) #Establish Max Value
    #print("max: ", max_l)

    d = max_l - min_l
    #print("d: ", d)

    L = cal_laplace(W_L)
    #print('Laplace:', L)
    zetas = []
    for i in range(1,num_filtrations+2):
        zetas.append(1/i)
    zetas = np.asarray(zetas)
    #zetas = zetas[::-1]
    PL = np.zeros((num_filtrations+1,n,n))
    for k in range(1,num_filtrations+1):
        PL[k,:,:] = np.where(L < (k/num_filtrations*d + min_l), L, 0) 
        #PL[k,:,:] = np.where(L < (k/num_filtrations*d + min_l), 1, 0)
        #print('Filtered Adj:', PL[k,:,:])
        print("Threshold for Filtration ", k, ": ", k/num_filtrations*d + min_l)
        np.fill_diagonal(PL[k,:,:],0)
        #row_sums = np.sum(PL[k,:,:], axis=1)
        #np.fill_diagonal(PL[k,:,:], row_sums)
        #PL[k,:,:] = cal_laplace(PL[k,:,:])
        #print(PL[k,:,:])

    P_L = np.sum(zetas[:, np.newaxis, np.newaxis] * PL, axis=0)*(-1)
    #P_L = np.sum(zetas[:, np.newaxis, np.newaxis] * PL, axis=0)
    #print('PL:', P_L)
    #P_L2 = np.copy(P_L)
    #np.fill_diagonal(P_L2,0)
    #P_L2 = -2*P_L2
    #P_L = P_L - P_L2
    return P_L

def MHSA_PCA(X_spatial, X_gene, hist, k1, num_neighbors, num_filtrations, s, MultiScale, Laplacian):
    if MultiScale == False:
        print('--------------No Multiscale--------------')
        if Laplacian == False:
            n1 = X_gene.shape[0]  
            W = hist_spatial_graph(X_spatial, hist, num_neighbors, s)
            W = np.asarray(W)
            normalized_W = W / W.sum(axis=1, keepdims=True)
            np.fill_diagonal(normalized_W,1)
            #print(W.shape)
            print('------------tPCA----------------')
            V = LPCA_Algorithm(X_gene, normalized_W, k1, n1)
            #print(V.shape)
            print('------Performing Embedding------')
            Q = V.T @ X_gene.T
        if Laplacian == True:
            print('---------Laplacian Selected----------')
            n1 = X_gene.shape[0]  
            W = hist_spatial_graph(X_spatial, hist, num_neighbors, s)
            W = np.asarray(W)
            #normalized_W = W / W.sum(axis=1, keepdims=True)
            #np.fill_diagonal(normalized_W,1)
            #normalized_W = cal_laplace(normalized_W)
            L = cal_laplace(W)
            #print(W.shape)
            print('------------tPCA----------------')
            V = LPCA_Algorithm(X_gene, L, k1, n1)
            #print(V.shape)
            print('------Performing Embedding------')
            Q = V.T @ X_gene.T
    if MultiScale == True:
        print('--------Multiscale View Selected--------')
        if Laplacian == True:
            print('---------Laplacian Selected----------')
            n1 = X_gene.shape[0]  
            W = hist_spatial_graph(X_spatial, hist, num_neighbors, s)
            W = np.asarray(W)
            print('-------Persistent Laplacian-------')
            PL = cal_persistent_laplace(W, num_filtrations)
            print('------------tPCA----------------')
            V = LPCA_Algorithm(X_gene, PL, k1, n1)
            #print(V.shape)
            print('------Performing Embedding------')
            Q = V.T @ X_gene.T
        if Laplacian == False:
            n1 = X_gene.shape[0]  
            W = hist_spatial_graph(X_spatial, hist, num_neighbors, s)
            W = np.asarray(W)
            print('-----------Filtration-----------')
            PL = cal_persistence(W, num_filtrations)
            print('------------tPCA----------------')
            V = sPCA_Algorithm(X_gene, PL, k1, n1)
            #print(V.shape)
            print('------Performing Embedding------')
            Q = V.T @ X_gene.T
    print(Q.shape)
    return Q

import warnings
warnings.filterwarnings("ignore")
hist = adata.uns['spatial']['DD73RA1_rep2']['images']['hires']
X_gene = adata.X
print('-------------MHSA-PCA Louvain---------------')
Q = MHSA_PCA(X_spatial, X_gene, hist, 35, 350, 7, 1, True, True)
Q = np.asarray(Q)

# kNN graph
louvain_adata = sc.AnnData(np.real(Q).T)
sc.pp.neighbors(louvain_adata, n_neighbors=20)  

# Louvain Clustering
sc.tl.louvain(louvain_adata, resolution=0.75)
adata.obs['louvain'] = louvain_adata.obs['louvain'].values
print('---------DONE----------')
from matplotlib.colors import ListedColormap
plt.rcParams["figure.figsize"] = (8, 8)
colors = ['red', 'orange', 'olivedrab', 'darkgreen', 'cadetblue', 'lightblue', 'blue', 'purple']
custom_cmap = ListedColormap(colors)
sc.pl.spatial(adata, color='louvain', cmap = custom_cmap, save='spatial_plot_lung1.png', size=1.3)
