import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
from numba import njit

def drop_dissimilar_edges(features, adj, threshold=0.01, metric='similarity', undirected=False):
        """
        GCNJaccard Preprocessing:
        Drop dissimilar edges.(Faster version using numba)
        use:
        > modified_adj = self.drop_dissimilar_edges(features, adj)
        """
        # this causes processed adj matrix to be sym
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
       
        if undirected:
              adj_triu = adj.copy()                                                      # modified for directed graph
        else:
              adj_triu = sp.triu(adj, format='csr')

        if sp.issparse(features):
            features = features.todense().A # make it easier for njit processing

        removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
        print('removed %s edges in the original graph' % removed_cnt)

        if undirected:
              modified_adj = adj_triu                                                    # modified for directed graph    
        else:
              modified_adj = adj_triu + adj_triu.transpose() 
              
        return modified_adj
  
@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def truncatedSVD(data, k=50):
        """
        GCNSVD Preprocessing
        use: 
        > modified_adj = self.truncatedSVD(adj, k=k)
        
        Truncated SVD on input data.

        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.

        Returns
        -------
        numpy.array
            reconstructed matrix.
        """
        print('=== GCN-SVD: rank={} ==='.format(k))
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            print("rank_after = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            print("rank_before = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
            print("rank_after = {}".format(len(diag_S.nonzero()[0])))

        return U @ diag_S @ V