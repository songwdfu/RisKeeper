import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import multiprocessing as mp
import sys
from time import time


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    c_adj = adj.toarray()
    adj = np.triu(c_adj, 1)
    adj = adj + adj.T
    adj=sp.coo_matrix(adj)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    #return adj, features, labels
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders, train=False):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    if train:
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))}) # if attack: do not feed in support
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def bisection(a,eps,xi=1e-5,ub=1):
    #mu=(np.sum(a)-eps)/(a.shape[1]*(a.shape[1]-1)/2)
    #upper_S_update = a - mu
    pa = np.clip(a, 0, ub)
    slack = False
    if np.sum(pa) <= eps:                                               # if 1'P(a) < eps, mu = 0
        # print('np.sum(pa) <= eps !!!!')
        slack = True
        upper_S_update = pa
    else:                                                               # if 1'P(a - mu 1) = eps, mu > 0 (a exceeding constraint)
        mu_l = np.min(a-1)                                              # bisection from min(a) - 1 to max(a)
        mu_u = np.max(a)
        mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l)>xi:
            #print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l)/2
            gu = np.sum(np.clip(a-mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a-mu_l, 0, ub)) - eps
            #print('gu:',gu)
            if gu == 0:
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
        upper_S_update = np.clip(a-mu_a, 0, ub)                         # return s = PI(a)

    return upper_S_update, slack

def bisection_cost(a, eps, c, xi=1e-8, ub=1):
    """
    modified PGD subject to total attack cost constraint
    c is the node cost vector with shape (number_of_nodes, )
    """
    shape = a.shape
    c = (c + c.T).ravel()
    a = a.ravel()
    pa = np.clip(a, 0, ub)
    slack = False
    if np.sum(c) <= eps:
        upper_S_update = pa                                              # cost constraint not effective
        slack = True
    elif np.matmul(c, pa) <= eps:
        upper_S_update = pa                                              # if c'P(a) < eps, mu=0
        slack = True
    else:
        c_nonzero = c[c!=0]                                              # if c'P(a - mu*c) = eps, mu>0
        mu_l = np.min(a-1) / np.max(c_nonzero)                           # bisection over mu \in [(amin-1) / cmax, amax / cmin]
        mu_u = np.max(a) / np.min(c_nonzero)
        mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l)>xi:
            # print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l)/2
            gu = np.matmul(c, (np.clip(a-mu_a*c, 0, ub))) - eps
            gu_l = np.matmul(c, (np.clip(a-mu_l*c, 0, ub))) - eps
            #print('gu:',gu)
            if gu == 0:
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
        upper_S_update = np.clip(a - mu_a*c, 0, ub)
    print(f'now_cost:{np.matmul(c, upper_S_update)}, constraint: {eps}')
    upper_S_update = upper_S_update.reshape(shape)
    
    return upper_S_update, slack


def HW_PGD_update(a, eps_count, eps_cost, node_cost, xi=1e-5, ub=1, delta=1e-3):
    """
    Halpern-Wittmann Algorithm for PGD under multiple convex constraints
    B. Halpern, Fixed points of nonexpanding maps, Bulletin of the AMS 73 (1967), 957 961.
    x_{k+1} = \frac{1}{k}a + \frac{k-1}{k} P_2(P_1(x_k))) for k = 0, 1, ..., where x_0 = a
    converges to projection of a onto C1 n C2
    """
    # init
    x = a
    x, slk = bisection(x, eps_count, xi, ub)                                 # first iteration, x_{k+1} = \frac{1}{k}a + \frac{k-1}{k} P_2(P_1(x_k)))
    x, slk_cost = bisection_cost(x, eps_cost, node_cost, xi, ub)             # if a constraint is slack, stop projecting
    x_new = 0.5*a + 0.5*x
    if slk_cost == True:
        upper_S_update = np.clip(x_new, 0, ub)
        return upper_S_update
    
    # cyclical projection
    k = 1
    while np.abs(x_new - x).sum() > delta:
        k += 1
        x = x_new
        # projection onto C1 (count constraint)
        x, slk = bisection(x, eps_count, xi, ub)
        if slk == True:
            upper_S_update = np.clip(x, 0, ub)
            return upper_S_update
        # projection onto C2 (cost constraint)
        x, slk_cost = bisection_cost(x, eps_count, node_cost, xi, ub)
        if slk_cost == True:
            upper_S_update = np.clip(x, 0, ub)
            return upper_S_update
        x_new = 1/k * a + (k-1)/k * x
        
    upper_S_update = np.clip(x_new, 0, ub)
    return upper_S_update


def HW_PGD_update_para(a, eps_count, eps_cost, node_cost, xi=1e-5, ub=1, delta=1e-2):
    """
    Parallel Version of Halpern-Wittmann Algorithm for PGD under multiple convex constraints
    B. Halpern, Fixed points of nonexpanding maps, Bulletin of the AMS 73 (1967), 957 961.
    x_{k+1} = \frac{1}{k}a + \frac{k-1}{k} P_2(P_1(x_k))) for k = 0, 1, ..., where x_0 = a
    converges to projection of a onto C1 n C2
    """
    x = a
    x1, slk = bisection(x, eps_count, xi, ub)                                 # first iteration, x_{k+1} = \frac{1}{k}a + \frac{k-1}{k} P_2(P_1(x_k)))
    x2, slk_cost = bisection_cost(x, eps_cost, node_cost, xi, ub)
    x_new = 0.5*a + 0.5*(x1 + x2)/2
    k = 1
    start_time = time()
    while (np.abs(x_new - x).sum() > delta):
        k += 1
        if k > 200: break
        x = x_new
        
        # Create a queue to store the return values
        result_queue = mp.Queue()

        # Create two processes, one for each function
        process1 = mp.Process(target=lambda q: q.put(bisection(x, eps_count, xi, ub)), args=(result_queue,))
        process2 = mp.Process(target=lambda q: q.put(bisection_cost(x, eps_cost, node_cost, xi, ub)), args=(result_queue,))

        # Start the processes
        process1.start()
        process2.start()

        # Get the return values from the queue
        x1, slk = result_queue.get()
        x2, slk_cost = result_queue.get()
        
        # Wait for the processes to finish
        process1.join()
        process2.join()

        # x1, slk = bisection(x, eps_count, xi, ub)
        # x2, slk_cost = bisection_cost(x, eps_cost, node_cost, xi, ub)
        if np.all(x1 == x2): return np.clip(x1, 0, ub)                                # if already in intersect, end alg
        # x_new = 1/k * a + (k-1)/k * (x1 + x2)/2                                     # original version
        x_new = 1/k * x + (k-1)/k * (x1 + x2)/2
    
    end_time = time()
    print(f'Projection finished in {end_time - start_time} after {k} iterations')
    upper_S_update = np.clip(x_new, 0, ub)
    return upper_S_update


def A_S_HW_PGD_update(a, eps_count, eps_cost, node_cost, xi=1e-5, ub=1, delta=1e-5):
    """
    Alternative Simultaneous Halpern-Wittmann Algorithm for PGD under multiple convex constraints
    arxiv.org/2304.09600
    """
    x = a
    k = 0
    d = x
    start_time = time()
    while k<500:                                               # update stops
        r = k//2
        d = x
        if k % 2 == 0:
            for i in range(r):
                x1, slk = bisection(x, eps_count, xi, ub)
                x = 1/r * d + (r-1)/r * x1
        else:
            for i in range(r):
                x2, slk_cost = bisection_cost(x, eps_cost, node_cost, xi, ub)
                x = 1/r * d + (r-1)/r * x2
        k += 1
        
    end_time = time()
    print(f'Projection finished in {end_time - start_time} after {k} iterations')
    upper_S_update = np.clip(x, 0, ub)
    return upper_S_update
                
def Dyk_PGD_update(a, eps_count, eps_cost, node_cost, xi=1e-5, ub=1, delta=1e-3):
    """Dykstra Alg for Projection onto intersection of two convex sets"""
    x_new = a
    k = 0                                                                   # debug
    x = np.zeros_like(x_new)
    p = np.zeros_like(x_new)
    q = np.zeros_like(x_new)
    start_time = time()
    while np.abs(x_new - x).sum() > delta:
        x = x_new
        y, _ = bisection(x+p, eps_count, xi, ub)
        p = x + p - y
        x_new, _ = bisection_cost(y+q, eps_cost, node_cost, xi, ub)
        q = y + q - x_new
        k += 1                                                              # debug
    
    end_time = time()
    print(f'Projection finished in {end_time - start_time} after {k} iterations')
    upper_S_update = np.clip(x_new, 0, ub)
    return upper_S_update
        
    
# def bisection_dual(a, eps_count, eps_cost, c, xi=1e-5, ub=1):
#     """modified PGD subject to both number of edge and total cost constrait"""
#     pa = np.clip(a, 0, ub)
#     if np.sum(c) <= eps_cost:                                           # if total cost constraint is not effective
#         return bisection(a, eps_count, xi, ub)
#     if (np.matmul(c, pa) <= eps_cost) and (np.sum(pa) <= eps_count):     # if both constraints are slack
#         upper_S_update = pa
#     if (np.matmul(c, pa) > eps_cost) and (np.sum(pa) <= eps_count):      # if only count constraint is slack, find lambda
#         upper_S_update = bisection_cost(a, eps_cost, c, xi, ub)
#     if (np.matmul(c, pa) <= eps_cost) and (np.sum(pa) > eps_count):      # if only total cost constraint is slack, find mu
#         upper_S_update = bisection(a, eps_count, xi, ub)
#     if (np.matmul(c, pa) > eps_cost) and (np.sum(pa) > eps_count):       # if both constraints are tight, 2-D bisection over lambda in [0, amin-1], mu in [0, (amin-1)/cmax]
#         c_max = np.max(c[c!=0])
#         mu_l, lam_l = 0, 0
#         mu_u = np.min(a-1) / c_max
#         lam_u = np.min(a-1)
#         mu_a, lam_a = (mu_u + mu_l)/2, (lam_u + lam_l)/2
#         while (np.abs(mu_u - mu_l) > xi) or (np.abs(lam_u - lam_l) > xi):
            
            
            
        
    return upper_S_update

def filter_potential_singletons(adj):
    """
    Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
    the entry have degree 1 and there is an edge between the two nodes.
    Returns
    -------
    tf.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
    where the returned tensor has value 0.
    """
    adj = np.squeeze(adj)
    N = adj.shape[-1]
    degrees = np.sum(adj, axis=0)
    degree_one = np.equal(degrees, 1)
    resh = np.reshape(np.tile(degree_one, [N]), [N, N])
    l_and = np.logical_and(resh, np.equal(adj, 1))
    logical_and_symmetric = np.logical_or(l_and, np.transpose(l_and)) 
    return logical_and_symmetric


