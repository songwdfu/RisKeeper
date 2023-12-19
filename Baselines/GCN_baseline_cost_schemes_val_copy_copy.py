import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from DeepRobust.deeprobust.graph.defense import GCN, GCN_softmax, GCN_Median, GCNJaccard, GCNSVD, MedianGCN, GCNJaccard_directed
from preprocess_utils import drop_dissimilar_edges, truncatedSVD
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataLoader
from DeepRobust.deeprobust.graph.global_attack.topology_attack_for_cost_scheme import PGDAttack, MinMax
from DeepRobust.deeprobust.graph.global_attack.mettack import Metattack
from deeprobust.graph.data import Dataset
from DeepRobust.deeprobust.graph.utils import *
import networkx as nx
import torch.nn.functional as F
import pandas as pd
import pickle as pkl
import argparse
from utils import load_data, preprocess_features, preprocess_adj, construct_feed_dict

def clustering_coefficient(adj_matrix):
    # Convert the adjacency matrix to a NetworkX graph
    G = nx.from_numpy_matrix(adj_matrix)

    # Calculate the clustering coefficient for each node
    clustering_coefficients = nx.clustering(G)

    # Convert the dictionary to a NumPy array
    num_nodes = len(adj_matrix)
    cc_array = np.zeros(num_nodes)
    for node, coefficient in clustering_coefficients.items():
        cc_array[node] = coefficient

    return cc_array

def load_npz(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def load_npz_raw(file_name):
    """
    for already processed bitcoin alpha network. See read_node_attr.ipynb 
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = loader['_A_obs']
        attr_matrix = loader['_X_obs']
        labels = loader.get('_z_obs')
    return adj_matrix, attr_matrix, labels

def test(new_adj, gcn=None, model_name=None):
    ''' test on GCN for preprocessing'''
    new_adj = new_adj.cpu().numpy()
    if gcn is None:
        # Poisoning setting
        # adj = normalize_adj_tensor(adj)
        if args.model == 'GCNJaccard' or args.model == 'GCNJaccard_directed':
          gcn = model_name(attr.shape[1], 32, nclass, binary_feature=args.binary_feature, device=device)
        else:
          gcn = model_name(attr.shape[1], 32, nclass, device=device)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(attr, new_adj, labels, train_indices, train_iters=400, verbose=False, idx_val=val_indices) # train with validation model picking
        gcn.eval()
        output = gcn.predict(attr, new_adj).cpu()
    else:
        # Evasion setting
        gcn.eval()
        if args.model == 'GCNJaccard' or args.model == 'GCNJaccard_directed':
            modified_adj = drop_dissimilar_edges(attr, new_adj, undirected=undirected).toarray() # control Jaccard directed or not here
            # _, modified_adj = to_tensor(attr, modified_adj, device=device)
        elif args.model == 'GCNSVD':
            modified_adj = truncatedSVD(new_adj)
            # _, modified_adj = to_tensor(attr, modified_adj, device=device)
        else:
            modified_adj = new_adj
        output = gcn.predict(attr, modified_adj).cpu()

    loss_test = F.cross_entropy(output[test_indices].detach(), torch.tensor(labels[test_indices], dtype=torch.long))
    acc_test = accuracy(output[test_indices].detach(), torch.tensor(labels[test_indices], dtype=torch.long))
    if nclass == 2:
        auc_score = roc_auc_score(labels[test_indices], output[test_indices, 1].cpu().detach().numpy(), average='micro')
    else:
        auc_score = roc_auc_score(labels_onehot[test_indices], output[test_indices].cpu().detach().numpy(), average='micro', multi_class='ovr')
    print(f"Test set results w/ constraint ratio {args.constraint_percentage}:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
        "auc= {:.4f}".format(auc_score))
    
    return acc_test, auc_score
    
def test_median(adj, features, gcn=None, model_name=None):
    """test on GCN for GCNMedian"""
    if gcn is None:
        # test on MedianGCN (poisoning attack)
        gcn = MedianGCN(nfeat=features.shape[1],
                        nhid=16,
                        nclass=nclass,
                        dropout=0.5, device=device)

        gcn = gcn.to(device)
        pyg_data = process_pyg(attr, adj, labels, train_indices, None, test_indices)
        gcn.fit([pyg_data], train_iters=400)
        gcn.eval()
        pyg_data = process_pyg(attr, adj, labels, test_indices, None, test_indices)
        output = gcn.predict([pyg_data])
    else:
        pyg_data = process_pyg(attr, adj, labels, test_indices, None, test_indices)
        # test on MedianGCN (evasion attack)
        gcn_median = GCN_Median(gcn.nfeat, gcn.hidden_sizes[0], gcn.nclass, device=device)
        gcn_median.load_state_dict(gcn.state_dict())
        gcn_median.to(device)
        output = gcn_median.predict(attr, adj.numpy())
  
    loss_test = F.cross_entropy(output[test_indices].detach().cpu(), torch.tensor(labels[test_indices], dtype=torch.long))
    acc_test = accuracy(output[test_indices].detach(), torch.tensor(labels[test_indices], dtype=torch.long))
    if nclass == 2:
        auc_test = roc_auc_score(labels[test_indices], output[test_indices, 1].cpu().detach().numpy(), average='micro')
    else:
        auc_test = roc_auc_score(labels_onehot[test_indices], output[test_indices].cpu().detach().numpy(), average='micro', multi_class='ovr')
        
    print(f"Test set results w/ constraint ratio {args.constraint_percentage}:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
        "auc= {:.4f}".format(auc_test.item()))
    
    return acc_test, auc_test

def process_pyg(features, adj, labels, idx_train, idx_val, idx_test):
    edge_index = torch.LongTensor(adj.nonzero()).T
    # by default, the features in pyg data is dense
    if sp.issparse(features):
        x = torch.FloatTensor(features.todense()).float()
    else:
        x = torch.FloatTensor(features).float()
    y = torch.LongTensor(labels)
    idx_train, idx_val, idx_test = idx_train, idx_val, idx_test
    data = Data(x=x, edge_index=edge_index, y=y)
    train_mask = index_to_mask(idx_train, size=y.size(0))
    val_mask = index_to_mask(idx_val, size=y.size(0))
    test_mask = index_to_mask(idx_test, size=y.size(0))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def bisection_beta(x, total_costs):
    f = lambda beta: np.exp(-beta * x).sum()
    dim = x.shape[0]
    left = 0
    right = -np.log(total_costs/dim) / np.clip(x.min(), 1e-5, None) + 100
    beta = (left + right) / 2 
    # while np.abs(f(beta)-total_costs) > 1e-8:
    while right-left > 1e-8:
        beta = (left + right) / 2 
        if (f(beta)-total_costs) * (f(left)-total_costs) > 0:
            left = beta
        else:
            right = beta
    return beta

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--model', type=str, default='GCN')
      parser.add_argument('--attack', type=str, default='PGDCost')
      parser.add_argument('--dataset', type=str, default='alpha')
      parser.add_argument('--attack_loss_type', type=str, default='CE')
      parser.add_argument('--constraint_percentage', type=float, default=0.05)
      parser.add_argument('--binary_feature', type=str, default='False')
      parser.add_argument('--cost_scheme', type=str, default='avg')
      parser.add_argument('--hyper_c', type=float, default=100.0)
      parser.add_argument('--device', type=int, default=1)
      args = parser.parse_args()
      args.binary_feature = True if args.binary_feature == 'True' else False
      
      SEED = 123 # 123
      np.random.seed(SEED) 
      torch.manual_seed(SEED) 
      torch.cuda.manual_seed(SEED)
      device = torch.device(f'cuda:{args.device}')
      
    #   try:
    #       result = pkl.load(open(f'results_new/result_newest_{args.model}_{args.attack}_{args.dataset}_{args.constraint_percentage}_{args.cost_scheme}_{args.hyper_c}.pkl', 'rb'))
    #       exit_ = True
    #       print('result found, exit..')
    #   except:
    #       exit_ = False
    #       pass
    #   if exit_: raise SystemExit
      
      if args.dataset == 'photo' or args.dataset == 'computers':
          if args.dataset=='computers':
              _A_obs, _X_obs, _z_obs = load_npz('amazon_electronics_computers.npz') # 13752 nodes, 767-dim node feature, 10 classes
          if args.dataset=='photo':
              _A_obs, _X_obs, _z_obs = load_npz('amazon_electronics_photo.npz')        
          _A_obs = _A_obs + _A_obs.T
          _A_obs[_A_obs > 1] = 1
          c_adj=_A_obs.toarray()
          c_adj=np.triu(c_adj,1)
          _A_obs=c_adj+c_adj.T
          adj=sp.coo_matrix(_A_obs).toarray()
          attr=_X_obs.toarray()
          labels = _z_obs
          enc = OneHotEncoder()
          _z_obs=np.expand_dims(_z_obs,1)
          _z_obs=enc.fit_transform(_z_obs)
          _z_obs=_z_obs.toarray()
          labels_onehot = _z_obs
          num_list=[i for i in range(adj.shape[0])]
          np.random.shuffle(num_list)
          new_y_train=np.zeros_like(_z_obs)
          new_y_val=np.zeros_like(_z_obs)
          new_y_test=np.zeros_like(_z_obs)
          new_y_train[:int(0.1*len(num_list))] = _z_obs[num_list[:int(0.1*len(num_list))]]                                               # deviding train val test set 1:1:8, no shuffle
          new_y_val[int(0.1*len(num_list)):int(0.2*len(num_list))] = _z_obs[num_list[int(0.1*len(num_list)):int(0.2*len(num_list))]]
          new_y_test[int(0.2*len(num_list)):] = _z_obs[num_list[int(0.2*len(num_list)):]]
          y_train=new_y_train
          y_val=new_y_val
          y_test=new_y_test
          new_train_mask=np.zeros(adj.shape[0])
          new_val_mask=np.zeros(adj.shape[0])
          new_test_mask=np.zeros(adj.shape[0])
          new_train_mask[num_list[:int(0.1*len(num_list))]]=1                                                                                    # corresponding masks
          new_val_mask[num_list[int(0.1*len(num_list)):int(0.2*len(num_list))]]=1
          new_test_mask[num_list[int(0.2*len(num_list)):]]=1
          train_mask=new_train_mask.astype(bool)
          val_mask=new_val_mask.astype(bool)
          test_mask=new_test_mask.astype(bool)

        #   train_indices = np.array(num_list[:int(0.1*len(num_list))])
        #   val_indices = np.array(num_list[int(0.1*len(num_list)):int(0.2*len(num_list))])
        #   test_indices = np.array(num_list[int(0.2*len(num_list)):])
          train_indices = np.nonzero(train_mask)[0]
          val_indices = np.nonzero(val_mask)[0]
          test_indices = np.nonzero(test_mask)[0]

      
      if args.dataset == 'cora' or args.dataset == 'citeseer':
        #   adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
            adj, attr, labels_onehot, train_indices, val_indices, test_indices = load_data(args.dataset)
            adj, attr = adj.toarray(), attr.toarray()
            labels = np.argmax(labels_onehot, 1)
            attr = normalize_feature(attr)
      
      if args.dataset == 'alpha' or args.dataset == 'otc':
          # load data
          adj, attr, labels = load_npz_raw(f"../GCN_ADV_Train/bitcoin_{args.dataset}_eigens.npz")
          # set up train and test mask, indices, labels
          train_mask = ~np.isnan(labels)
          train_indices = np.nonzero(train_mask)[0]
          np.random.shuffle(train_indices)
          val_indices = train_indices[:len(train_indices)//2]                         # random selection
          train_indices = train_indices[len(train_indices)//2:]
          train_mask = np.zeros_like(train_mask)
          val_mask = np.zeros_like(train_mask)
          train_mask[train_indices] = 1                                               # get train and val masks
          val_mask[val_indices] = 1                                               

          new_y_train = labels[train_mask]                                                # get train and val labels (-1 and 1)
          y_train = np.zeros([train_mask.shape[0], 2])
          y_train[train_mask, 0] = (new_y_train == -1).astype(int)
          y_train[train_mask, 1] = (new_y_train == 1).astype(int)

          new_y_val = labels[val_mask]
          y_val = np.zeros([train_mask.shape[0], 2])
          y_val[val_mask, 0] = (new_y_val == -1).astype(int)
          y_val[val_mask, 1] = (new_y_val == 1).astype(int)

          test_mask = np.isnan(labels)
          y_test = np.zeros([test_mask.shape[0], 2]).astype(int)

          test_mask, val_mask = val_mask, test_mask                                   ###############
          y_test, y_val = y_val, y_test  

          new_y_train[new_y_train==-1] = 0
          new_y_val[new_y_val==-1] = 0
          y_test = new_y_val
          y_train = new_y_train
          test_indices = val_indices
          val_indices = np.nonzero(val_mask)[0]

          adj_matrix = adj.copy()
          adj[adj!=0] = 1
          attr.real
          labels[labels==-1] = 0
      
      nclass = labels.max()+1 if labels.max() > 1 else 2
      
      # build model
      if args.model == 'GCNJaccard':
            model_name = GCNJaccard
            undirected = True
            # model = model_name(attr.shape[1], 32, 2, device=device,binary_feature=args.binary_feature) # modified: False
      if args.model == 'GCNJaccard_directed':
            model_name = GCNJaccard_directed
            undirected = False
      if args.model == 'GCN':
            model_name = GCN_softmax
            # model = model_name(attr.shape[1], 32, 2, device=device)
      if args.model == 'GCNSVD':
            model_name = GCNSVD
            # model = model_name(attr.shape[1], 32, 2, device=device)
      if args.model == 'MedianGCN':
            model_name = MedianGCN
            # model = model_name(attr.shape[1], 32, 2, device=device)
      
      SEED = 123 # 123
      np.random.seed(SEED) 
      torch.manual_seed(SEED) 
      torch.cuda.manual_seed(SEED)
      
      model = GCN_softmax(attr.shape[1], 32, nclass, device=device)
      model = model.to(device)
      if args.dataset == 'alpha' or args.dataset == 'otc':
         model.fit(features=attr, adj=adj, labels=labels, idx_train=train_indices, train_iters=400, verbose=True, idx_val=val_indices, patience=10) # vals
      else:
        # model.fit(features=attr, adj=adj, labels=labels, idx_train=train_indices, train_iters=400, verbose=True, idx_val=val_indices, patience=20) # val # tmp debug
        model.fit(features=attr, adj=adj, labels=labels, idx_train=train_indices, train_iters=400, verbose=True, idx_val=val_indices, patience=10) # vals
      
      # test on clean graph
      model.eval()
      acc_test = model.test(test_indices)
      new_pred = model.predict(attr, adj)
      # save predicted pseudo labels
      new_pred = torch.argmax(new_pred, 1).cpu()
    #   pkl.dump(new_pred.numpy(), open(f'pred_{args.model}_{args.dataset}.pkl', 'wb'))
      
      # Setup Attack Model
      print('=== setup attack model ===')
      if args.attack == 'PGDAttack':
          atk_name = PGDAttack
          atk_model = atk_name(model=model, nnodes=adj.shape[0], loss_type=args.attack_loss_type, device=device)
      if args.attack == 'PGDCost':
          atk_name = PGDAttack
          atk_model = atk_name(model=model, nnodes=adj.shape[0], loss_type=args.attack_loss_type, device=device)
      if args.attack == 'MinMax':
          atk_name = MinMax
          atk_model = atk_name(model=model, nnodes=adj.shape[0], loss_type=args.attack_loss_type, device=device)
      if args.attack == 'Metattack':
          atk_name = Metattack
          atk_model = atk_name(model=model, nnodes=adj.shape[0], device=device, undirected=True, lambda_=0.5)

      atk_model = atk_model.to(device)

      atk_epochs = 200 # 250
      
      cost_constraint = args.constraint_percentage * 100
      hyper_c = args.hyper_c
    #   hyper_c = 75
      
    #   perturbations = int(ptb_rate * (adj.shape[0]//2))
      features = torch.from_numpy(attr)
      adj_tensor = torch.from_numpy(adj)

      pseudo_labels = new_pred.cpu().numpy()
      pseudo_labels[train_indices] = labels[train_indices]
      pseudo_labels = torch.from_numpy(pseudo_labels)
      # Besides, we need to add the idx into the whole process
      pseudo_indices = np.concatenate([train_indices, test_indices])

      
    #   if args.attack == 'Metattack':
    #     # atk_model.attack(attr, adj, pseudo_labels.numpy(), pseudo_indices, val_indices, perturbations)  
    #     atk_model.attack(attr, adj, labels, train_indices, np.union1d(val_indices, test_indices), perturbations, ll_constraint=False)  
        
      if args.attack == 'PGDCost': # PGD attack with cost
        node_costs = pkl.load(open(f'/data/wenda/GCN_ADV_Train/node_costs_new/node_costs_{args.dataset}_{args.constraint_percentage}_{int(args.hyper_c)}.pkl', 'rb'))
        # node_costs /= 2
        # node_costs = np.zeros_like(node_cost s)   # temp, remove cost
        # node_costs = 0.5 * np.ones_like(node_costs)   # temp, remove cost
        # node_costs = pkl.load(open(f'/data/wenda/GCN_ADV_Train/bitcoin_data/node_costs_{args.dataset}_0.05_cc0.8.pkl', 'rb'))
        node_costs_original = node_costs
        total_node_costs = node_costs.sum()
        mean_node_cost = node_costs.mean()
        print(f'mean cost per node: {mean_node_cost}')
        
        if args.cost_scheme == 'raw':
            node_costs = np.zeros_like(node_costs)
        elif args.cost_scheme == 'random':
            cost_weights = np.random.rand(node_costs.shape[0])
            cost_weights /= cost_weights.sum()
            node_costs = total_node_costs * cost_weights
            node_costs = node_costs[:, np.newaxis]
        elif args.cost_scheme == 'avg':
            node_costs = mean_node_cost * np.ones_like(node_costs)
        elif args.cost_scheme == 'deg':
            node_degrees = adj.sum(1)
            node_degrees = np.exp(-node_degrees)
            node_degrees /= node_degrees.sum()
            node_costs = total_node_costs * node_degrees
            node_costs = node_costs[:, np.newaxis]
            node_costs[node_costs < 1e-5] = 0
        elif args.cost_scheme == 'deg_original':
            node_degrees = adj.sum(1)
            beta = bisection_beta(node_degrees, total_node_costs)
            node_costs = np.exp(-beta * node_degrees)
            node_costs = node_costs[:, np.newaxis]
            node_costs[node_costs < 1e-5] = 0    
        elif args.cost_scheme == 'clust_coef':
            coefficients = clustering_coefficient(adj)
            coefficients = np.exp(-coefficients)
            coefficients /= coefficients.sum()
            node_costs = total_node_costs * coefficients
            node_costs = node_costs[:, np.newaxis]
            node_costs[node_costs < 1e-5] = 0
        elif args.cost_scheme == 'clust_coef_original':
            coefficients = clustering_coefficient(adj)
            beta = bisection_beta(coefficients, total_node_costs)
            node_costs = np.exp(-beta * coefficients)
            node_costs = node_costs[:, np.newaxis]
            node_costs[node_costs < 1e-5] = 0
        elif args.cost_scheme == 'ours':
            pass
        else: raise ValueError('Cost Scheme Not Implemented')
        
        node_costs = node_costs.astype(np.float64)
            
        print(f'new node cost sum: {node_costs.sum()}')
        output, loss, acc, adj_norm = atk_model.attack_cost(attr, adj, pseudo_labels.numpy(), pseudo_indices, cost_constraint, node_costs, epochs=atk_epochs, hyper_c=hyper_c) # attack GCN
        
    #   else:
    #     output, loss, acc, adj_norm = atk_model.attack_cost(attr, adj, pseudo_labels.numpy(), pseudo_indices, perturbations , epochs=atk_epochs)
        
      result = dict()
      
      print(args.cost_scheme, args.dataset)
      
      
      print(f'# {args.dataset}, {args.cost_scheme}, {args.constraint_percentage}, perturbed {atk_model.adj_changes.sum()} edges')
      
      print('=== testing GCN on clean graph ===')
      clean_acc, clean_auc = test(torch.tensor(adj), gcn=model, model_name=None)
      
      modified_adj = atk_model.modified_adj
      # pkl.dump(modified_adj.cpu().numpy(), open(f'meta_modified_adjs/{args.dataset}_{args.perturb_ratio}.pkl', 'wb'))
    #   modified_adj = pkl.load(open('modified_adj_otc_015.pkl','rb'))
    #   modified_adj = torch.from_numpy(modified_adj[0])
      
      print('=== testing GCN on Evasion attack ===')
      if args.model=='MedianGCN':
        evasion_acc, evasion_auc = test_median(modified_adj.cpu(), attr, model, None)
      else:
        evasion_acc, evasion_auc = test(modified_adj, gcn=model, model_name=None)

      print('=== testing GCN on Poisoning attack ===')
      if args.model=='MedianGCN':
        poisoning_acc, poisoning_auc = test_median(modified_adj.cpu(), attr, None, model_name)
      else:
        poisoning_acc, poisoning_auc = test(modified_adj, gcn=None, model_name=model_name)
        
      result['clean_acc'] = clean_acc
      result['clean_auc'] = clean_auc
      result['evasion_acc'] = evasion_acc
      result['evasion_auc'] = evasion_auc
      result['poisoning_acc'] = poisoning_acc
      result['poisoning_auc'] = poisoning_auc
      
    #   pkl.dump(result, open(f'results_new/result_newest_{args.model}_{args.attack}_{args.dataset}_{args.constraint_percentage}_{args.cost_scheme}_{hyper_c}.pkl', 'wb'))