import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from DeepRobust.deeprobust.graph.defense import GCN, GCNJaccard, GCNSVD
import numpy as np
from torch_geometric.data import Data, DataLoader
from DeepRobust.deeprobust.graph.global_attack.topology_attack import PGDAttack, MinMax
from DeepRobust.deeprobust.graph.global_attack.mettack import Metattack
from deeprobust.graph.data import Dataset
from DeepRobust.deeprobust.graph.utils import *
import torch.nn.functional as F
import pandas as pd
import pickle as pkl
import argparse

SEED = 123 # 123
np.random.seed(SEED) 
torch.manual_seed(SEED) 
torch.cuda.manual_seed(SEED)
device = torch.device('cuda:1')

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
    ''' test on GCN '''
    new_adj = new_adj.cpu().numpy()
    if gcn is None:
        # adj = normalize_adj_tensor(adj)
        if model_name == GCNJaccard:
          gcn = model_name(attr.shape[1], 32, 2, binary_feature=args.binary_feature, device=device)
        else:
          gcn = model_name(attr.shape[1], 32, 2, device=device)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(attr, new_adj, labels, train_indices, train_iters=400, verbose=True, idx_val=None) # train with validation model picking
        gcn.eval()
        output = gcn.predict().cpu()
    else:
        gcn.eval()
        output = gcn.predict(attr, new_adj).cpu()

    loss_test = F.nll_loss(output[test_indices].detach(), torch.tensor(labels[test_indices], dtype=torch.long))
    acc_test = accuracy(output[test_indices].detach(), torch.tensor(labels[test_indices], dtype=torch.long))
    print(f"Test set results w/ perturb ratio {args.perturb_ratio}:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--model', type=str, default='GCN')
      parser.add_argument('--attack', type=str, default='PGDAttack')
      parser.add_argument('--dataset', type=str, default='alpha')
      parser.add_argument('--attack_loss_type', type=str, default='CW')
      parser.add_argument('--perturb_ratio', type=float, default=0.05)
      parser.add_argument('--binary_feature', type=str, default='False')
      args = parser.parse_args()
      args.binary_feature = True if args.binary_feature == 'True' else False
      
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
      attr = attr.real
      labels[labels==-1] = 0
      
      # build model
      if args.model == 'GCNJaccard':
            model_name = GCNJaccard
            model = model_name(attr.shape[1], 32, 2, device=device,binary_feature=args.binary_feature) # modified: False
      if args.model == 'GCN':
            model_name = GCN
            model = model_name(attr.shape[1], 32, 2, device=device)
      if args.model == 'GCNSVD':
            model_name = GCNSVD
            model = model_name(attr.shape[1], 32, 2, device=device)
      
      # fit model
      model = model.to(device)
      model.fit(features=attr, adj=adj, labels=labels, idx_train=train_indices, train_iters=134, verbose=True, idx_val=None) # val
      
      # test on clean graph
      model.eval()
      output = model.test(test_indices)
      new_pred = model.predict(attr, adj)
      # save predicted pseudo labels
      new_pred = torch.argmax(new_pred, 1).cpu()
      pkl.dump(new_pred.numpy(), open(f'pred_{args.model}_{args.dataset}.pkl', 'wb'))
      
      # Setup Attack Model
      print('=== setup attack model ===')
      if args.attack == 'PGDAttack':
          atk_name = PGDAttack
          atk_model = atk_name(model=model, nnodes=adj.shape[0], loss_type=args.attack_loss_type, device=device)
      if args.attack == 'MinMax':
          atk_name = MinMax
          atk_model = atk_name(model=model, nnodes=adj.shape[0], loss_type=args.attack_loss_type, device=device)
      if args.attack == 'Metattack':
          atk_name = Metattack
          atk_model = atk_name(model=model, nnodes=adj.shape[0], device=device, undirected=False, lambda_=0.5)

      atk_model = atk_model.to(device)

      atk_epochs = 250 # 250
      ptb_rate = args.perturb_ratio

      perturbations = int(ptb_rate * (adj.sum())//2)
    #   perturbations = int(ptb_rate * (adj.shape[0]//2))
      features = torch.from_numpy(attr)
      adj_tensor = torch.from_numpy(adj)

      pseudo_labels = model.predict(attr, adj)
      pseudo_labels = torch.argmax(pseudo_labels, 1).cpu()
      # Besides, we need to add the idx into the whole process
      pseudo_indices = np.concatenate([train_indices, test_indices])

      idx_others = list(set(np.arange(len(labels))) - set(train_indices))
      pseudo_labels = torch.cat([torch.from_numpy(labels[train_indices]), pseudo_labels[idx_others]])
      if args.attack == 'Metattack':
        # atk_model.attack(attr, adj, pseudo_labels.numpy(), pseudo_indices, val_indices, perturbations)  
        atk_model.attack(attr, adj, labels, train_indices, np.union1d(val_indices, test_indices), perturbations, ll_constraint=False)  
      else: # PGD attack
        output, loss, acc, adj_norm = atk_model.attack(attr, adj, pseudo_labels.numpy(), pseudo_indices, perturbations, epochs=atk_epochs)

      print('=== testing GCN on clean graph ===')
      test(torch.tensor(adj), gcn=model, model_name=None)
      
      modified_adj = atk_model.modified_adj
      pkl.dump(modified_adj.cpu().numpy(), open(f'meta_modified_adjs/{args.dataset}_{args.perturb_ratio}.pkl', 'wb'))
    #   modified_adj = pkl.load(open('modified_adj_otc_015.pkl','rb'))
    #   modified_adj = torch.from_numpy(modified_adj[0])
      
      print('=== testing GCN on Evasion attack ===')
      test(modified_adj, gcn=model, model_name=None)

      print('=== testing GCN on Poisoning attack ===')
      test(modified_adj, gcn=None, model_name=model_name)