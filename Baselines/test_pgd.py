import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
# sys.path.insert(0, os.path.abspath('../'))
# sys.path.append('/home/liaojunlong/GCFL/GCN_ADV_Train-master_old/DeepRobust')
# sys.path.append('/home/liaojunlong/GCFL/GCN_ADV_Train-master_old/DeepRobust/graph')
#from deeprobust.graph.defense import GCN
from DeepRobust.deeprobust.graph.defense import GCNJaccard,GCN,GCNSVD
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import pickle as pkl
import networkx as nx
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
def load_npz(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name,allow_pickle=True) as loader:
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
def my_load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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

    return adj, features, labels,idx_train,idx_val,idx_test,train_mask,val_mask,test_mask,y_train,y_val,y_test
    #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')

args = parser.parse_args()

# device = "cpu"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)



_A_obs, _X_obs, labels = load_npz('amazon_electronics_photo.npz')
print(_X_obs.shape)
_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
adj=_A_obs
features=_X_obs
# Load data
num_list=[i for i in range(adj.shape[0])]
idx_train=num_list[:int(0.1*len(num_list))]
idx_val=num_list[int(0.1*len(num_list)):int(0.2*len(num_list))]
idx_test=num_list[int(0.2*len(num_list)):]
new_y_train=np.zeros_like(labels)
new_y_val=np.zeros_like(labels)
new_y_test=np.zeros_like(labels)
new_y_train[:int(0.1*len(num_list))]=labels[num_list[:int(0.1*len(num_list))]]
new_y_val[int(0.1*len(num_list)):int(0.2*len(num_list))]=labels[num_list[int(0.1*len(num_list)):int(0.2*len(num_list))]]
new_y_test[int(0.2*len(num_list)):]=labels[num_list[int(0.2*len(num_list)):]]
y_train=new_y_train
y_val=new_y_val
y_test=new_y_test
new_train_mask=np.zeros(adj.shape[0])
new_val_mask=np.zeros(adj.shape[0])
new_test_mask=np.zeros(adj.shape[0])
new_train_mask[:int(0.1*len(num_list))]=1
new_val_mask[int(0.1*len(num_list)):int(0.2*len(num_list))]=1
new_test_mask[int(0.2*len(num_list)):]=1
train_mask=new_train_mask
val_mask=new_val_mask
test_mask=new_test_mask

'''
#data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
#adj, features, labels = data.adj, data.features, data.labels
#print(adj,features,labels)
adj, features, labels,idx_train,idx_val,idx_test,train_mask,val_mask,test_mask,y_train,y_val,y_test= my_load_data('citeseer')
labels=[np.argmax(i) for i in labels]
# features = normalize_feature(features)

#idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#idx_train=torch.tensor([i for i in idx_train],dtype=torch.long)
idx_train=[i for i in idx_train]
idx_val=[i for i in idx_val]
idx_test=[i for i in idx_test]
'''
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels)
rows, cols = adj.shape

# Fill the diagonal elements with zeros
for i in range(min(rows, cols)):
    adj[i, i] = 0
# Setup Victim Model
victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=32,
        dropout=0.5, weight_decay=5e-4, device=device)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train,train_iters=400,idx_val=idx_val,patience=10, verbose=True)
victim_model.eval()
outs=victim_model.predict()
outs=outs.cpu().detach().numpy()
pred=np.array([np.argmax(i) for i in outs])
y_train_label=np.array([np.argmax(i) for i in y_train])
attack_label = pred *(1-train_mask)+ y_train_label * train_mask
# Setup Attack Model

model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, weight_decay=5e-4,device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train,train_iters=400,idx_val=idx_val, patience=10) # train without model picking
    gcn.eval()
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    #model.attack(features, adj, labels, idx_train, perturbations)
    #perturbations = int(args.ptb_rate * (adj.sum()//2))
    test(adj)
    model.attack(features, adj, attack_label, np.array(idx_train+idx_test), perturbations)
    print('=== testing GCN on original(clean) graph ===')  
    test(adj)
    print('=== testing GCN on modified graph ===')  
    modified_adj = model.modified_adj
    # modified_features = model.modified_features
    test(modified_adj.cpu().detach().numpy())
    #perturbations = int(0.1 * (adj.sum()//2))
    #model.attack(features, adj, attack_label, idx_train+idx_test, perturbations)
    #modified_adj = model.modified_adj
    #test(modified_adj)
    #perturbations = int(0.15 * (adj.sum()//2))
    #model.attack(features, adj, attack_label, idx_train+idx_test, perturbations)
    #modified_adj = model.modified_adj
    #test(modified_adj)
    #model.attack(features, adj, attack_label, idx_train+idx_test, perturbations)
    #modified_adj = model.modified_adj
    #test(modified_adj)


    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()

