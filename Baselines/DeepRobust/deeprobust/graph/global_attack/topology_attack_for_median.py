"""
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
    Tensorflow Implementation:
        https://github.com/KaidiXu/GCN_ADV_Train
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torch_geometric.data import Data

from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack


class PGDAttack_for_median(BaseAttack):
    """PGD attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import PGDAttack
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(PGDAttack_for_median, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, epochs=200, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        victim_model.eval()
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            # loss = F.cross_entropy(output[idx_train], labels[idx_train])
            acc = (torch.argmax(output[idx_train], 1) == labels[idx_train]).sum()/idx_train.shape[0]      # debugging
            loss = self._loss(output[idx_train], labels[idx_train])
            print(f'attack epoch {t+1}: loss= {loss}, acc= {acc}')                  # debugging
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        # debug: 
        acc = (torch.argmax(output[idx_train], 1) == labels[idx_train]).sum()/idx_train.shape[0]      # debugging
        loss = self._loss(output[idx_train], labels[idx_train])
        return output, loss, acc, adj_norm
        # self.check_adj_tensor(self.modified_adj)

    def process_pyg(self, features, adj, labels, idx_train, idx_val, idx_test):
        if type(features) != np.ndarray and features.device.type == 'cuda':
            features, adj, labels = features.cpu(), adj.cpu(), labels.cpu()
        # edge_index = torch.LongTensor(adj.nonzero()).T  # requires grad
        edge_index = torch.tensor(adj.nonzero()).T
        # by default, the features in pyg data is dense
        if sp.issparse(features):
            x = torch.FloatTensor(features.todense()).float()
        else:
            x = torch.FloatTensor(features).float()
        y = torch.LongTensor(labels)
        idx_train, idx_val, idx_test = idx_train, idx_val, idx_test
        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = utils.index_to_mask(idx_train, size=y.size(0))
        val_mask = utils.index_to_mask(idx_val, size=y.size(0))
        test_mask = utils.index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    
    def attack_cost(self, ori_features, ori_adj, labels, idx_train, cost_constraint, total_defense_cost=3373.167, epochs=200, hyper_c=1, **kwargs):
        """Generate perturbations on the input graph. wrt to the cost constraint

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """
        pyg_data = self.process_pyg(features=ori_features, adj=ori_adj, labels=labels, idx_train=idx_train, idx_val=None, idx_test=None)
        
        victim_model = self.surrogate
        # allocate costs evenly onto all nodes, alpha 3373.167, otc ...
        avg_cost = total_defense_cost / ori_adj.shape[0]
        edge_costs = torch.from_numpy(np.ones(ori_adj.shape[0]*(ori_adj.shape[0]-1)//2) * avg_cost * 2).to(self.device).float()
        # edge_costs = torch.from_numpy((node_costs + node_costs.T).ravel()).to(self.device).to(torch.float32)

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        pyg_data = self.process_pyg(features=ori_features, adj=ori_adj, labels=labels, idx_train=idx_train, idx_val=None, idx_test=None).to(self.device)
        victim_model.eval()
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            adj_norm.requires_grads = True
            # edge_index = torch.LongTensor(adj_norm.cpu().nonzero()).T
            edge_index = adj_norm.nonzero().T
            pyg_data.edge_index = edge_index.to(self.device)
            torch.cuda.empty_cache()
            output = victim_model(pyg_data)
            # loss = F.cross_entropy(output[idx_train], labels[idx_train])
            acc = (torch.argmax(output[idx_train], 1) == labels[idx_train]).sum()/idx_train.shape[0]      # debugging
            margin = self._loss(output[idx_train], labels[idx_train])
            cost_loss = hyper_c * torch.matmul(self.adj_changes, edge_costs)
            
            loss = margin + cost_loss # add cost loss
            
            print(f'attack epoch {t+1}: total loss= {loss}, margin= {margin}, cost_loss= {cost_loss} acc= {acc}')                  # debugging
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection_cost(cost_constraint, edge_costs)
            
            torch.cuda.empty_cache()

        self.random_sample_cost(ori_adj, ori_features, labels, idx_train, cost_constraint, edge_costs)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        # debug: 
        acc = (torch.argmax(output[idx_train], 1) == labels[idx_train]).sum()/idx_train.shape[0]      # debugging
        loss = self._loss(output[idx_train], labels[idx_train])
        return output, loss, acc, adj_norm
        # self.check_adj_tensor(self.modified_adj)

    def random_sample(self, ori_adj, ori_features, labels, idx_train, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                acc = (torch.argmax(output[idx_train], 1) == labels[idx_train]).sum()/idx_train.shape[0] # debug
                # loss = F.cross_entropy(output[idx_train], labels[idx_train])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
                    best_acc = acc  # debug
            self.adj_changes.data.copy_(torch.tensor(best_s))
            print(f'best_acc of resample: {best_acc}') # debug
            
    def random_sample_cost(self, ori_adj, ori_features, labels, idx_train, cost_constraint, edge_costs):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if np.matmul(sampled, edge_costs.cpu().numpy()) > cost_constraint:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                pyg_data = self.process_pyg(ori_features, adj_norm, labels, idx_train, None, None).to(self.device)
                output = victim_model(pyg_data)
                loss = self._loss(output[idx_train], labels[idx_train])
                acc = (torch.argmax(output[idx_train], 1) == labels[idx_train]).sum()/idx_train.shape[0] # debug
                # loss = F.cross_entropy(output[idx_train], labels[idx_train])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
                    best_acc = acc  # debug
            self.adj_changes.data.copy_(torch.tensor(best_s))
            print(f'best_acc of resample: {best_acc}') # debug

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.cross_entropy(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))
            
    def projection_cost(self, cost_constraint, edge_costs):
        """projection for total attack cost constraint"""
        if torch.matmul(edge_costs, torch.clamp(self.adj_changes, 0, 1)) > cost_constraint:
            c = edge_costs[np.nonzero(edge_costs)]
            left = (self.adj_changes-1).min() / c.max()
            right = self.adj_changes.max() / c.min()
            miu = self.bisection_cost(left, right, cost_constraint, edge_costs, epsilon=1e-4)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu * edge_costs, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj):

        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    def bisection_cost(self, a, b, cost_constraint, edge_costs, epsilon):
        def func(x):
            return torch.matmul(edge_costs, torch.clamp(self.adj_changes - x * edge_costs, 0, 1)) - cost_constraint

        miu = a
        while ((b-a) >= epsilon):
        # while np.abs(func(miu).detach().cpu()) > epsilon:
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
        
class MinMax(PGDAttack_for_median):
    """MinMax attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MinMax
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(MinMax, self).__init__(model, nnodes, loss_type, feature_shape, attack_structure, attack_features, device=device)


    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)

        # optimizer
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)

        epochs = 200
        victim_model.eval()
        for t in tqdm(range(epochs)):
            # update victim model
            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # generate pgd attack
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            # adj_grad = self.adj_changes.grad

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            # self.adj_changes.grad.zero_()
            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
