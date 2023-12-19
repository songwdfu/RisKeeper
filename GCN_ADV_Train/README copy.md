# Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware Attacks
This is tf implementation of the adversarial training of RisKeeper\\
credit jlliao
## Environment 
To build env, run 
```
pip install -r requirements.txt
```
The original PGD was implemented on tf1, this is a modified version using tf 2.12.0, using `tensorflow.compat.v1`\\
If code fails to run, make sure scipy and numpy are compatible as given in `requirements.txt`

## Training and Testing 
### Data Processing
Preprocessing of Cora and Citeseer datasets are as given in the original paper, with feature normalized. Amazon Photo and Computer datasets were divided as mentioned in the paper. \\
Bitcoin datasets are weigheted signed graphs that do not have node attributes. Eigen method (https://dl.acm.org/doi/abs/10.1145/3533271.3561793) was used to create node features from weighted signed edges and then the weight and sign of edges are discarded, but the edges are still directed. In this way it is converted to what our method could handle, just be careful the graphs are signed unlike the above four datasets.

### Current Setting
The current version: run `adv_train_pgd_cost_constraint.py` for cost learning and evasion testing, `poisoning_pgd_bitcoin.py` for poisoning testing (using saved learned costs)\\
see args defined in FLAGS, `perturb_ratio` and `cost_constraint` essentially works the same way, see calculation of variable `eps_cost`. The reported setting use `cost_constraint` $0.8$ and `perturb_ratio` $0.05, 0.10, 0.15$. The `hyper_c_ratio` is for changing the hyper parameter $\lambda$ to $\text{hyper_c_ratio} * 253.45/\text{eps} * \text{FLAGS.hyper_c_ratio}$, when `hyper_c_ratio` set to $1.0$, it is the tuned setting for good results.

Part of the Cost-Aware PGD attack is in `PGD_attack_cost_constraint.py`, with the bisection for finding Lagrange multiplier in `utils.py`, the backbone is in `models.py`.\\
`models.py` contains three parts in `class GCN`: the training of surrogate GCN model, training of cost allocation model (also a GCN), and gradient descent for attacking. Update of surrogate and cost allocation models are via ADAM, attack update is via GD with fixed learning rate.

Now the cost allocation model uses a sigmoid output activation and a min-max scaling to make sure cost of each node is between 0 and 1, which in testing showed good effects. There were attempts of changing it into the following new setting where the output of cost allocation model is normalized and the sum of node costs equals a fixed $B$, however we yet to see good results due to limited time. The following new setting was once considered more reasonable as it gives same attacking cost constraint `eps_cost` for different dataset as $B * r$, which no longer concern with number of edges `total_edges` in the graph as it does now.

### Convert to New Settings
To conver to the new setting, modify `models.py` according to `models_copy.py` line 248, where a normalization of learned node costs are performed. This will ensure sum of node costs equals $B=100$ and distribute the cost more evenly (larger std). We already played a bit with the output design of cost allocation model by removing min-max scale or using relu activation instead of sigmoid, but yet to see significant improvements in performance.\\
New version (under development): `adv_train_pgd_cost_constraint_new.py` and `poisoning_pgd_bitcoin_new.py` for node cost learning + evasion testing and poisoning testing respectively, here the original args `perturb_ratio` and `cost_constraint` are combined to be `constraint_percentage` which is $r$ in attacking cost constraint $B*r$, `hyper_c` was set to 1. These hyper params can be tuned.

### Model File Saves
By the end of the adversarial training, `now_s` and `now_node_costs` are final attack perturbation vector and final learned costs respectively. `atk_accs` and `atk_aucs` records surrogate performance on clean graph first, and then defense performance in each epochs. The saved node costs are used to perform case study of cost distributions and baseline evaluations in `../BitcoinBaselines/`, where sum of node costs are needed to ensure fairness of comparisons.

## Case Study
The case study utilizes `orca_py` (https://github.com/qema/orca-py) in `../BitcoinGraphlets/` for Graphlet counting and extraction. The original implementations on the four datasets were lost but one can refer to implementations on bitcoin datasets. Manual convertions of graph data into required `.in` files are needed. One can use pandas on generated `.out` files from orca alg for statistics of costs on graphlets.

## Baselines & Meta Attack
The baseline models `GCN, GCNJaccard, GCNSVD, MedianGCN` and the experiment of effect of learned cost on Meta Attack are implemented with pytorch under `../BitcoinBaselines/` utilizing `DeepRobust` repo, see `README.md` under subdir for details. Notice that the source codes are highly modified on top of the original repo.\\