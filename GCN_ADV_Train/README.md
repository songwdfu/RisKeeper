# Adversarial Training for RisKeeper in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware Attacks

## About
This is the TensorFlow implementation of the adversarial training of RisKeeper. Built on the [GCN_ADV_Train](https://github.com/KaidiXu/GCN_ADV_Train) repo.

## Environment 
To build env, run 
```
$ cd GCN_ADV_Train/
$ pip install -r requirements.txt
```
The original PGD was implemented on tf1, this is a modified version using tf 2.12.0, using `tensorflow.compat.v1`

If code fails to run, make sure scipy and numpy are compatible as given in `requirements.txt`

## Training and Testing 
### Data
The preprocessed data is available through [this link](https://drive.google.com/file/d/1lQtfUuvtO3zglQtwlL_gWcMtqNO5cUVp/view?usp=sharing), on a linux machine, put it in `GCN_ADV_Train/` directory, and unzip it with:
```
$ tar -xzvf riskeeper_data.tar.gz
```

### Current Experiments
The current version: run `adv_train_pgd_cost_constraint.py` for cost learning and evasion testing, `poisoning_pgd_cost_constraint.py` for poisoning testing (using saved learned costs)

see args defined in FLAGS, `perturb_ratio` and `cost_constraint` essentially works together, see calculation of variable `eps_cost`. The reported setting use `cost_constraint` $0.8$ and `perturb_ratio` $0.05, 0.10, 0.15$. The `hyper_c_ratio` is for changing the hyper parameter $\lambda$ to `hyper_c_ratio  * 253.45 / eps`. Set `discrete` to `True` to obtain discrete modified adj matrix after training finished.

To run all current experiment for adversarial training of RisKeeper, run:
```
$ cd GCN_ADV_Train/
$ bash run_datasets.sh
```

To run adversarial training with certain args under evasion setting, for example, run:
```
$ cd GCN_ADV_Train/
$ python adv_train_pgd_cost_constraint.py --dataset cora --perturb_ratio 0.05 --cost_constraint 0.8 --discrete True
```

This will also save the perturbations for use in poisoning attack and the allocated node costs for use in Baseline experiments.

Then, to test in poisoning setting with certain args, for example, run:
```
$ cd GCN_ADV_Train/
$ python poisoning_pgd_cost_constraint.py --dataset cora --perturb_ratio 0.05 --cost_constraint 0.8 --discrete True
```

The implementation of the Cost-Aware PGD attack is in `PGD_attack_cost_constraint.py`, with the bisection for finding Lagrange multiplier in `utils.py`(`bisection_cost` function), the backbone is in `models.py`.

`models.py` contains three parts in `class GCN`: the training of surrogate GCN model, training of cost allocation model (also a GCN), and gradient descent for attacking. Update of surrogate and cost allocation models are via ADAM, attack update is via GD with fixed learning rate. The cost allocation model uses a sigmoid output activation and a min-max scaling to make sure cost of each node is between 0 and 1, which in testing showed good effects. 

### Model File Saves
By the end of the adversarial training, `now_s` and `now_node_costs` are final attack perturbation vector and final learned costs respectively. `atk_accs` and `atk_aucs` records surrogate performance on clean graph first, and then defense performance in each epochs. The saved node costs are used to perform case study of cost distributions and baseline evaluations in `../Baselines/`, where sum of node costs are needed to ensure fairness of comparisons.

## Baselines & Transferring to Meta Attack
The baseline models `GCN, GCNJaccard, GCNSVD, MedianGCN` are implemented with pytorch under `../Baselines/` utilizing `DeepRobust` repo, see [`../Baselines/README.md`](../Baselines/README.md) for details. Notice that the source codes are highly modified on top of the original repo. 

For running experiments for transfering onto Meta attack, please first run:
```
$ cd GCN_ADV_Train/
$ bash run_meta_pre.sh
```

Then head to `../Baselines/` and read the [`README.md`](../Baselines/README.md) file for instructions.
