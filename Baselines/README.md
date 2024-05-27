# Implementation of Baseline Models in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware Attacks

## About
This is the pytorch implementation of baselines and also Meta Attack in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware. This repo is developed on top of [DeepRobust](https://github.com/DSE-MSU/DeepRobust) repo with source code modified.

## Environment
To build env, first install PyTorch
```
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install setuptools==68.0.0
$ pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```
Then run:
```
$ cd Baselines/
$ pip install -r requirements.txt
```
Then, make sure you install deeprobust from the local `DeepRobust/` subdirectory instead of pip, for the cost-aware implementations
```
$ cd DeepRobust/
$ python setup.py install
```

## Training and Testing
### Data
The preprocessed data is available through [this link](https://drive.google.com/file/d/1lQtfUuvtO3zglQtwlL_gWcMtqNO5cUVp/view?usp=sharing), on a linux machine, put it in `Baselines/` directory, and unzip it with:
```
$ tar -xzvf riskeeper_data.tar.gz
```

### PGD Attack Baselines
See `GCN_baseline_pgd_cost.py` for currently implemented baseline models including GCN, GCNJaccard, GCNSVD, and MedianGCN, and also different heuristic cost allocation schemes  including avg, random, deg_original, clust_coef_original. Specify args `attack` to be `PGDCost` for cost-aware PGD attack. When testing attacks, combinations of models and cost_schemes can be used. Defaultly different cost schemes are tested with GCN, and different models are tested with avg cost scheme. Before running baseline with certain `perturb_ratio` and `cost_constraint`, make sure to run RisKeeper under same setting first to obtain learned cost allocations, as it is needed for the sum of node costs to be equal.

For running PGD baseline model experiments with specific args, use `GCN_baseline_pgd_cost.py` and set `cost_scheme` to `avg`, available `model` include `'GCN', 'GCNSVD', 'GCNJaccard', 'MedianGCN'`. For example for `GCNJaccard`, run:
```
$ cd Baselines/
$ python GCN_baseline_pgd_cost.py --dataset cora --attack PGDCost --perturb_ratio 0.05 --cost_constraint 0.8 --hyper_c_ratio 1.0 --cost_scheme avg --model GCNJaccard --binary_feature True --device 0
```

For running PGD baseline cost allocation schemes experiments with specific args, use `GCN_baseline_pgd_cost.py` with `model` set to `GCN`, available `cost_schemes` include `'raw', 'avg', 'random', 'deg_original', 'clust_coef_original'`. For example for `random` scheme, run:
```
$ cd Baselines/
$ python GCN_baseline_pgd_cost.py --dataset cora --attack PGDCost --perturb_ratio 0.05 --cost_constraint 0.8 --hyper_c_ratio 1.0 --cost_scheme random --model GCN --binary_feature True --device 0
```


### Transfering to Meta Attack
See `GCN_baseline_meta_cost.py` for transfering to cost-aware Meta Attack.

For running current Meta experiments, first go to [`../GCN_ADV_Train/`](../GCN_ADV_Train/README.md) and use `../GCN_ADV_Train/adv_train_pgd_cost_constraint.py` (see [README.md](../GCN_ADV_Train/README.md) for usage) to obtain the costs, then use `GCN_baseline_meta_cost.py` to test the performance of RisKeeper and baselines under cost-aware Meta attack. For testing RisKeeper's performance, set `model` to `GCN` and `cost_scheme` to `ours`. For example, run:
```
$ cd Baselines/
# To test performance of RisKeeper:
$ python GCN_baseline_meta_cost.py --model GCN --attack MetaCost --perturb_ratio 0.05 --dataset cora --binary_feature True --cost_scheme ours --hyper_c_ratio 0.4 --cost_constraint 0.2 --device 0
# To test performance of baselines:
$ python GCN_baseline_meta_cost.py --model GCNJaccard --attack MetaCost --perturb_ratio 0.05 --dataset cora --binary_feature True --cost_scheme avg --hyper_c_ratio 0.4 --cost_constraint 0.2 --device 0
```

The usage is same as the above cost-aware PGD except specifying argument `attack` to be `MetaCost`. 