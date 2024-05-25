# Implementation of Baseline Models in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware 
This is the pytorch implementation of baselines and also Meta Attack in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware. This repo is developed on top of DeepRobust repo (https://github.com/DSE-MSU/DeepRobust) with source code modified.
## Environment
To build env, run:
```
$ cd Baselines/
$ pip install -r requirements.txt
```

## Training and Testing
### Data
The preprocessed data is available through [this link](https://drive.google.com/file/d/1lQtfUuvtO3zglQtwlL_gWcMtqNO5cUVp/view?usp=sharing), on a linux machine, put it in `GCN_ADV_Train/` directory, and unzip it with:
```
$ tar -xzvf riskeeper_data.tar.gz
```

### PGD Attack Baselines
See `GCN_baseline_pgd_cost.py` for currently implemented baseline models including GCN, GCNJaccard, GCNSVD, and MedianGCN, and also different heuristic cost allocation schemes  including avg, random, deg_original, clust_coef_original. Specify args `attack` to be `PGDCost` for cost-aware PGD attack. When testing attacks, combinations of models and cost_schemes can be used. Defaultly different cost schemes are tested with GCN, and different models are tested with avg cost scheme. Before running baseline with certain `perturb_ratio` and `cost_constraint`, make sure to run RisKeeper under same setting first to obtain learned cost allocations, as it is needed for the sum of node costs to be equal.

For all current PGD baseline experiments on different models and different cost_schemes respectively, run:
```
$ cd Baselines/
$ bash run_baselines_all.sh <device>
```

For running PGD baseline with specific args, for example, run:
```
$ cd Baselines/
$ python GCN_baseline_pgd_cost.py --dataset cora --attack PGDCost --perturb_ratio 0.05 --cost_constraint 0.8 --hyper_c_ratio 1.0 --cost_scheme avg --model GCNJaccard --binary_feature True --device 0
```

### Transfering to Meta Attack
See `GCN_baseline_meta_cost.py` for transfering to cost-aware Meta Attack.

For running current Meta experiments, first go to `../GCN_ADV_Train/` and run `bash run_meta_pre.sh` to obtain the costs, then run:
```
$ cd Baselines/
$ bash run_baselines_meta.sh <device>
```

For testing the effect of cost obtained from RisKeeper, simply specify `cost_schemes` to be `ours`

For baseline models, the usage is same as the above cost-aware PGD except specifying arg `attack` to be `MetaCost`.

For running Meta baseline with specific args, for example, run:
```
$ cd Baselines/
$ python GCN_baseline_meta_cost.py --dataset cora --attack MetaCost --perturb_ratio 0.05 --cost_constraint 0.4 --hyper_c_ratio 0.2 --cost_scheme avg --model GCNJaccard --binary_feature True --device 0
```