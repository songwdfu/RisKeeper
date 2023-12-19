# Implementation of Baseline Models in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware 
This is the pytorch implementation of baselines and also Meta Attack in Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware. This repo is developed on top of DeepRobust repo with source code modified.
## Environment
To build env, run:
```
pip install -r requirements.txt
```
## Training and Testing
### Data Processing
Is the same as for RisKeeper in `../GCN_ADV_Train/`

### Baseline Models
See `GCN_baseline_cost_schemes_copy.py` for currently implemented baseline models including GCN, GCNJaccard, GCNSVD, and MedianGCN, and also different heuristic cost allocation schemes  including avg, random, deg_original, clust_coef_original. Specify args `attack` to be `PGDCost` for cost-aware PGD attack. When testing attacks, combinations of models and cost_schemes can be used. Defaultly different cost schemes are tested with GCN, and different models are tested with avg cost scheme. Before running baseline with certain `perturb_ratio` and `cc` (`cost_constraint` in `../GCN_ADV_Train/`), make sure to run RisKeeper under same setting first to obtain learned cost allocations, as it is needed for the sum of node costs to be equal.

For all current baseline experiments on different models and different cost_schemes respectively, run:
```
bash run_baselines_final_copy_2.sh
```
and 
```
bash run_baselines_final.sh
```

For running baseline with specific args, for example, run:
```
python GCN_baseline_cost_schemes_copy.py --dataset cora --attack PGDCost --perturb_ratio 0.05 --cc 0.8 --cost_scheme avg --model GCNJaccard --binary_feature True
```

### Meta Attack
See `GCN_baseline_cost_surrogate_meta.py` for cost-aware Meta Attack, usage is same as the above except specifying arg `attack` to be `MetaCost`.