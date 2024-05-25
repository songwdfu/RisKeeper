#!/bin/bash
# Define the models, perturb_ratio, and datasets as arrays
if [ "$#" -ne 1 ]; then
    echo "usage: bash run_baselines_pgd_schemes.sh <device>"
fi
models=('GCN' 'GCNSVD' 'GCNJaccard' 'MedianGCN')
perturb_ratios=("0.05" "0.10" "0.15")
datasets=('cora' 'citeseer' 'computers' 'photo')
cost_schemes=('raw' 'avg' 'random' 'deg_original' 'clust_coef_original' 'ours')

for seed in "123" "124" "125" "126" "127"; do
    for dataset in "${datasets[@]}"; do
        for cost_scheme in "${cost_schemes[@]}"; do
            echo "Running cost_scheme: $cost_scheme, dataset: $dataset, seed: $seed"
            python GCN_baseline_pgd_cost.py --model GCN --attack PGDCost --perturb_ratio 0.05 --cost_constraint 0.8 --dataset $dataset --seed $seed --binary_feature True --device $1 --cost_scheme $cost_scheme
        done
    done
done