#!/bin/bash
# meta baselines
if [ "$#" -ne 1 ]; then
    echo "usage: bash run_baselines_meta.sh <device>"
fi
# ours
models=('GCN')
perturb_ratios=("0.05" "0.10" "0.15")
datasets=('cora' 'citeseer' 'photo' 'computers')
for model in "${models[@]}"; do
    for perturb_ratio in "${perturb_ratios[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Running meta ours: $model, perturb_ratio: $perturb_ratio, dataset: $dataset"
            python GCN_baseline_meta_cost.py --model $model --attack MetaCost --perturb_ratio $perturb_ratio --dataset $dataset --binary_feature True --cost_scheme ours --hyper_c_ratio 0.4 --cost_constraint 0.2 --device $1
        done
    done
done
# baseline models
models=('GCN' 'GCNSVD' 'GCNJaccard' 'MedianGCN')
perturb_ratios=("0.05" "0.10" "0.15")
datasets=('cora' 'citeseer' 'photo' 'computers')
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for perturb_ratio in "${perturb_ratios[@]}"; do
            echo "Running meta baseline for model: $model, perturb_ratio: $perturb_ratio, dataset: $dataset"
            python GCN_baseline_meta_cost.py --model $model --attack MetaCost --perturb_ratio $perturb_ratio --dataset $dataset --binary_feature True --cost_scheme avg --hyper_c_ratio 0.4 --cost_constraint 0.2 --device $1
        done
    done
done