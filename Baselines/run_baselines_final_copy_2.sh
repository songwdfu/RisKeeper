#!/bin/bash
cd /data/wenda/BitcoinBaselines


# Define the models, perturb_ratio, and datasets as arrays
models=('GCN' 'GCNSVD' 'GCNJaccard' 'MedianGCN')
# models=('MedianGCN')
perturb_ratios=("0.05" "0.10" "0.15")
# perturb_ratios=("0.15")
datasets=('cora' 'citeseer' 'computers' 'photo')
# datasets=('computers' 'photo')
cost_schemes=('raw' 'avg' 'random' 'deg_original' 'clust_coef_original' 'ours')

for seed in "123" "124" "125" "126" "127"; do
# for seed in "124"; do
    # cost_scheme baselines
    for model in "GCN"; do
        for dataset in "${datasets[@]}"; do
            for cost_scheme in "${cost_schemes[@]}"; do
                echo "Running cost_scheme: $cost_scheme, dataset: $dataset, seed: $seed"
                python GCN_baseline_cost_schemes_copy.py --model "$model" --attack "PGDCost" --perturb_ratio "0.05" --cc "0.8" --dataset "$dataset" --seed "$seed" --binary_feature "True" --device "1" --cost_scheme "$cost_scheme"
            done
        done
    done
    # # models baselines
    # for perturb_ratio in "${perturb_ratios[@]}"; do
    #     for model in "${models[@]}"; do
    #         for dataset in "${datasets[@]}"; do
    #             echo "Running baseline: model: $model, ratio: $perturb_ratio dataset: $dataset, seed: $seed"
    #             python GCN_baseline_cost_schemes_copy.py --model "$model" --attack "PGDCost" --perturb_ratio "$perturb_ratio" --cc "0.8" --dataset "$dataset" --seed "$seed" --binary_feature "True" --device "1" --cost_scheme "avg"
    #         done
    #     done
    # done
done