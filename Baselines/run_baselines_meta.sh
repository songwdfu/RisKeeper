#!/bin/bash
cd /data/wenda/BitcoinBaselines

# Define the models, perturb_ratio, and datasets as arrays
# models=('GCN' 'GCNSVD' 'GCNJaccard' 'MedianGCN')
models=('GCN')
# constraint_percentages=("0.05" "0.10" "0.15")
constraint_percentages=("0.05")
datasets=('cora' 'citeseer' 'photo' 'computers')
cost_schemes=('raw' 'avg' 'random' 'deg_original' 'clust_coef_original' 'ours')
# for model in "${models[@]}"; do
for cost_scheme in "${cost_schemes[@]}"; do
    for dataset in "${datasets[@]}"; do
        for constraint_percentage in "${constraint_percentages[@]}"; do
            echo "Running cost_scheme: $cost_scheme, constraint_percentage: $constraint_percentage, dataset: $dataset"
            python GCN_baseline_cost_surrogate_meta.py --model "GCN" --attack "MetaCost" --constraint_percentage "$constraint_percentage" --dataset "$dataset" --binary_feature "True" --cost_scheme "$cost_scheme"
        done
    done
done

# models=('GCN')
# # models=('GCN')
# constraint_percentages=("0.05" "0.10" "0.15")
# datasets=('cora' 'citeseer' 'photo' 'computers')
# # datasets=('computers')
# for model in "${models[@]}"; do
#     for constraint_percentage in "${constraint_percentages[@]}"; do
#         for dataset in "${datasets[@]}"; do
#             echo "Running ours: $model, constraint_percentage: $constraint_percentage, dataset: $dataset"
#             python GCN_baseline_cost_surrogate_meta.py --model "$model" --attack "MetaCost" --constraint_percentage "$constraint_percentage" --dataset "$dataset" --binary_feature "True" --cost_scheme "ours"
#         done
#     done
# done