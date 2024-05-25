#!/bin/bash
for seed in "123" "124" "125" "126" "127";do
    # Evasion
    for cc in "0.8"; do
        for dataset in "cora" "citeseer" "computers" "photo"; do
        # for dataset in "cora" "citeseer"; do
            for perturb_ratio in "0.05" "0.10" "0.15"; do
                echo "Running evasion dataset: $dataset, $perturb_ratio, cc$cc, seed$seed"
                python adv_train_pgd_cost_constraint.py --dataset "$dataset" --perturb_ratio "$perturb_ratio" --cost_constraint "$cc" --discrete "True" --seed "$seed"
                echo "Done evasion dataset: $dataset, $perturb_ratio, cc$cc, seed$seed"
            done
        done
    done
    # Poisoning
    for cc in "0.8"; do
        for dataset in "cora" "citeseer" "computers" "photo"; do
            for perturb_ratio in "0.05" "0.10" "0.15"; do
                echo "Running poisoning dataset: $dataset, $perturb_ratio, cc$cc, seed$seed"
                python poisoning_pgd_bitcoin.py --dataset "$dataset" --perturb_ratio "$perturb_ratio" --cost_constraint "$cc"  --discrete "True" --seed "$seed"
            done
        done
    done

done

