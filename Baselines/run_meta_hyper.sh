# for dataset in "cora" "citeseer" "computers" "photo"; do
# for hyper_c_ratio in "8" "11" "12" "13" "14" "15" "16" "20"; do
# cost_schemes=('raw' 'avg' 'random' 'deg_original' 'clust_coef_original' 'ours')
cost_schemes=('avg' 'random' 'deg_original' 'clust_coef_original' 'ours')
for seed in "123" "124" "125" "126" "127"; do
# for seed in "123"; do
      # for hyper_c_ratio in "0.4" "0.8" "1.0" "1.4" "1.8" "2.2" "2.6"; do
      for hyper_c_ratio in "0.4"; do
            for perturb_ratio in "0.05" "0.1" "0.15"; do
                  # for perturb_ratio in "0.05"; do
                  for cost_scheme in "${cost_schemes[@]}"; do
                        echo $seed $hyper_c_ratio $perturb_ratio
                        # python GCN_baseline_cost_surrogate_meta.py --cost_scheme "avg" --model "GCN" --dataset "cora" --device "2" --perturb_ratio "$perturb_ratio" --hyper_c_ratio "$hyper_c_ratio" --cc "0.2" --seed "$seed"
                        # python GCN_baseline_cost_surrogate_meta.py --cost_scheme "ours" --model "GCN" --dataset "cora" --device "2" --perturb_ratio "$perturb_ratio" --hyper_c_ratio "$hyper_c_ratio" --cc "0.2" --seed "$seed"
                        python GCN_baseline_cost_surrogate_meta.py --cost_scheme "$cost_scheme" --attack "MetaCost" --model "GCN" --dataset "cora" --device "2" --perturb_ratio "$perturb_ratio" --hyper_c_ratio "$hyper_c_ratio" --cc "0.2" --seed "$seed"
                  done
            done
      done
done