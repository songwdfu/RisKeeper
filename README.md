# Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware Attacks

## About
This repo is the official code for AAAI24:["Value at Adversarial Risk: A Graph Defense Strategy against Cost-Aware Attacks"](https://ojs.aaai.org/index.php/AAAI/article/view/29282)

## Dependencies
The Riskeeper Model and other baselines are implemented with Tensorflow and PyTorch respectively. They would require seperate environments. For dependencies of each environment, see README in subdirectories [GCN_ADV_Train/](GCN_ADV_Train/README.md) and [Baselines/](Baselines/README.md). 

For building environment of RisKeeper's adversarial training, run:
```
$ cd GCN_ADV_Train/
$ pip install -r requirements.txt
```

For building environment of baselines models, follow the instructions in [`README.md`](Baselines/README.md) file in `Baselines/` subdirectory.
```
$ cd Baselines/
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install setuptools==68.0.0
$ pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
$ pip install -r requirements.txt
$ cd DeepRobust/
$ python setup.py install
```

In addition, `cuda 12.0` and `python 3.8` was used in our project. 

## File Folders
`GCN_ADV_Train/`: contains training, testing and model file for the adversarial training of RisKeeper and the surrogate model.
`Baselines/`: contains training and testing code for baseline models as well as baseline cost allocation schemes.

## Usage: How to run the code

### 1. Download the data
The preprocessed data is available through [this link](https://drive.google.com/file/d/1lQtfUuvtO3zglQtwlL_gWcMtqNO5cUVp/view?usp=sharing), on a linux machine, copy it into `GCN_ADV_Train/` and `Baselines/` respectively, and unzip it with:
```
$ tar -xzvf riskeeper_data.tar.gz
```

### 2. Run Adversarial Training for RisKeeper
To run the adversarial training of RisKeeper, go to the [`GCN_ADV_Train/`]((GCN_ADV_Train/)) subdirectory and read the [`README.md`](GCN_ADV_Train/README.md) file for further instructions.

For running current experiments for RisKeeper's training, setup the TensorFlow environment as instructed in the readme file and use `adv_train_pgd_cost_constraint.py`, for example:
```
$ cd GCN_ADV_Train/
$ python adv_train_pgd_cost_constraint.py --dataset cora --perturb_ratio 0.05 --cost_constraint 0.8 --discrete True
```
For testing RisKeeper's cost allocation under poisoning setting use `poisoning_pgd_cost_constraint.py`, for example, run:
```
$ cd GCN_ADV_Train/
$ python poisoning_pgd_cost_constraint.py --dataset cora --perturb_ratio 0.05 --cost_constraint 0.8 --discrete True
```

### 3. Run Baseline Models
Before running baseline models, make sure to run the adversarial training for RisKeeper first, as the sum of costs would be required to ensure fair comparison. 

For running baseline experiments for Cost-Aware PGD attack, setup the PyTorch environment as instructed in the [`README.md`](Baselines/README.md) file in [`Baselines/`](Baselines/) subdirectory and use `GCN_baseline_pgd_cost.py`. For example, run:
```
$ cd Baselines/
# Test GCNJaccard baseline
$ python GCN_baseline_pgd_cost.py --dataset cora --attack PGDCost --perturb_ratio 0.05 --cost_constraint 0.8 --hyper_c_ratio 1.0 --cost_scheme avg --model GCNJaccard --binary_feature True --device 0
# Test random cost_scheme baseline
$ python GCN_baseline_pgd_cost.py --dataset cora --attack PGDCost --perturb_ratio 0.05 --cost_constraint 0.8 --hyper_c_ratio 1.0 --cost_scheme random --model GCN --binary_feature True --device 0
```

### 4. Transferring to Meta Attack
For running transferring to Cost-Aware Meta Attack experiments, first run the adversarial training with corresponding args in [`GCN_ADV_Train/`](GCN_ADV_Train/) to obtain the cost allocation. For example, run:
```
$ cd GCN_ADV_Train/
$ python adv_train_pgd_cost_constraint.py --dataset cora --perturb_ratio 0.05 --hyper_c_ratio 0.4 --cost_constraint 0.2 --discrete True
```
Then in [`Baselines/`](Baselines/), use `GCN_baseline_meta_cost.py` with corresponding args to test performance of RisKeeper and baselines. For example, run:
```
$ cd Baselines/
$ # To test performance of RisKeeper:
$ python GCN_baseline_meta_cost.py --model GCN --attack MetaCost --perturb_ratio 0.05 --dataset cora --binary_feature True --cost_scheme ours --hyper_c_ratio 0.4 --cost_constraint 0.2 --device 0
# To test performance of baselines:
$ python GCN_baseline_meta_cost.py --model GCNJaccard --attack MetaCost --perturb_ratio 0.05 --dataset cora --binary_feature True --cost_scheme avg --hyper_c_ratio 0.4 --cost_constraint 0.2 --device 0
```

## Acknowledgements
Part of this code is built on "DeepRobust": https://github.com/DSE-MSU/DeepRobust, and "GCN_ADV_Train": https://github.com/KaidiXu/GCN_ADV_Train repositories.

## Cite 
If you find this work helpful, please cite:
```
@article{liao2024valueatrisk,
	title = {Value at Adversarial Risk: A Graph Defense Strategy against Cost-Aware Attacks},
	author = {Liao, Junlong and Fu, Wenda and Wang, Cong and Wei, Zhongyu and Xu, Jiarong},
      booktitle={AAAI},
	volume = {38},
	number = {12},
	pages = {13763-13771},
	year = {2024},
}
```