# Value at Adversarial Risk: A Graph Defense Strategy Against Cost-Aware Attacks

## About
This repo is the official code for AAAI24:["Value at Adversarial Risk: A Graph Defense Strategy against Cost-Aware Attacks"](https://ojs.aaai.org/index.php/AAAI/article/view/29282)

## Dependencies
The Riskeeper Model and other baselines are implemented with Tensorflow and PyTorch respectively. They would require seperate environments. For dependencies of each environment, see README in subdirectories [GCN_ADV_Train/README.md] and [Baselines/README.md]. 

For building environment of RisKeeper's adversarial training, run:
```
cd GCN_ADV_Train/
pip install -r requirements.txt
```

For building environment of baselines models, run:
```
cd Baselines/
pip install -r requirements.txt
```

In addition, cuda 12.0 was used in our projecct. 

## File Folders
`GCN_ADV_Train\`: contains training, testing and model file for the adversarial training of RisKeeper and the surrogate model.
`Baselines`: contains training and testing code for baseline models as well as baseline cost allocation schemes.

## Usage: How to run the code

### 1. Download the data
The preprocessed data is available through [this link](https://drive.google.com/file/d/1lQtfUuvtO3zglQtwlL_gWcMtqNO5cUVp/view?usp=sharing), on a linux machine, copy it into `GCN_ADV_Train/` and `Baselines/` respectively, and unzip it with:
```
$ tar -xzvf riskeeper_data.tar.gz
```

### 2. Run Adversarial Training for RisKeeper
To run the adversarial training of RisKeeper, go to the `GCN_ADV_Train` subdirectory and read the `README.md` file for further instructions.

for running all current experiments for RisKeeper's training, setup the TensorFlow environment as instructed in the readme file and run
```
$ cd GCN_ADV_Train/
$ bash run_datasets.sh
```

### 3. Run Baseline Models
Before running baseline models, make sure to run the adversarial training for RisKeeper first, as the sum of costs would be required to ensure fair comparison

for running all baseline experiments for Cost-Aware PGD attack, setup the PyTorch environment as instructed in the readme file and run
```
$ cd Baselines/
$ bash run_baselines_pgd_all.sh
```

### 4. Transferring to Meta Attack
For running transferring to Cost-Aware Meta Attack experiments, first run the adversarial training with corresponding args in `GCN_ADV_Train/`
```
$ cd GCN_ADV_Train/
$ bash run_meta_pre.sh
```
Then in `Baselines/`, run
```
$ cd Baselines/
$ bash run_baselines_meta.sh
```

## Acknowledgements
Part of this code is built on [DeepRobust](https://github.com/DSE-MSU/DeepRobust) and [GCN_ADV_Train] https://github.com/KaidiXu/GCN_ADV_Train

## Cite 
If you find this work helpful, please cite
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