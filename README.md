# 1st Place Solution

This repository contains code for the 1st place solution of the BYU - Locating Bacterial Flagellar Motors 2025
Competition [here](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025).

Solution Summary: [here](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583143)

My solution uses a 3D U-Net trained with heavy augmentations and auxiliary loss functions. During inference, I rank each tomogram based on the max predicted pixel value and use quantile thresholding to determine if a motor is present. Please read the solution summary for more details.

<h1 align="center">
<img src="./imgs/model.jpg" alt="Stage1" width="800">
</h1>

# Setup

This section covers how to reproduce model training. The hardware requirements below serve as a guideline and can be adjusted based on available resources. Some decent cloud options are [Lambda Labs](https://lambda.ai/service/gpu-cloud), [Runpod](https://www.runpod.io/pricing), and [Paperspace](https://www.paperspace.com/pricing).

### Hardware Requirements

| Component     | Recommended                     |
|---------------|---------------------------------|
| OS            | Ubuntu 22.04                    |
| RAM           | ≥ 32 GB                         |
| Disk Space    | ≥ 200 GB                        |
| CPU Cores     | ≥ 8                             |
| CUDA Version  | 12.4                            |
| GPU           | NVIDIA A100 (80GB)              |

### Miniconda Install

1. Get bash script for miniconda download from [here](https://docs.conda.io/en/main/miniconda.html#linux-installers)

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Exectute setup script

```
chmod u+x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

### Miniconda Environment Setup

1. Set channels

```
conda config --set channel_priority flexible
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --add channels nvidia
conda config --add channels pytorch
conda config --show channels
```

2. Create environment

```
conda create --prefix ../byu_env python=3.10
conda activate ../byu_env/
```

3. Install packages

```
# Conda packages
conda install pytorch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 pytorch-cuda==11.8 monai==1.4.0 wandb==0.19.6 tqdm==4.67.1

# Pip packages
pip install -r requirements.txt
```

### Directory Structure

Before training, you will need to manually create the data directory. 

1. The raw competition data should be placed under `./data/raw/`.

2. The external tomograms from [here](https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset?select=volumes_704) should be placed under `./data/processed/fold_-100/`.

3. `r3d18_KM_200ep.pt`, `r3d200_KM_200ep.pt` and `folds_all.csv` can be downloaded from [here](https://www.kaggle.com/datasets/brendanartley/solution-ds-byu-1st-place-metadata) and placed accordingly.

The directory structure should be as follows:

```
./data/
├── checkpoints/
├── model_zoo/
│   ├── r3d18_KM_200ep.pt
│   └── r3d200_KM_200ep.pt
├── processed/
│   ├── fold_-100/
│   │   ├── aba2013-04-06-7.npy
│   │   ├── aba2013-04-06-8.npy
│   │   └── ...
│   └── folds_all.csv
├── raw/
│   ├── test/
│   └── train/
└── sample_submission.csv
└── train_labels.csv
```

### Preprocessing

Once everything is setup correctly, you can process the competition data. This will populate the `./data/processed/` directory.

```
python -m src.pre.run
```

### Training

To train a model, run the following commands. Each model takes 35-40 hours to train. You can train for 250 epochs and get the same performance.

You can repeat this step to train multiple models for an ensemble.

```
# Make executable
chmod u+x run.sh

# Run in background
nohup ./run.sh > nohup.out &
```

