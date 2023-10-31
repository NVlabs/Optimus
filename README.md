# OPTIMUS: Imitating Task and Motion Planning with Visuomotor Transformers
<div style="text-align: center;">

This repository is the official implementation of Imitating Task and Motion Planning with Visuomotor Transformers.

[Murtaza Dalal](https://mihdalal.github.io/)$^1$, [Ajay Mandlekar](https://ai.stanford.edu/~amandlek/)$^2$,  [Caelan Garrett](http://web.mit.edu/caelan/www/)$^2$, [Ankur Handa](https://ankurhanda.github.io/)$^2$, [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)$^1$, [Dieter Fox](https://homes.cs.washington.edu/~fox/)$^2$

$^1$ CMU, $^2$ NVIDIA

[Project Page](https://mihdalal.github.io/optimus/) | [Arxiv](https://arxiv.org/abs/2305.16309) | [Video](https://www.youtube.com/watch?v=2ItlsuNWi6Y)


<img src="assets/optimus_gallery_gif.gif" width="100%" title="main gif">
<div style="margin:10px; text-align: justify;">
Optimus is a framework for training large scale imitation policies for robotic manipulation by distilling Task and Motion Planning into visuomotor Transformers. In this release we include datasets for replicating our results on Robosuite as well as code for performing TAMP data filtration and training/evaluating visuomotor Transformers on TAMP data.

If you find this codebase useful in your research, please cite:
```bibtex
@inproceedings{dalal2023optimus,
    title={Imitating Task and Motion Planning with Visuomotor Transformers},
    author={Dalal, Murtaza and Mandlekar, Ajay and Garrett, Caelan and Handa, Ankur and Salakhutdinov, Ruslan and Fox, Dieter},
    journal={Conference on Robot Learning},
    year={2023}
}
```

# Table of Contents

- [Installation](#installation)
- [Dataset Download](#dataset-download)
- [TAMP Data Cleaning](#tamp-data-cleaning)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Task Visualizations](#task-visualizations)
- [Troubleshooting and Known Issues](#troubleshooting-and-known-issues)
- [Citation](#citation)

# Installation
To install dependencies, please run the following commands:
```
sudo apt-get update
sudo apt-get install -y \
    htop screen tmux \
    sshfs libosmesa6-dev wget curl git \
    libeigen3-dev \
    liborocos-kdl-dev \
    libkdl-parser-dev \
    liburdfdom-dev \
    libnlopt-dev \
    libnlopt-cxx-dev \
    swig \
    python3 \
    python3-pip \
    python3-dev \
    vim \
    git-lfs \
    cmake \
    software-properties-common \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    g++ \
    python2.7 \
    wget \
    vulkan-utils \
    mesa-vulkan-drivers \
    apt nano rsync \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
    software-properties-common  net-tools  unzip  virtualenv \
    xpra xserver-xorg-dev libglfw3-dev patchelf python3-pip -y \
    && add-apt-repository -y ppa:openscad/releases && apt-get update && apt-get install -y openscad
```

Please add the following to your bashrc/zshrc:
```
export MUJOCO_GL='egl'
WANDB_API_KEY=...
```

To install python requirements:

```
conda create -n optimus python=3.8
conda activate optimus
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

# Dataset Download

#### Method 1: Using `download_datasets.py` (Recommended)

`download_datasets.py` (located at `optimus/scripts`) is a python script that provides a programmatic way of downloading the datasets. This is the preferred method, because this script also sets up a directory structure for the datasets that works out of the box with the code for reproducing policy learning results.

A few examples of using this script are provided below:

```
# default behavior - just download Stack dataset
python download_datasets.py

# download datasets for Stack and Stack Three
python download_datasets.py --tasks Stack StackThree

# download all datasets, but do a dry run first to see what will be downloaded and where
python download_datasets.py --tasks all --dry_run

# download all datasets for all tasks 
python download_datasets.py --tasks all # this downloads Stack, StackThree, StackFour and StackFive 
```

#### Method 2: Using Direct Download Links

You can download the datasets manually through Google Drive.

**Google Drive folder with all datasets:** [link](https://drive.google.com/drive/folders/1Dfi313igOuvc5JUCMTzQrSixUKndhW__?usp=drive_link)

# TAMP Data Cleaning
As described in Section 3.2 of the Optimus paper, we develop two TAMP demonstration filtering strategies to curb variance in the expert supervision: 1) Prune out demonstrations that have out of distribution trajectory length 2) Remove demonstrations that exit the visible workspace. In practice, we remove trajectories that have length greater than 2 standard deviations than the mean and exit a pre-defined workspace which includes all visible regions from the fixed camera viewpoint. 

We include the data filtration code in `filter_trajectories.py` and give example usage below.
```
# usage:
python optimus/scripts/filter_trajectories.py --hdf5_paths datasets/<>.hdf5 --x_bounds <> <> --y_bounds <> <> --z_bounds <> <> --val_ratio <> --filter_key_prefix <> --outlier_traj_length_sd <>

# example
python optimus/scripts/filter_trajectories.py --hdf5_paths datasets/robosuite_stack.hdf5 --outlier_traj_length_sd 2 --x_bounds -0.2 0.2 --y_bounds -0.2 0.2 --z_bounds 0 1.1 --val_ratio 0.1
```

For the datasets that we have released, we have already performed these filtration operations, so you do not need to do so. Please do not run the below 

# Model Training
After downloading the appropriate datasets you’re interested in using by running the `download_datasets.py` script, you can train policies using the `pl_train.py` script in `optimus/scripts`. Our training code wraps around [robomimic](https://robomimic.github.io/) (the key difference is we use PyTorch Lightning), please see the robomimic docs for a detailed overview of the imitation learning code. Following the robomimic format, our training configs are located in optimus/exps/local/robosuite/, with a different folder for each environment (stack, stackthree, stackfour, stackfive). We demonstrate example usage below: 
```
# usage:
python optimus/scripts/pl_train.py --config optimus/exps/local/robosuite/<env>/bc_transformer.json

# example:
python optimus/scripts/pl_train.py --config optimus/exps/local/robosuite/stack/bc_transformer.json
```

# Model Inference 
Given a checkpoint (from a training run), if you want to perform inference and evaluate the model, you can use `run_trained_agent_pl.py`. This script is based on `run_trained_agent.py` in [robomimic](https://robomimic.github.io/) but adds support for our PyTorch Lightning checkpoints. Concretely you need to specify the path to a specific ckpt file (for `--agent`) and the directory which contains `config.json` (for `--resume_dir`). We demonstrate example usage below: 
```
# usage:
python optimus/scripts/run_trained_agent_pl.py --agent /path/to/trained_model.ckpt --resume_dir /path/to/training_dir --n <> --video_path <>.mp4

# example:
python optimus/scripts/run_trained_agent_pl.py --agent optimus/trained_models/robosuite_trained_models/bc_transformer_stack_image/20231026101652/models/model_epoch_50_Stack_success_1.0.ckpt --resume_dir optimus/trained_models/robosuite_trained_models/bc_transformer_stack_image/20231026101652/ --n 10 --video_path stack.mp4
```

# Troubleshooting and Known Issues

- If your training seems to be proceeding slowly (especially for image-based agents), it might be a problem with robomimic and more modern versions of PyTorch. We recommend PyTorch 1.12.1 (on Ubuntu, we used `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`). It is also a good idea to verify that the GPU is being utilized during training.
- If you run into trouble with installing [egl_probe](https://github.com/StanfordVL/egl_probe) during robomimic installation (e.g. `ERROR: Failed building wheel for egl_probe`) you may need to make sure `cmake` is installed. A simple `pip install cmake` should work.
- If you run into other strange installation issues, one potential fix is to launch a new terminal, activate your conda environment, and try the install commands that are failing once again. One clue that the current terminal state is corrupt and this fix will help is if you see installations going into a different conda environment than the one you have active.

If you run into an error not documented above, please search through the [GitHub issues](https://github.com/NVlabs/optimus/issues), and create a new one if you cannot find a fix.

## Citation

Please cite [the Optimus paper](https://arxiv.org/abs/2305.16309) if you use this code in your work:

```bibtex
@inproceedings{dalal2023optimus,
    title={Imitating Task and Motion Planning with Visuomotor Transformers},
    author={Dalal, Murtaza and Mandlekar, Ajay and Garrett, Caelan and Handa, Ankur and Salakhutdinov, Ruslan and Fox, Dieter},
    journal={Conference on Robot Learning},
    year={2023}
}
```
