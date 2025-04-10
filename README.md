# EScAIP: Efficiently Scaled Attention Interatomic Potential

This repository contains the official implementation of the [Efficiently Scaled Attention Interatomic Potential (NeurIPS 2024)](https://openreview.net/forum?id=Y4mBaZu4vy).

> Scaling has been a critical factor in improving model performance and generalization across various fields of machine learning.
It involves how a model’s performance changes with increases in model size or input data, as well as how efficiently computational resources are utilized to support this growth.
Despite successes in scaling other types of machine learning models, the study of scaling in Neural Network Interatomic Potentials (NNIPs) remains limited. NNIPs act as surrogate models for ab initio quantum mechanical calculations, predicting the energy and forces between atoms in molecules and materials based on atomic configurations. The dominant paradigm in this field is to incorporate numerous physical domain constraints into the model, such as symmetry constraints like rotational equivariance. We contend that these increasingly complex domain constraints inhibit the scaling ability of NNIPs, and such strategies are likely to cause model performance to plateau in the long run. In this work, we take an alternative approach and start by systematically studying NNIP scaling properties and strategies. Our findings indicate that scaling the model through attention mechanisms is both efficient and improves model expressivity. These insights motivate us to develop an NNIP architecture designed for scalability: the Efficiently Scaled Attention Interatomic Potential (EScAIP).
EScAIP leverages a novel multi-head self-attention formulation within graph neural networks, applying attention at the neighbor-level representations.
Implemented with highly-optimized attention GPU kernels, EScAIP achieves substantial gains in efficiency---at least 10x speed up in inference time, 5x less in memory usage---compared to existing NNIP models. EScAIP also achieves state-of-the-art performance on a wide range of datasets including catalysts (OC20 and OC22), molecules (SPICE), and materials (MPTrj).
We emphasize that our approach should be thought of as a philosophy rather than a specific model, representing a proof-of-concept towards developing general-purpose NNIPs that achieve better expressivity through scaling, and continue to scale efficiently with increased computational resources and training data.

**Current notes, Nov 2024**: the model is being transitioned to the "hydra" configuration (from FairChem) and so is still under active development. All pre-trained models and checkpoints from the paper will be made public soon once this is done.

## Install

Step 1: Install mamba solver for conda (optional)

```bash
conda install mamba -n base -c conda-forge
```

Step 2: Install the dependencies

```bash
mamba env create -f env.yml
conda activate escaip
```

Step 3: Install FairChem core package

```bash
git submodule update --init --recursive
pip install -e fairchem/packages/fairchem-core
```

Step 4: Install the pre-commit (if you want to contribute)

```bash
pre-commit install
```

## Train

Use `main.py` to train the models. Th usage is similar to FairChem (refer to the [FairChem documentation](https://fair-chem.github.io/core/model_training.html) for more details):

Single GPU training:

```bash
python main.py --mode train --config-yml {CONFIG} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

Use `start_exp.py` to start a run in the background:

```bash
python start_exp.py --config-yml {CONFIG} --cvd {GPU NUM} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

Multi-GPU training (same device):

```bash
torchrun --standalone --nproc_per_node={N} main.py --distributed --num-gpus {N} {...}
```

For distributed training on NERSC, refer to the [NERSC distributed training with submitit](NERSC_dist_train.md).

## Simulation

Use `simulate.py` to simulate the models. It requires to install the MDSim package:

```bash
pip install -e MDsim
```

A separate config file is needed for the simulation. Some example configs can be found in `configs/s2ef/MD22/datasets/{DATASET}/simulation.yml`.

```bash
python simulate.py --simulation_config_yml {SIMULATION CONFIG} --model_dir {CHECKPOINT DIR} --model_config_yml {MODEL CONFIG}--identifier {IDENTIFIER}
```

### Example Usage

For example, to simulate on MD22 DHA split:

```bash
python simulate.py --simulation_config_yml configs/s2ef/MD22/datasets/DHA/simulation.yml --model_dir checkpoints/MD22_DHA/ --model_config_yml configs/s2ef/MD22/EScAIP/DHA.yml --identifier test_simulation
```

After the simulation, the results can be found in the checkpoint directory under the `identifier` folder. Use `scripts/analyze_rollouts_md17_22.py` to analyze the results:

```bash
PYTHONPATH=./ python scripts/analyze_rollouts_md17_22.py --md_dir checkpoints/MD22_DHA/md_sim_test_simulation --gt_traj /data/md22/md22_AT-AT.npz --xlim 25
```

Refer to the [MDSim repo](https://github.com/kyonofx/MDsim) for more details.


## Model Architecture

Refer to the [model architecture](model_architecture.md) for more details. A detailed description of model configurations can be found [here](configs/example_config_EScAIP.yml).

Some notes on the configs:
- The model uses `torch.compile` to compile the model for better performance. [But it's not supporting second order gradient.](https://github.com/pytorch/pytorch/issues/91469) So to use gradient energy, it has to be disabled.
    - The implicit batch needs to be padded to use `torch.compile`. The size is controlled by `max_num_nodes_per_batch`.
- The attention kernels are selected by `atten_name`, which has 4 options:
    - `math`: the default attention by pytorch, supports all datatypes and second order gradient.
    - `memory_efficient`: the memory efficient kernel by pytorch, supports fp32 and fp16, but no second order gradient.
    - `flash`: the flash attention kernel by pytorch, supports only fp16, no second order gradient.
- The `use_fp16_backbone` will turn the graph attention backbone to fp16, and the output head will still be fp32. For now, we're opting to use AutoMixedPrecision instead, as it's more stable.

## Pretrained Models

We will release the pretrained models in the paper soon (after the model transition is fanalized).

## Citation

If you find this work useful, please consider citing the following:

```
@inproceedings{
qu2024the,
title={The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains},
author={Eric Qu and Aditi S. Krishnapriyan},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=Y4mBaZu4vy}
}
```
