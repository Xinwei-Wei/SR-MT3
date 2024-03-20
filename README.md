## Introduction

This repository contains the code for the paper "Transformer based Online Continuous Multi-Target Tracking with State Regression" (<https://ieeexplore.ieee.org/document/10382279>).

The work for SR-MT3 was developed based on the [MT3](https://github.com/JulianoLagana/MT3) architecture proposed by J. Pinto *et al.* in "Next Generation Multitarget Trackers: Random Finite Set Methods vs Transformer-based Deep Learning" (<https://arxiv.org/abs/2104.00734>).

## Conda environment configuration

In order to set up a conda environment with all the necessary dependencies, run the command:

``` bash
conda env create -f conda-env/environment-<gpu/cpu>.yml
```

## Model training

Run an training process using the `training.py` script. Example usage:

``` bash
python src/training.py -tp configs/tasks/scen1.1.yaml -mp configs/models/mt3.pro.yaml
```

Task and scenario parameters such as FOV, target number, noise covariance, etc, are found in the file `configs/tasks/scen1.1.yaml`.
Training hyperparameters such as batch size, learning rate, checkpoint interval, etc, are found in the file `configs/models/mt3.pro.yaml`. 

## Model prediction

After a training has generated checkpoints, you can predict target states using the `PredictDemo.py` script. Example usage: 

``` bash
python src/PredictDemo.py -rp src/results/experiment_name
```

If you would like to evaluate the performance by other algorithms, you could specify the `matFile` parameter in `PredictDemo.py` script. This parameter is a file in where the measurements, model predictions and ground-truths of the simulation runs will be stored.
