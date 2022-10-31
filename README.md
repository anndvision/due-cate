# due-cate

Determninistic Uncertainty Estimation for Conditional Average Treatment Effects

## Citation

If you use this repository, please cite:
```
@article{van2021on,
  title={On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty},
  author={van Amersfoort, Joost and Smith, Lewis and Jesson, Andrew and Key, Oscar and Gal, Yarin},
  journal={arXiv preprint arXiv:2102.11409},
  year={2021}
}
```

## Installation
```
cd due-cate
conda env create -f environment.yml
pip install -e .
```

## Deep Kernel GP Experiments

### IHDP Deep Kernel GP

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp \
        --root assets/ \
    deep-kernel-gp
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp/deep_kernel_gp/kernel-Matern32_ip-100-dh-200_do-1_dp-3_ns--1.0_dr-0.1_sn-0.95_lr-0.001_bs-100_ep-1000/
```

### IHDP Covariate Shift Deep Kernel GP

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp-cov \
        --root assets/ \
    deep-kernel-gp
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp-cov/deep_kernel_gp/kernel-Matern32_ip-100-dh-200_do-1_dp-3_ns--1.0_dr-0.1_sn-0.95_lr-0.001_bs-100_ep-1000/
```

## Ensemble Experiments

### IHDP Ensemble

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp \
        --root assets/ \
    ensemble
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp/tarnet/dh-200_do-2_dp-3_ns--1.0_dr-0.2_sn-0.95_lr-0.001_bs-100_ep-500/
```

### IHDP Covariate Shift Ensemble

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp-cov \
        --root assets/ \
    ensemble
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp-cov/tarnet/dh-200_do-2_dp-3_ns--1.0_dr-0.2_sn-0.95_lr-0.001_bs-100_ep-500/
```
