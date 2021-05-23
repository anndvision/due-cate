# due-cate

Determninistic Uncertainty Estimation for Conditional Average Treatment Effects

## DUE Experiments

### IHDP DUE

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp \
        --root /users/andson/workbench/repos/quince/assets/ \
    due
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp/due/kernel-Matern12_ip-100-dh-200_nc-1_dp-3_ns--1.0_dr-0.1_sn-0.95_lr-0.001_bs-100_ep-1000/
```

### IHDP Covariate Shift DUE

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp-cov \
        --root /users/andson/workbench/repos/quince/assets/ \
    due
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp-cov/due/kernel-Matern12_ip-100-dh-200_nc-1_dp-3_ns--1.0_dr-0.1_sn-0.95_lr-0.001_bs-100_ep-1000/
```

## Ensemble Experiments

### IHDP Ensemble

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp \
        --root /users/andson/workbench/repos/quince/assets/ \
    ensemble
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp/ensemble/dh-200_nc-2_dp-3_ns--1.0_dr-0.2_sn-0.95_lr-0.001_bs-100_ep-500/
```

### IHDP Covariate Shift Ensemble

```.sh
due-cate \
    train \
        --job-dir experiments/ --num-trials 1000 --gpu-per-trial 0.2 \
    ihdp-cov \
        --root /users/andson/workbench/repos/quince/assets/ \
    ensemble
```

```.sh
due-cate evaluate ----experiment-dir experiments/ihdp-cov/ensemble/dh-200_nc-2_dp-3_ns--1.0_dr-0.2_sn-0.95_lr-0.001_bs-100_ep-500/
```
