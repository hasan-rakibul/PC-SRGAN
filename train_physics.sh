#!/bin/bash

option=$1

export CUBLAS_WORKSPACE_CONFIG=:4096:8 # making it deterministic
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [[ $option = Allen-Cahn_Periodic ]]; then
    echo "Training Allen-Cahn Periodic"
    python src/SRGAN_train.py \
--config_path ./configs/train/Allen-Cahn_Periodic_Physics.yaml >&1 | tee ./bash_logs/Allen_Cahn_Periodic_Physics_$(date +%y%m%d-%H%M).txt
elif [[ $option = Allen-Cahn_Neumann ]]; then
    echo "Training Allen-Cahn Neumann"
    python src/SRGAN_train.py \
--config_path ./configs/train/Allen-Cahn_Neumann_Physics.yaml >&1 | tee ./bash_logs/Allen-Cahn_Neumann_Physics_$(date +%y%m%d-%H%M).txt
elif [[ $option = Erikson-Johnson ]]; then
    echo "Training Erikson-Johnson"
    python src/SRGAN_train.py \
--config_path ./configs/train/Erikson-Johnson_Physics.yaml >&1 | tee ./bash_logs/Erikson-Johnson_Physics_$(date +%y%m%d-%H%M).txt
else
    echo "No supported option $1. Check the argument."
fi