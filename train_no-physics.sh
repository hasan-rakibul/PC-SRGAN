#!/bin/bash

option=$1

export CUBLAS_WORKSPACE_CONFIG=:4096:8 # making it deterministic
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [[ $option = Allen-Cahn_Periodic ]]; then
    echo "Training Allen-Cahn Periodic No Physics"
    python src/SRGAN_train.py \
--config_path ./configs/train/Allen-Cahn_Periodic_No-Physics.yaml >&1 | tee ./bash_logs/Allen_Cahn_Periodic_No-Physics_$(date +%y%m%d-%H%M).txt
elif [[ $option = Allen-Cahn_Periodic_x4 ]]; then
    echo "Training Allen-Cahn Periodic_x4 No Physics"
    python src/SRGAN_train.py \
--config_path ./configs/train/Allen-Cahn_Periodic_No-Physics_x4.yaml >&1 | tee ./bash_logs/Allen_Cahn_Periodic_No-Physics_x4_$(date +%y%m%d-%H%M).txt
elif [[ $option = Allen-Cahn_Neumann ]]; then
    echo "Training Allen-Cahn Neumann No Physics"
    python src/SRGAN_train.py \
--config_path ./configs/train/Allen-Cahn_Neumann_No-Physics.yaml >&1 | tee ./bash_logs/Allen-Cahn_Neumann_No-Physics_$(date +%y%m%d-%H%M).txt
elif [[ $option = Erikson-Johnson ]]; then
    echo "Training Erikson-Johnson No Physics"
    python src/SRGAN_train.py \
--config_path ./configs/train/Erikson-Johnson_No-Physics.yaml >&1 | tee ./bash_logs/Erikson-Johnson_No-Physics_$(date +%y%m%d-%H%M).txt
else
    echo "Your entered argument: $option is not valid. Argument must be one of the following: Allen-Cahn_Periodic, Allen-Cahn_Neumann, Erikson-Johnson"
fi