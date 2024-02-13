#!/bin/bash

option=$1

if [[ $option = Allen-Cahn_Periodic ]]; then
    echo "Testing Allen-Cahn Periodic No Physics"
    python src/SRGAN_test.py \
--config_path ./configs/test/Allen-Cahn_Periodic_No-Physics.yaml >&1 | tee ./bash_logs/Allen_Cahn_Periodic_No-Physics_Test_$(date +%y%m%d-%H%M).txt
elif [[ $option = Allen-Cahn_Neumann ]]; then
    echo "Testing Allen-Cahn Neumann No Physics"
    python src/SRGAN_test.py \
--config_path ./configs/test/Allen-Cahn_Neumann_No-Physics.yaml >&1 | tee ./bash_logs/Allen-Cahn_Neumann_No-Physics_Test_$(date +%y%m%d-%H%M).txt
elif [[ $option = Erikson-Johnson ]]; then
    echo "Testing Erikson-Johnson No Physics"
    python src/SRGAN_test.py \
--config_path ./configs/test/Erikson-Johnson_No-Physics.yaml >&1 | tee ./bash_logs/Erikson-Johnson_No-Physics_Test_$(date +%y%m%d-%H%M).txt
else
    echo "No supported option $1. Check the argument."