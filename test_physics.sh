#!/bin/bash

option=$1

if [[ $option = Allen-Cahn_Periodic ]]; then
    echo "Testing Allen-Cahn Periodic"
    python src/SRGAN_test.py \
--config_path ./configs/test/Allen-Cahn_Periodic_Physics.yaml >&1 | tee ./bash_logs/Allen_Cahn_Periodic_Physics_Test_$(date +%y%m%d-%H%M).txt
elif [[ $option = Allen-Cahn_Neumann ]]; then
    echo "Testing Allen-Cahn Neumann"
    python src/SRGAN_test.py \
--config_path ./configs/test/Allen-Cahn_Neumann_Physics.yaml >&1 | tee ./bash_logs/Allen-Cahn_Neumann_Physics_Test_$(date +%y%m%d-%H%M).txt
elif [[ $option = Erikson-Johnson ]]; then
    echo "Testing Erikson-Johnson"
    python src/SRGAN_test.py \
--config_path ./configs/test/Erikson-Johnson_Physics.yaml >&1 | tee ./bash_logs/Erikson-Johnson_Physics_Test_$(date +%y%m%d-%H%M).txt
else
    echo "No supported option $1. Check the argument."
fi