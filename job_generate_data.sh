#!/bin/bash --login
 
#SBATCH --job-name=PC-SRGAN
#SBATCH --output=bash_logs/%j_%x.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --partition=work
#SBATCH --account=pawsey1001

eval "$(conda shell.bash hook)"
conda activate fenics
python src/FEM_Allen_Cahn_Periodic_x4.py
