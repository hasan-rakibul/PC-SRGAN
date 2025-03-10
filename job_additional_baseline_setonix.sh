#!/bin/bash
 
#SBATCH --job-name=Baseline+
#SBATCH --output=bash_logs/%j_%x.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

singularity exec $SINGULARITY_CONTAINER bash -c "\
source .venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python additional_baselines/esrgan.py "
