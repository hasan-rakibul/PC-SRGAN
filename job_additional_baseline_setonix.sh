#!/bin/bash
 
#SBATCH --job-name=RRDB-ESRGAN
#SBATCH --output=bash_logs/%j_%x.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

################## Pretraining RRDB (or using it as SR No Physics) for ESRGAN ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# python additional_baselines/train_rrdb.py \
# --config_path=additional_baselines/configs/train_rrdb_no-physics.yaml"

################## Train RRDB (Physics) as SR Model ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# python additional_baselines/train_rrdb.py \
# --config_path=additional_baselines/configs/train_rrdb_physics.yaml"

# ################## Train ESRGAN (No Physics) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# python additional_baselines/train_esrgan.py \
# --config_path=additional_baselines/configs/train_esrgan_no-physics.yaml"

# ################## Train ESRGAN (Physics) ##################
singularity exec $SINGULARITY_CONTAINER bash -c "\
source .venv/bin/activate && \
python additional_baselines/train_esrgan.py \
--config_path=additional_baselines/configs/train_esrgan_physics.yaml"

################## Test - make sure to set the correct (pre-determined, if doing train-test at one go) 
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# python additional_baselines/test_esrgan.py"
