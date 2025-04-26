#!/bin/bash
 
#SBATCH --job-name=ESRGAN
#SBATCH --output=bash_logs/%j_%x.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

################## Train RRDB - i.e., Pretraining for ESRGAN ##################
singularity exec $SINGULARITY_CONTAINER bash -c "\
source .venv/bin/activate && \
python additional_baselines/train_rrdb.py"


#@@@@@@@@@@@@@@@ Test RRDB - make sure to set the correct (pre-determined, if doing train-test at one go) 
# path to the test data
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# python additional_baselines/test_esrgan.py"


################## Train ESRGAN ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# python additional_baselines/train_esrgan.py"
