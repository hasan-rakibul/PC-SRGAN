#!/bin/bash
 
#SBATCH --job-name=PC-SRGAN
#SBATCH --output=bash_logs/%j_%x.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

################## Train SRGAN (Physics) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./train_physics.sh Allen-Cahn_Periodic"

##### PC-SRGAN_x4
singularity exec $SINGULARITY_CONTAINER bash -c "\
source .venv/bin/activate && \
./train_physics.sh Allen-Cahn_Periodic_x4"

##### SRGAN_x4
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./train_no-physics.sh Allen-Cahn_Periodic_x4"

# for testing, make sure to set the correct pre-determined path of the trained ckpt in the test config
########3E PC-SRGANs##########33
################## Test PC-SRGAN (Periodic) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_physics.sh Allen-Cahn_Periodic"

################## Test PC-SRGAN_x4 (Periodic) ##################
singularity exec $SINGULARITY_CONTAINER bash -c "\
source .venv/bin/activate && \
./test_physics.sh Allen-Cahn_Periodic_x4"

################## Test PC-SRGAN (Neumann) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_physics.sh Allen-Cahn_Neumann"

################## Test PC-SRGAN (Eriksson-Johnson) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_physics.sh Erikson-Johnson"

######### Just SRGANs ##########
################## Test SRGAN (Periodic) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_no-physics.sh Allen-Cahn_Periodic"

################## Test SRGAN_x4 (Periodic) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_no-physics.sh Allen-Cahn_Periodic_x4"

################## Test SRGAN (Neumann) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_no-physics.sh Allen-Cahn_Neumann"

# ################## Test SRGAN (Eriksson-Johnson) ##################
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source .venv/bin/activate && \
# ./test_no-physics.sh Erikson-Johnson"
