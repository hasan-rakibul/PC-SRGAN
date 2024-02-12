#!/bin/bash
 
#SBATCH --job-name=SRGAN_Erikson_Johnson_Train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1
#SBATCH --account=OD-224713

source /scratch2/has118/.env/pinn/bin/activate

# No-Physics:
bash /scratch2/has118/physics-nn-fluid-dynamics/train_no-physics.sh
#  >> /scratch2/has118/physics-nn-fluid-dynamics/configs/train/Erikson-Johnson_Subset70_No-Physics_9Feb.log