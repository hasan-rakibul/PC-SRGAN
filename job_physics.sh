#!/bin/bash
 
#SBATCH --job-name=PI-SRGAN_Erikson_Johnson_Train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1
#SBATCH --account=OD-224713

source /scratch2/has118/.env/pinn/bin/activate

# Physics: 
bash /scratch2/has118/physics-nn-fluid-dynamics/train_physics.sh
# >> /scratch2/has118/physics-nn-fluid-dynamics/Erikson-Johnson_Subset70_Physics_9Feb.log

# bash /scratch2/has118/physics-nn-fluid-dynamics/test_physics.sh