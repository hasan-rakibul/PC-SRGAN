#!/bin/bash
 
#SBATCH --job-name=Erikson_Johnson_SRGAN-Train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1
#SBATCH --account=OD-224713

source /scratch2/has118/.env/pinn/bin/activate

# Physics: 
# bash /scratch2/has118/physics-nn-fluid-dynamics/train_physics.sh >> /scratch2/has118/physics-nn-fluid-dynamics/Erikson-Johnson_Physics_8Feb.log
# bash /scratch2/has118/physics-nn-fluid-dynamics/test_physics.sh

# No-Physics:
bash /scratch2/has118/physics-nn-fluid-dynamics/train_no-physics.sh >> /scratch2/has118/physics-nn-fluid-dynamics/configs/train/Erikson-Johnson_No-Physics_8Feb.log