#!/bin/bash
 
#SBATCH --job-name=physics-SRGAN-EE
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=3GB
#SBATCH --gres=gpu:1
#SBATCH --account=OD-224713

source /scratch2/has118/.env/pinn/bin/activate

# Physics: 
# bash /scratch2/has118/physics-nn-fluid-dynamics/train_physics.sh >> /scratch2/has118/physics-nn-fluid-dynamics/job_more-train_physics_BDF_6Feb.log

# No-Physics:
bash /scratch2/has118/physics-nn-fluid-dynamics/train_no-physics.sh >> /scratch2/has118/physics-nn-fluid-dynamics/job_more-train_no-physics_6Feb.log