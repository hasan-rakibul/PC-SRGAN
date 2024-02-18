#!/bin/bash
 
#SBATCH --job-name=TR-PHY
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --gres=gpu:1
#SBATCH --account=OD-224713

source /scratch2/has118/.env/pinn/bin/activate

bash /scratch2/has118/physics-nn-fluid-dynamics/train_physics.sh Allen-Cahn_Periodic
# bash /scratch2/has118/physics-nn-fluid-dynamics/train_physics.sh Allen-Cahn_Neumann
# bash /scratch2/has118/physics-nn-fluid-dynamics/train_physics.sh Erikson-Johnson
