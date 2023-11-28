#!/bin/bash
 
#SBATCH --job-name=Bao22
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --account=OD-224713

# Application specific commands: 
source /scratch2/has118/.env/pinn/bin/activate
python /scratch2/has118/physics-nn-fluid-dynamics/src/Bao22-PGSR.py