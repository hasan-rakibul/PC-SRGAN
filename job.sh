#!/bin/bash
 
#SBATCH --job-name=Bao22-2
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --account=OD-224713

# Application specific commands: 
source /scratch2/has118/.env/Bao22/bin/activate
python /scratch2/has118/physics-nn-fluid-dynamics/src/Bao22-PGSR.py >> /scratch2/has118/physics-nn-fluid-dynamics/job.log