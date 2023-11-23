# physics-nn-fluid-dynamics

# Setting up environment
I first install python3.11 in Ubuntu 22.04.3 LTS  

```bash
sudo apt install python3.11 
sudo apt install python3.11-venv 
```

Create virtual environment 
```bash
python3.11 -m venv ~/.venv/pinn
```

Activate virtual environment. The virtual environment needs to be activated every time you open a new terminal. Alternatively keep the activation command in your `.bashrc` file.
```bash
source ~/.venv/pinn/bin/activate 
```

Install necessary files on the virtual environment 
```bash
python -m pip install -r requirements-SRGAN.txt
 # python -m pip install -r requirements.txt 
```

# Others' code/data
## Bao-UAI-PRU 
- Dataset is unavailable
- Code is in Tensorflow v1

### Setting environment of Bao-UAI-PRU
- The latest version of Tensorflow v1 is 1.15 that is supported by Python 3.7 as the latest. 

## Local Ubuntu machine
Install python3.7 first and then tensorflow 1.15:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7
sudo apt install python3.7-distutils
python3.7 -m pip install -r requirements-Bao22.txt
python3.7 -m pip install -U Pillow
```
## CSIRO Bracewell
```bash
module load python/3.7.11
python -m venv .env/Bao22/
source .env/Bao22/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-Bao22.txt
```

# Tutorial / Learning resources 
- [Darcy's Law - Flow in a Porous Medium](https://geo.libretexts.org/Courses/University_of_California_Davis/GEL_056%3A_Introduction_to_Geophysics/Geophysics_is_everywhere_in_geology.../02%3A_Diffusion_and_Darcy's_Law/2.05%3A_Darcy's_Law_-_Flow_in_a_Porous_Medium)

## CSIRO Cluster
- [https://confluence.csiro.au/display/SC/Useful+information+for+new+users](https://confluence.csiro.au/display/SC/Useful+information+for+new+users)
- [https://confluence.csiro.au/display/MLAIFSP/Remote+editing+with+VS+Code+on+bracewell](https://confluence.csiro.au/display/MLAIFSP/Remote+editing+with+VS+Code+on+bracewell)
- [https://confluence.csiro.au/display/~mac581/Oddities+of+our+HPC](https://confluence.csiro.au/display/~mac581/Oddities+of+our+HPC)