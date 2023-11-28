# physics-nn-fluid-dynamics

# Setting up environment
## Creating python virtual environment
Inside your preferred directory (e.g.,`/scratch2/<ident>`):
```bash
python -m venv .env/pinn/
```
Activate virtual environment. Keep the activation command in your `.bashrc` file:
```bash
source /scratch2/<ident>/.env/pinn/bin/activate
```

Or, if you don't need the environment everytime. Youn can activate in each terminal session using the same command above.

To deactivate the current environment:
```bash
deactivate
```

## CSIRO Bracewell
Here the latest available python version is 3.9.4

```bash
module load python/3.9.4
```
Creat and activate a virtual environment (as described earlier).

Install packages:
```bash
python -m pip install -r requirements-SRGAN.txt
```

## Local machine

I first install python3.11 in Ubuntu 22.04.3 LTS  

```bash
sudo apt install python3.11 
sudo apt install python3.11-venv 
```

Create and activate virtual environment.
```bash
python3.11 -m venv ~/.venv/pinn
source ~/.venv/pinn/bin/activate 
```

Install necessary files on the virtual environment 
```bash
python -m pip install -r requirements-SRGAN.txt
 # python -m pip install -r requirements.txt 
```

# Others' code/data
## SRGAN
- Download model weights and datasets
```bash
bash src/SRGAN_download_weights.sh SRGAN_x4-SRGAN_ImageNet
bash src/SRGAN_download_weights.sh SRResNet_x4-SRGAN_ImageNet
bash src/SRGAN_download_weights.sh DiscriminatorForVGG_x4-SRGAN_ImageNet
bash src/SRGAN_download_datasets.sh SRGAN_ImageNet
bash src/SRGAN_download_datasets.sh Set5
```
- Check configuration at `configs/train/*.yaml`
- Split images
```bash
python src/SRGAN_split_images.py
```
- Train
```bash
python src/train_gan.py
```

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
- [https://confluence.csiro.au/display/SC/Interactive+access+and+visualization](https://confluence.csiro.au/display/SC/Interactive+access+and+visualization)

### Submitting batch job
- [https://confluence.csiro.au/display/SC/Sample+Slurm+Job+Scripts](https://confluence.csiro.au/display/SC/Sample+Slurm+Job+Scripts)