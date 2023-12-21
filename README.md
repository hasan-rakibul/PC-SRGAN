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

## FeniCS
pip install is so much pain; it needs to build from source because of the C++ dependencies. The failed attempt with pip at the end of this document.

Better to install using conda:
```bash
conda create -n fenics python=3.9.4
conda activate fenics
conda install -c conda-forge fenics
```
And it worked like a charm!

In **CSIRO Bracewell**, I can have minoconda using `module load miniconda3`

As I should not install in home directory, let's change them in conda configuration.

`conda config` will create a `.condarc` file in home directory. Open it and add the following lines:
```bash
envs_dirs:
  - /scratch2/<ident>/.conda/envs
pkgs_dirs:
  - /scratch2/<ident>/.conda/pkgs
```
Then install as above.
  
## CSIRO Bracewell
Python version is **3.9.4**

```bash
module load python/3.9.4
```
Creat and activate a virtual environment (as described earlier).

Install packages:
```bash
python -m pip install -r requirements-SRGAN.txt
```

## Local machine

I first install python3.9 in Ubuntu 22.04.3 LTS  

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 python3.9-venv python3.9-distutils
```

Create and activate virtual environment.
```bash
python3.9 -m venv ~/.venv/pinn
source ~/.venv/pinn/bin/activate 
```

Install necessary files on the virtual environment 
```bash
python -m pip install -r requirements-SRGAN.txt
python -m pip install -r requirements.txt 
```

# Others' code/data
## SRGAN
- Download model weights and datasets
```bash
bash src/SRGAN_download_weights.sh SRGAN_x4-SRGAN_ImageNet
bash src/SRGAN_download_weights.sh SRGAN_x8-SRGAN_ImageNet

bash src/SRGAN_download_weights.sh SRResNet_x4-SRGAN_ImageNet
bash src/SRGAN_download_weights.sh SRResNet_x8-SRGAN_ImageNet

bash src/SRGAN_download_weights.sh DiscriminatorForVGG_x4-SRGAN_ImageNet
bash src/SRGAN_download_weights.sh DiscriminatorForVGG_x8-SRGAN_ImageNet

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
- Test
```bash
python src/SRGAN_test.py --config_path ./configs/test/SRGAN_x4-SRGAN_ImageNet-Set5.yaml
python src/SRGAN_test.py --config_path ./configs/test/SRGAN_x8-SRGAN_ImageNet.yaml
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
sudo apt install python3.7 python3.7-distutils
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
- [https://wiki.pathmind.com/generative-adversarial-network-gan](https://wiki.pathmind.com/generative-adversarial-network-gan)

## CSIRO Cluster
- [https://confluence.csiro.au/display/SC/Useful+information+for+new+users](https://confluence.csiro.au/display/SC/Useful+information+for+new+users)
- [https://confluence.csiro.au/display/~mac581/Oddities+of+our+HPC](https://confluence.csiro.au/display/~mac581/Oddities+of+our+HPC)
- [https://confluence.csiro.au/display/SC/Interactive+access+and+visualization](https://confluence.csiro.au/display/SC/Interactive+access+and+visualization)
- Configure VSCode: [https://confluence.csiro.au/display/MLAIFSP/Remote+editing+with+VS+Code+on+bracewell](https://confluence.csiro.au/display/MLAIFSP/Remote+editing+with+VS+Code+on+bracewell)
- Configure Conda: [https://confluence.csiro.au/display/IMT/Conda+and+python+in+HPC](https://confluence.csiro.au/display/IMT/Conda+and+python+in+HPC)

### Submitting batch job
- [https://confluence.csiro.au/display/SC/Sample+Slurm+Job+Scripts](https://confluence.csiro.au/display/SC/Sample+Slurm+Job+Scripts)
- [https://confluence.csiro.au/pages/viewpage.action?pageId=1540489611](https://confluence.csiro.au/pages/viewpage.action?pageId=1540489611)
- [https://confluence.csiro.au/display/VCCRI/SLURM](https://confluence.csiro.au/display/VCCRI/SLURM)
- [https://confluence.csiro.au/display/SC/Requesting+resources+in+Slurm](https://confluence.csiro.au/display/SC/Requesting+resources+in+Slurm)
- [https://confluence.csiro.au/display/SC/Running+jobs+in+an+interactive+batch+shell](https://confluence.csiro.au/display/SC/Running+jobs+in+an+interactive+batch+shell)
- [https://confluence.csiro.au/display/GEES/HPC+Cheat+Sheet](https://confluence.csiro.au/display/GEES/HPC+Cheat+Sheet)

# Notes
## Freezing layers
In addition to controlling `param.requires_grad`, I need to enable `eval()` mode corresponding to `batchnorm layers`, if any. Details: [https://discuss.pytorch.org/t/should-i-use-model-eval-when-i-freeze-batchnorm-layers-to-finetune/39495](https://discuss.pytorch.org/t/should-i-use-model-eval-when-i-freeze-batchnorm-layers-to-finetune/39495)

# Failed attempt to install FeniCS using pip
Taken and adapted from [https://fenics.readthedocs.io/en/latest/installation.html](https://fenics.readthedocs.io/en/latest/installation.html)

```bash
PYBIND11_VERSION=2.2.3
wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz
tar -xf v${PYBIND11_VERSION}.tar.gz && cd pybind11-${PYBIND11_VERSION}
mkdir build && cd build && cmake -DPYBIND11_TEST=off .. && sudo make install

pip3 install fenics-ffc --upgrade
```
Install [boost](https://www.boost.org/) (because the cmake in the below dolfin installation failed, saying boost not found): [here's a tutorial](https://linux.how2shout.com/how-to-install-boost-c-on-ubuntu-20-04-or-22-04/)

Could NOT find Eigen3, so installed using `sudo apt install libeigen3-dev`

There were two files where I needed to add `#include <algorithm>` because it didn't recognise `std::count` and `std::min_element`. YOu will know which files when you get the error.

`endian.hpp` location is changed in recent boost version. In two files, `#include <boost/detail/endian.hpp>` needed to be changed to `#include <boost/predef/other/endian.h>`


```bash
FENICS_VERSION=$(python -c"import ffc; print(ffc.__version__)")
git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/dolfin
git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/mshr
mkdir dolfin/build && cd dolfin/build && cmake .. && sudo make install && cd ../..
mkdir mshr/build   && cd mshr/build   && cmake .. && make install && cd ../..
cd dolfin/python && pip3 install . && cd ../..
cd mshr/python   && pip3 install . && cd ../..
```

Lastly there's the error I coudn't solve: mshr bitbucket link was shown invalid and 'ERROR: Failed building wheel for fenics-dolfin' with `make install` of dolfin.