# Important details on running the code
1. The dataset is not included in the repository. It should be stored as follows:
```
data
├── Allen-Cahn_Periodic
│   ├── train
│   │   ├── mesh_7
│   │   │   ├── <folders per parameter set>
│   │   │   │   ├── <npy files>
│   │   ├── mesh_63
│   │   │   ├── <folders per parameter set>
│   │   │   │   ├── <npy files>
│   ├── validation
...
│   ├── test
...
```
**Data processing steps (not required if you already have npy files with train/validation/test split):**<br>
Step 1. Generate the dataset using FEM<br>
Step 2. Convert the generated vtk files to numpy files using `src/vtk-to-npy.py`<br>
Step 3. Split the dataset into train, validation, and test sets using `src/split_dataset.py`<br>

2. Download necessary pretrained SRGAN weight because we bootstrap the generator model from it.
```bash
bash src/SRGAN_download_weights.sh SRGAN_x8-SRGAN_ImageNet
```
3. Install necessary packages mentioned in `requirements.txt` (Python version 3.9.4)
4. Check and configure configuration files at `configs/train/*.yaml`

5. Train and test scripts

Directly run bash script:
```bash
./train_physics.sh
./train_no-physics.sh
./test_physics.sh
./test_no-physics.sh
```
Or, SLURM scripts if you need to submit a job:
```bash
sbatch job_physics.sh
sbatch job_no-physics.sh
```

# Guides on setting up and using environment
My note on using CSIRO Bracewell is [here](https://hasan-rakibul.github.io/csiro-bracewell-for-deep-learning.html).
## Creating python virtual environment
Inside your preferred directory (e.g.,`/scratch2/<ident>`), you can create a virtual environment (instructions can be found [here](https://hasan-rakibul.github.io/personal-note-git-linux-etc-commands.html)).

## FeniCS _(only required for generating the dataset)_
pip install was so much pain; it needs to build from source because of the C++ dependencies. The failed attempt with pip at the end of this document.

Better to install using conda:
```bash
conda create -n fenics python=3.9.4
conda activate fenics
conda install -c conda-forge fenics
```
And it worked like a charm!
  
## CSIRO Bracewell
Python version is **3.9.4**

```bash
module load python/3.9.4
```
Creat and activate a virtual environment.

Install packages:
```bash
python -m pip install -r requirements.txt
```

# Others' code/data _(not required for the final model)_
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
- Split images
```bash
python src/SRGAN_split_images.py
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
- [PSNR and SSIM](https://towardsdatascience.com/super-resolution-a-basic-study-e01af1449e13)

# Notes

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