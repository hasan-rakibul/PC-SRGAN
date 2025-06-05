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
3. To directly test the model, pretrained checkpoints of four experimental setups are available [here in Google Drive](https://drive.google.com/drive/folders/1MkQsvRpItVb7VaShLBVHgFmWQkbcTK1B?usp=sharing). Download and store them as the `results` directory.
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
Better to install using conda:
```bash
conda create -n fenics python=3.9.4
conda activate fenics
conda install -c conda-forge fenics
```

## CSIRO Bracewell
Python version is **3.9.4**

```bash
module load python/3.9.4
```
Creat and activate a virtual environment.

Install packages:
```bash
python -m pip install -r requirements_torch.txt
python -m pip install -r requirements.txt
```

# Acknowledgement
We are immensely grateful to the contributors of [SRGAN-PyTorch](https://github.com/Lornatang/SRGAN-PyTorch) based on which we have developed our codebase.