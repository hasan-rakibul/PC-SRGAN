
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # making it deterministic
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python src/SRGAN_train.py \
--config_path 'configs/train/SRGAN_x8-SRGAN_FEM_1_channel_no-physics.yaml' 
# >&1 | tee train_Allen-Cahn_Periodic-BC_physics_no-normalisation_19Jan.txt