
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # making it deterministic

python src/train_gan.py \
--config_path 'configs/train/SRGAN_x8-SRGAN_FEM_1_channel_physics.yaml' >&1 | tee train_GAN_FEM_physics_no-freeze_no-warmup_09Jan.txt