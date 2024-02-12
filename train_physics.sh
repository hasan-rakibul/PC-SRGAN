
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # making it deterministic
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python src/SRGAN_train.py \
--config_path 'configs/train/Erikson-Johnson_Subset70_Physics.yaml'
# >&1 | tee _19Jan.txt