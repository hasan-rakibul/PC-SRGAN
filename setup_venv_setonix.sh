#!/bin/bash

# remove dir if exists
if [ -d ".venv" ]; then
  rm -rf .venv
  echo "Removed existing .venv directory"
fi

module load pytorch/2.2.0-rocm5.7.3

singularity exec $SINGULARITY_CONTAINER bash -c "\
python3 -m venv --system-site-package .venv && \
source .venv/bin/activate && \
python3 -m pip install --upgrade pip && \
python3 -m pip install -r additional_baselines/requirements.txt && \
echo 'Venv setup complete' "
