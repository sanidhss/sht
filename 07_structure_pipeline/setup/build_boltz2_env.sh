#!/bin/bash
# Env B: Boltz-2 (GPU torch; isolated from structprep)
set -euo pipefail
export CONDA_PKGS_DIRS=/work/hdd/bisa/sshah21/conda_pkgs
export PIP_CACHE_DIR=/work/hdd/bisa/sshah21/pip_cache
ENV=/work/hdd/bisa/sshah21/envs/boltz2
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh

mamba create -y -p "$ENV" -c conda-forge python=3.11 pip
conda activate "$ENV"
pip install --no-input boltz -U
echo "BOLTZ2_BUILD_OK"
