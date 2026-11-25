#!/bin/bash
# Env A: ligand prep + ProLIF + PyRosetta (CPU; no torch/jax)
set -euo pipefail
export CONDA_PKGS_DIRS=/work/hdd/bisa/sshah21/conda_pkgs
export PIP_CACHE_DIR=/work/hdd/bisa/sshah21/pip_cache
ENV=/work/hdd/bisa/sshah21/envs/structprep
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh

mamba create -y -p "$ENV" -c conda-forge \
  python=3.11 pip numpy pandas scipy \
  rdkit openbabel mdanalysis biopython gemmi \
  openmm pdbfixer zstandard

conda activate "$ENV"
pip install --no-input prolif pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(skip_if_installed=True)"
echo "STRUCTPREP_BUILD_OK"
