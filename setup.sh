#!/usr/bin/env bash
set -eo pipefail
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q '^sky_sanctuary_gs '; then
  echo "Environment sky_sanctuary_gs already exists."
else
  conda env create -f environment.yml
fi

conda activate sky_sanctuary_gs
python -m pip install --upgrade pip
python -m pip install \
  "numpy>=1.26,<3" \
  "Pillow>=10.2,<12" \
  "imageio>=2.34,<3" \
  "imageio-ffmpeg>=0.5,<1" \
  "scipy>=1.11,<2" \
  "torch>=2.2,<3" \
  "tqdm>=4.66,<5"
