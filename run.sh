#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sky_sanctuary_gs
python src/pipeline.py --mode full --out outputs --stage build
for start in $(seq 0 4 99); do
  end=$((start + 4))
  python src/pipeline.py --mode full --out outputs --stage render \
    --frame-start "$start" --frame-end "$end"
done
python src/pipeline.py --mode full --out outputs --stage encode
