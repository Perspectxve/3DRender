#!/usr/bin/env bash
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sky_sanctuary_gs
python src/pipeline.py --mode quick --out outputs_quick --stage build
for start in 0 12 24; do
  end=$((start + 12))
  python src/pipeline.py --mode quick --out outputs_quick --stage render \
    --frame-start "$start" --frame-end "$end"
done
python src/pipeline.py --mode quick --out outputs_quick --stage encode
