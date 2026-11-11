#!/usr/bin/env bash
# Full pipeline on synthetic data (about 2 minutes on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")"

python scripts/generate_synthetic_data.py --n-people 1000 --out-dir data/synthetic
python scripts/build_dataset.py \
    --data data/synthetic/social_care_synthetic.xlsx \
    --embeddings data/synthetic/other_condition_embeddings.pkl \
    --out data/synthetic/landmarks.pkl
python scripts/train_evaluate.py --data data/synthetic/landmarks.pkl --out-dir results/synthetic
