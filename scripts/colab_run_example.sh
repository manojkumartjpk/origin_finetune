#!/usr/bin/env bash
set -euo pipefail

# Example sequence after uploading Roboflow COCO exports.
# Adjust paths to match your Colab Drive mount.

python -m src.data.prepare_from_roboflow_coco_export \
  --export-root data/raw/drywall_join \
  --dataset-tag drywall-join-detect \
  --out-dir data/processed

python -m src.data.prepare_from_roboflow_coco_export \
  --export-root data/raw/cracks \
  --dataset-tag cracks \
  --out-dir data/processed

python -m src.data.merge_manifests \
  --inputs data/processed/manifest_drywall-join-detect.csv data/processed/manifest_cracks.csv \
  --out data/processed/manifest_all.csv

python -m src.data.resplit_manifest \
  --manifest-csv data/processed/manifest_all.csv \
  --out data/processed/manifest_all_resplit.csv \
  --seed 42

python -m src.train_clipseg \
  --manifest-csv data/processed/manifest_all_resplit.csv \
  --output-dir checkpoints/clipseg_takehome \
  --epochs 8 --batch-size 4 --image-size 352

python -m src.eval_clipseg \
  --manifest-csv data/processed/manifest_all_resplit.csv \
  --model-dir checkpoints/clipseg_takehome \
  --split test \
  --save-vis-dir outputs/eval_vis
