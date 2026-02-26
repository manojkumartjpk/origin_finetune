# Results Tracking

Use this directory for small, git-tracked artifacts only:
- metrics JSONs
- summary JSONs
- comparison CSVs
- selected report visuals (a few PNGs)
- notes/manifests for experiments

Do not store large artifacts here:
- raw datasets
- processed datasets
- checkpoints/model weights
- full prediction dumps

Suggested layout:
- `results/baselines/<run_id>/`
- `results/finetuned/<run_id>/`
- `results/experiments/<run_id>/`

