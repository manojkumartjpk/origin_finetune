from __future__ import annotations

import json
from pathlib import Path


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(
        _md(
            """# Origin Take-home: Prompted Segmentation (Structured Workflow)

This notebook is organized to preserve *git-trackable* results at each stage:
1. Dataset prep
2. CLIPSeg zero-shot baseline
3. SAM3 baseline (optional, if access/setup works)
4. CLIPSeg fine-tuning
5. Improvement experiments (threshold/epochs/image size/etc.)

Use `results/` for small artifacts only (metrics JSON, notes, selected visuals).
Do **not** commit datasets/checkpoints.
"""
        )
    )

    cells.append(
        _code(
            """#@title 1) Mount Drive (optional but recommended)
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
"""
        )
    )

    cells.append(
        _code(
            """#@title 2) Clone / update repo
from pathlib import Path

REPO_URL = "https://github.com/<your-username>/origin_finetune.git"  # @param {type:"string"}
BRANCH = "main"  # @param {type:"string"}

BASE_DIR = Path('/content')
if USE_DRIVE:
    BASE_DIR = Path('/content/drive/MyDrive/origin_takehome')
BASE_DIR.mkdir(parents=True, exist_ok=True)

REPO_DIR = BASE_DIR / 'origin_finetune'
if REPO_DIR.exists() and (REPO_DIR / '.git').exists():
    %cd {REPO_DIR}
    !git fetch origin
    !git checkout {BRANCH}
    !git pull --ff-only origin {BRANCH}
else:
    %cd {BASE_DIR}
    !git clone -b {BRANCH} {REPO_URL} {REPO_DIR}
    %cd {REPO_DIR}

print("Repo dir:", REPO_DIR)
"""
        )
    )

    cells.append(
        _code(
            """#@title 3) Install dependencies
%cd {REPO_DIR}
!python -m pip install -q --upgrade pip
!pip install -q -r requirements.txt roboflow requests
"""
        )
    )

    cells.append(
        _code(
            """#@title 4) Create local artifact folders (tracked + untracked)
%cd {REPO_DIR}
!mkdir -p outputs/metrics outputs/eval_vis outputs/report_panels results/baselines results/finetuned results/experiments
"""
        )
    )

    cells.append(
        _code(
            """#@title 5) Enter Roboflow API key (hidden input)
import os
from getpass import getpass

if 'ROBOFLOW_API_KEY' not in os.environ or not os.environ['ROBOFLOW_API_KEY']:
    os.environ['ROBOFLOW_API_KEY'] = getpass('Enter ROBOFLOW_API_KEY: ')
print('API key set:', bool(os.environ.get('ROBOFLOW_API_KEY')))
"""
        )
    )

    cells.append(
        _code(
            """#@title 6) Download Roboflow datasets (forked versions)
%cd {REPO_DIR}
!mkdir -p data/raw

!python -m src.data.download_roboflow \
    --api-key "$ROBOFLOW_API_KEY" \
    --workspace "manojs-workspace-mbjw9" \
    --project "drywall-join-detect-jdsh1" \
    --version 1 \
    --format coco \
    --out-dir data/raw/drywall_join

!python -m src.data.download_roboflow \
    --api-key "$ROBOFLOW_API_KEY" \
    --workspace "manojs-workspace-mbjw9" \
    --project "cracks-3ii36-9iz5c" \
    --version 1 \
    --format coco \
    --out-dir data/raw/cracks
"""
        )
    )

    cells.append(
        _code(
            """#@title 7) Convert Roboflow COCO exports -> merged manifest -> label-aware resplit
%cd {REPO_DIR}
!mkdir -p data/processed

!python -m src.data.prepare_from_roboflow_coco_export \
    --export-root data/raw/drywall_join \
    --dataset-tag drywall-join-detect \
    --out-dir data/processed

!python -m src.data.prepare_from_roboflow_coco_export \
    --export-root data/raw/cracks \
    --dataset-tag cracks \
    --out-dir data/processed

!python -m src.data.merge_manifests \
    --inputs data/processed/manifest_drywall-join-detect.csv data/processed/manifest_cracks.csv \
    --out data/processed/manifest_all.csv

!python -m src.data.resplit_manifest \
    --manifest-csv data/processed/manifest_all.csv \
    --out data/processed/manifest_all_resplit.csv \
    --seed 42
"""
        )
    )

    cells.append(
        _md(
            """## CLIPSeg Baseline (Zero-shot)
Run this first and archive metrics to `results/baselines/` before any fine-tuning.
"""
        )
    )

    cells.append(
        _code(
            """#@title 8) Evaluate zero-shot CLIPSeg baseline
%cd {REPO_DIR}
!python -m src.eval_clipseg \
  --manifest-csv data/processed/manifest_all_resplit.csv \
  --model-dir CIDAS/clipseg-rd64-refined \
  --split test \
  --save-vis-dir outputs/eval_vis_clipseg_zeroshot \
  --max-vis 4 \
  --metrics-out outputs/metrics/clipseg_zeroshot_test.json
"""
        )
    )

    cells.append(
        _code(
            """#@title 9) Archive zero-shot CLIPSeg baseline artifacts into tracked results/
%cd {REPO_DIR}
!python scripts/archive_experiment.py \
  --category baselines \
  --run-id clipseg_zeroshot_v1 \
  --summary-json outputs/metrics/clipseg_zeroshot_test.json \
  --copy outputs/eval_vis_clipseg_zeroshot \
  --notes "Zero-shot CLIPSeg baseline on manifest_all_resplit test split"
"""
        )
    )

    cells.append(
        _md(
            """## SAM3 Baseline (Optional / Stretch)
Use this section only if SAM3 access + setup works in Colab. Keep CLIPSeg zero-shot and fine-tuned results as the primary deliverables.

Suggested policy:
- First run a **subset** (e.g. 50-100 test images)
- Save metrics in the same JSON schema as `src.eval_clipseg.py`
- Archive to `results/baselines/sam3_zeroshot_*`
"""
        )
    )

    cells.append(
        _code(
            """#@title 10) SAM3 baseline placeholder (setup + notes)
# Fill this section only if SAM3 access/setup works before the deadline.
# Recommended outputs:
# - outputs/metrics/sam3_zeroshot_test.json
# - outputs/eval_vis_sam3_zeroshot/
print("SAM3 baseline placeholder: implement only if setup succeeds within time budget.")
"""
        )
    )

    cells.append(
        _code(
            """#@title 11) Archive SAM3 baseline (run only after SAM3 metrics exist)
%cd {REPO_DIR}
!python scripts/archive_experiment.py \
  --category baselines \
  --run-id sam3_zeroshot_v1 \
  --summary-json outputs/metrics/sam3_zeroshot_test.json \
  --copy outputs/eval_vis_sam3_zeroshot \
  --notes "SAM3 zero-shot baseline (subset or full test; note exact scope in report)"
"""
        )
    )

    cells.append(
        _md(
            """## CLIPSeg Fine-tuning (Main Result)
Use unique output directories per experiment to avoid overwriting.
"""
        )
    )

    cells.append(
        _code(
            """#@title 12) Fine-tune CLIPSeg (main run)
%cd {REPO_DIR}
MAIN_RUN_ID = "clipseg_ft_e8_352_v1"  # @param {type:"string"}
FT_OUTPUT_DIR = f"checkpoints/{MAIN_RUN_ID}"
print("FT_OUTPUT_DIR =", FT_OUTPUT_DIR)

!python -m src.train_clipseg \
    --manifest-csv data/processed/manifest_all_resplit.csv \
    --output-dir {FT_OUTPUT_DIR} \
    --epochs 8 \
    --batch-size 4 \
    --image-size 352 \
    --lr 2e-5 \
    --grad-accum-steps 1
"""
        )
    )

    cells.append(
        _code(
            """#@title 13) Evaluate fine-tuned CLIPSeg (main run)
%cd {REPO_DIR}
MAIN_RUN_ID = "clipseg_ft_e8_352_v1"  # must match previous cell if rerun separately
FT_OUTPUT_DIR = f"checkpoints/{MAIN_RUN_ID}"
FT_EVAL_VIS_DIR = f"outputs/eval_vis_{MAIN_RUN_ID}"
FT_METRICS_OUT = f"outputs/metrics/{MAIN_RUN_ID}_test.json"

!python -m src.eval_clipseg \
  --manifest-csv data/processed/manifest_all_resplit.csv \
  --model-dir {FT_OUTPUT_DIR} \
  --split test \
  --save-vis-dir {FT_EVAL_VIS_DIR} \
  --max-vis 4 \
  --metrics-out {FT_METRICS_OUT}
"""
        )
    )

    cells.append(
        _code(
            """#@title 14) Create side-by-side report panels (orig | GT | pred) for main run
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

repo = REPO_DIR
MAIN_RUN_ID = "clipseg_ft_e8_352_v1"  # @param {type:"string"}
manifest = pd.read_csv(repo / 'data/processed/manifest_all_resplit.csv')
test_df = manifest[manifest['split'] == 'test'].copy()
pred_dir = repo / 'outputs' / f'eval_vis_{MAIN_RUN_ID}'
panel_dir = repo / 'outputs' / f'report_panels_{MAIN_RUN_ID}'
panel_dir.mkdir(parents=True, exist_ok=True)

saved = 0
for _, row in test_df.iterrows():
    image_id = str(row['image_id'])
    pred_path = pred_dir / f"{image_id}__pred.png"
    gt_path = pred_dir / f"{image_id}__gt.png"
    if not pred_path.exists() or not gt_path.exists():
        continue
    img = Image.open(row['image_path']).convert('RGB')
    gt = Image.open(gt_path).convert('L')
    pred = Image.open(pred_path).convert('L')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(gt, cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
    axes[2].imshow(pred, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
    fig.suptitle(f"{image_id} | {row['label']}")
    out = panel_dir / f"{image_id}__panel.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("saved", out)
    saved += 1
    if saved >= 4:
        break
print("panels saved:", saved)
"""
        )
    )

    cells.append(
        _code(
            """#@title 15) Model size (runtime & footprint section) for main run
%cd {REPO_DIR}
MAIN_RUN_ID = "clipseg_ft_e8_352_v1"  # @param {type:"string"}
!du -sh checkpoints/{MAIN_RUN_ID}
!du -sh checkpoints/{MAIN_RUN_ID}/*
"""
        )
    )

    cells.append(
        _code(
            """#@title 16) Archive fine-tuned CLIPSeg main run artifacts into tracked results/
%cd {REPO_DIR}
MAIN_RUN_ID = "clipseg_ft_e8_352_v1"  # @param {type:"string"}
!python scripts/archive_experiment.py \
  --category finetuned \
  --run-id {MAIN_RUN_ID} \
  --summary-json outputs/metrics/{MAIN_RUN_ID}_test.json \
  --copy checkpoints/{MAIN_RUN_ID}/best_metrics.json checkpoints/{MAIN_RUN_ID}/train_history.json outputs/report_panels_{MAIN_RUN_ID} \
  --notes "Main fine-tuned CLIPSeg run (full fine-tuning, 8 epochs, image_size=352)"
"""
        )
    )

    cells.append(
        _md(
            """## Improvement Experiments (run one at a time, archive each)
Recommended order:
1. Threshold sweep (cheap)
2. More epochs (12/16)
3. Larger image size (512, smaller batch)
4. Class balancing / weighting
5. Prompt augmentation expansion
6. Post-processing
"""
        )
    )

    cells.append(
        _code(
            """#@title 17) Threshold sweep helper on main checkpoint (cheap improvement test)
%cd {REPO_DIR}
MAIN_RUN_ID = "clipseg_ft_e8_352_v1"  # @param {type:"string"}
for thr in [0.3, 0.4, 0.5, 0.6]:
    out_json = f"outputs/metrics/{MAIN_RUN_ID}_test_thr{str(thr).replace('.', '')}.json"
    print("\\n=== threshold", thr, "===")
    !python -m src.eval_clipseg \
      --manifest-csv data/processed/manifest_all_resplit.csv \
      --model-dir checkpoints/{MAIN_RUN_ID} \
      --split test \
      --threshold {thr} \
      --metrics-out {out_json}
"""
        )
    )

    cells.append(
        _code(
            """#@title 18) Custom CLIPSeg experiment template (new checkpoint dir each time)
%cd {REPO_DIR}
EXP_RUN_ID = "clipseg_ft_e12_352_trial1"  # @param {type:"string"}
EXP_EPOCHS = 12  # @param {type:"integer"}
EXP_IMAGE_SIZE = 352  # @param {type:"integer"}
EXP_BATCH = 4  # @param {type:"integer"}
EXP_LR = 2e-5  # @param {type:"number"}

!python -m src.train_clipseg \
    --manifest-csv data/processed/manifest_all_resplit.csv \
    --output-dir checkpoints/{EXP_RUN_ID} \
    --epochs {EXP_EPOCHS} \
    --batch-size {EXP_BATCH} \
    --image-size {EXP_IMAGE_SIZE} \
    --lr {EXP_LR}

!python -m src.eval_clipseg \
    --manifest-csv data/processed/manifest_all_resplit.csv \
    --model-dir checkpoints/{EXP_RUN_ID} \
    --split test \
    --metrics-out outputs/metrics/{EXP_RUN_ID}_test.json

!python scripts/archive_experiment.py \
  --category experiments \
  --run-id {EXP_RUN_ID} \
  --summary-json outputs/metrics/{EXP_RUN_ID}_test.json \
  --copy checkpoints/{EXP_RUN_ID}/best_metrics.json checkpoints/{EXP_RUN_ID}/train_history.json \
  --notes "Custom experiment; record changes in run_id and report notes"
"""
        )
    )

    cells.append(
        _code(
            """#@title 19) Example inference for required output mask naming (main run)
%cd {REPO_DIR}
import pandas as pd
from pathlib import Path

MAIN_RUN_ID = "clipseg_ft_e8_352_v1"
m = pd.read_csv('data/processed/manifest_all_resplit.csv')
row = m.iloc[0]
print('Example image path:', row['image_path'])
print('Example label:', row['label'])

# Uncomment one:
# !python -m src.infer_clipseg --model-dir checkpoints/{MAIN_RUN_ID} --image "{row['image_path']}" --prompt "segment crack" --out-dir outputs/pred_masks
# !python -m src.infer_clipseg --model-dir checkpoints/{MAIN_RUN_ID} --image "{row['image_path']}" --prompt "segment taping area" --out-dir outputs/pred_masks
"""
        )
    )

    cells.append(
        _code(
            """#@title 20) Inspect tracked result artifacts before committing
%cd {REPO_DIR}
!find results -maxdepth 4 -type f | sort
!git status --short
"""
        )
    )

    cells.append(
        _md(
            """## Optional: Commit/Pull/Push from Colab
If you want to push result artifacts from Colab, authenticate carefully (PAT/token). Prefer committing only notebook/code/`results/` files.

If you prefer safer workflow:
- finish runs in Colab
- copy/pull changed repo files locally
- commit/push from local machine
"""
        )
    )

    cells.append(
        _md(
            """## Submission Checklist
- GitHub link (codebase)
- Colab link (shareable)
- PDF report with:
  - Methodology
  - Data-preparation
  - Results (metrics table + visuals + runtime/footprint)
  - Failure cases + potential solutions

Security note: rotate/revoke your Roboflow API key after submission.
"""
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "origin_takehome_clipseg.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out_path = Path("colab_origin_takehome.ipynb")
    out_path.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

