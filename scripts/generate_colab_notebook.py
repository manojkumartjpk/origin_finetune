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
    cells = []
    cells.append(
        _md(
            """# Origin Take-home: Prompted Segmentation (Colab)

This notebook runs a full pipeline for the take-home assignment:
- clone repo
- install dependencies
- download Roboflow datasets (when possible)
- convert annotations to binary masks
- fine-tune CLIPSeg
- evaluate and export example masks

## Important note (discovered during API inspection)
Both linked projects are currently **object-detection** projects. `drywall-join-detect` has published versions and can be downloaded programmatically. `cracks-3ii36` currently reports **0 published versions**, so SDK download may fail and require **manual Roboflow export** (COCO preferred) or a versioned dataset link.
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
            """#@title 2) Set workspace paths and clone repo
from pathlib import Path

# Replace with your GitHub repo after you push this project
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

print('Repo dir:', REPO_DIR)
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
            """#@title 4) Enter Roboflow API key (hidden input)
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
            """#@title 5) Try programmatic download from Roboflow
# Expected outcome:
# - Drywall dataset downloads successfully (latest version, COCO)
# - Cracks dataset may fail because it has no published versions (versionless project)

%cd {REPO_DIR}
!mkdir -p data/raw

RF_WORKSPACE = "manojs-workspace-mbjw9"  # @param {type:"string"}
DRYWALL_PROJECT = "drywall-join-detect-jdsh1"  # @param {type:"string"}
CRACKS_PROJECT = "cracks-3ii36-9iz5c"  # @param {type:"string"}

!python -m src.data.download_roboflow \\
    --api-key "$ROBOFLOW_API_KEY" \\
    --workspace "$RF_WORKSPACE" \\
    --project "$DRYWALL_PROJECT" \\
    --version 1 \\
    --format coco \\
    --out-dir data/raw/drywall_join

!python -m src.data.download_roboflow \\
    --api-key "$ROBOFLOW_API_KEY" \\
    --workspace "$RF_WORKSPACE" \\
    --project "$CRACKS_PROJECT" \\
    --version 1 \\
    --format coco \\
    --out-dir data/raw/cracks
"""
        )
    )
    cells.append(
        _md(
            """## If cracks download fails (likely)
Use one of these options:
1. In Roboflow UI, export **COCO** (or COCO Segmentation if available) and upload/unzip into `data/raw/cracks/`
2. Ask Origin/recruiter for a downloadable versioned link or export ZIP for `cracks-3ii36`

After that, continue with the next cells.
"""
        )
    )
    cells.append(
        _code(
            """#@title 6) (Optional) Unzip manually uploaded exports into expected folders
import zipfile

DRYWALL_ZIP = ""  # @param {type:"string"}
CRACKS_ZIP = ""  # @param {type:"string"}

for zip_path, out_dir in [(DRYWALL_ZIP, REPO_DIR / 'data/raw/drywall_join'), (CRACKS_ZIP, REPO_DIR / 'data/raw/cracks')]:
    if zip_path:
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        print(f'Extracted {zip_path} -> {out_dir}')
"""
        )
    )
    cells.append(
        _code(
            """#@title 7) Inspect raw dataset folders and find annotation files
%cd {REPO_DIR}
!find data/raw -maxdepth 3 -type f | head -200
"""
        )
    )
    cells.append(
        _code(
            """#@title 8) Convert Roboflow COCO exports to unified binary-mask manifest
            # Update --export-root if your uploaded folder names differ.
%cd {REPO_DIR}
!mkdir -p data/processed

!python -m src.data.prepare_from_roboflow_coco_export \\
    --export-root data/raw/drywall_join \\
    --dataset-tag drywall-join-detect \\
    --out-dir data/processed

!python -m src.data.prepare_from_roboflow_coco_export \\
    --export-root data/raw/cracks \\
    --dataset-tag cracks \\
    --out-dir data/processed

!python -m src.data.merge_manifests \\
    --inputs data/processed/manifest_drywall-join-detect.csv data/processed/manifest_cracks.csv \\
    --out data/processed/manifest_all.csv

!python -m src.data.resplit_manifest \\
    --manifest-csv data/processed/manifest_all.csv \\
    --out data/processed/manifest_all_resplit.csv \\
    --seed 42
"""
        )
    )
    cells.append(
        _code(
            """#@title 9) Train CLIPSeg (T4-friendly baseline)
%cd {REPO_DIR}
!python -m src.train_clipseg \\
    --manifest-csv data/processed/manifest_all_resplit.csv \\
    --output-dir checkpoints/clipseg_takehome \\
    --epochs 8 \\
    --batch-size 4 \\
    --image-size 352 \\
    --lr 2e-5 \\
    --grad-accum-steps 1
"""
        )
    )
    cells.append(
        _code(
            """#@title 10) Evaluate on test split and save sample masks
%cd {REPO_DIR}
!python -m src.eval_clipseg \\
    --manifest-csv data/processed/manifest_all_resplit.csv \\
    --model-dir checkpoints/clipseg_takehome \\
    --split test \\
    --save-vis-dir outputs/eval_vis \\
    --max-vis 4
"""
        )
    )
    cells.append(
        _code(
            """#@title 11) Create side-by-side visuals (orig | GT | pred) for report
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

repo = REPO_DIR
manifest = pd.read_csv(repo / 'data/processed/manifest_all_resplit.csv')
test_df = manifest[manifest['split'] == 'test'].copy()
vis_dir = repo / 'outputs' / 'report_panels'
vis_dir.mkdir(parents=True, exist_ok=True)

pred_dir = repo / 'outputs' / 'eval_vis'
saved = 0
for _, row in test_df.iterrows():
    image_id = str(row['image_id'])
    pred_path = pred_dir / f\"{image_id}__pred.png\"
    gt_path = pred_dir / f\"{image_id}__gt.png\"
    if not pred_path.exists() or not gt_path.exists():
        continue
    img = Image.open(row['image_path']).convert('RGB')
    gt = Image.open(gt_path).convert('L')
    pred = Image.open(pred_path).convert('L')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(gt, cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
    axes[2].imshow(pred, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
    fig.suptitle(f\"{image_id} | {row['label']}\")
    out = vis_dir / f\"{image_id}__panel.png\"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    saved += 1
    if saved >= 4:
        break
print('panels saved:', saved)
"""
        )
    )
    cells.append(
        _code(
            """#@title 12) Example inference for required output mask naming
%cd {REPO_DIR}
!python - <<'PY'
import pandas as pd
m = pd.read_csv('data/processed/manifest_all_resplit.csv')
row = m.iloc[0]
print('Example image path:', row['image_path'])
print('Example label:', row['label'])
PY

# Then run one of these after replacing <IMAGE_PATH>:
# !python -m src.infer_clipseg --model-dir checkpoints/clipseg_takehome --image <IMAGE_PATH> --prompt \"segment crack\" --out-dir outputs/pred_masks
# !python -m src.infer_clipseg --model-dir checkpoints/clipseg_takehome --image <IMAGE_PATH> --prompt \"segment taping area\" --out-dir outputs/pred_masks
"""
        )
    )
    cells.append(
        _md(
            """## Submission checklist
- Push this repo to GitHub and include the link
- Fill `report/report_template.md` with metrics, visuals, failure cases, runtime, and settings
- Export the report to PDF
- Attach both in the email reply thread

## Security note
You shared a Roboflow API key in chat. Rotate/revoke it after your assignment is submitted.
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
