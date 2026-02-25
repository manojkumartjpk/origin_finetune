# Origin Take-home: Prompted Segmentation for Drywall QA

Colab-friendly pipeline to fine-tune a text-conditioned segmentation model (`CLIPSeg`) for:
- `segment crack`
- `segment taping area`

## What this repo includes
- Roboflow COCO export (split folders) -> binary mask conversion (`src/data/prepare_from_roboflow_coco_export.py`)
- Generic single-COCO conversion utility (`src/data/prepare_from_coco.py`)
- Unified manifest generation (`src/data/merge_manifests.py`)
- Label-aware re-splitting to ensure both prompts appear in train/val/test (`src/data/resplit_manifest.py`)
- CLIPSeg fine-tuning (`src/train_clipseg.py`)
- Evaluation (mIoU / Dice + runtime) (`src/eval_clipseg.py`)
- Inference -> required PNG masks (`src/infer_clipseg.py`)
- Report template (`report/report_template.md`)

## Recommended workflow (Google Colab T4)
1. Create a new Colab notebook (GPU: T4).
2. Clone/upload this repo.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Export both Roboflow datasets in **COCO Segmentation** format (preferred), upload into:
- `data/raw/drywall_join/`
- `data/raw/cracks/`

5. Run the end-to-end commands in `scripts/colab_run_example.sh` (adjust paths as needed).

## Expected output mask format
- PNG, single channel (`L`)
- same spatial size as source image
- pixel values `{0, 255}`
- file name format: `imageid__segment_crack.png`

## Notes / assumptions
- If Roboflow export contains bounding boxes instead of polygon masks, the converters fall back to box masks.
- This should be clearly documented in the report because it affects the best possible segmentation quality.

## Example commands
### Convert Roboflow exports
```bash
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
```

### Train
```bash
python -m src.train_clipseg \
  --manifest-csv data/processed/manifest_all_resplit.csv \
  --output-dir checkpoints/clipseg_takehome \
  --epochs 8 \
  --batch-size 4 \
  --image-size 352
```

### Evaluate
```bash
python -m src.eval_clipseg \
  --manifest-csv data/processed/manifest_all_resplit.csv \
  --model-dir checkpoints/clipseg_takehome \
  --split test \
  --save-vis-dir outputs/eval_vis
```

### Inference
```bash
python -m src.infer_clipseg \
  --model-dir checkpoints/clipseg_takehome \
  --image path/to/image.jpg \
  --prompt "segment crack" \
  --out-dir outputs/pred_masks
```

## Submission checklist
- GitHub link (this repo)
- PDF report (`report/report_template.md` -> PDF)
- Visual examples (`orig | GT | pred`)
- Metrics table (mIoU, Dice)
- Failure cases and fixes
- Runtime + model footprint
