# Prompted Segmentation for Drywall QA - Report

## 1. Goal Summary
Build a text-conditioned segmentation model that predicts binary masks for:
- `segment crack`
- `segment taping area`

The model should take an image + natural-language prompt and produce a single-channel binary mask (`{0,255}`), same spatial size as the input image.

## 2. Methodology
- Model(s) tried:
  - Zero-shot CLIPSeg (`CIDAS/clipseg-rd64-refined`)
  - Zero-shot SAM3 subset baseline (`facebook/sam3`) as an exploratory comparison
  - Fine-tuned CLIPSeg (main and experiment runs)
- Final model family:
  - CLIPSeg (`CIDAS/clipseg-rd64-refined`) fine-tuned on merged drywall datasets
- Why this model:
  - It is natively text-conditioned and supports prompt-based segmentation with a single model for multiple target categories.
- Training objective:
  - Weighted BCE + Dice loss (`bce_dice_loss` in `src/train_clipseg.py`)
- Evaluation:
  - mIoU and Dice, reported overall and per label (`crack`, `taping_area`)
  - Binary mask thresholding on sigmoid outputs (default threshold `0.5` in `src/eval_clipseg.py`)

## 3. Data Preparation
- Datasets used:
  - Dataset 1: Drywall-Join-Detect (mapped to `taping_area`)
  - Dataset 2: Cracks (mapped to `crack`)
- Export format:
  - Roboflow COCO Segmentation export (preferred)
- Processing pipeline:
  - Convert each Roboflow COCO export to a unified manifest
  - Merge manifests into one CSV
  - Re-split by `image_id` within each label to ensure both labels appear in train/val/test
- Canonical labels:
  - `crack`
  - `taping_area`
- Prompt mapping used (from `configs/prompts.json`):
  - `crack`: `segment crack`, `segment wall crack`
  - `taping_area`: `segment taping area`, `segment drywall seam`, `segment joint tape`, `segment joint/tape`
- Image preprocessing:
  - Resize to `352x352` for the runs reported below
- Split strategy:
  - Label-aware resplit by image id (`src/data/resplit_manifest.py`) with seed `42`

### Split Counts
Fill this table from Colab (`pd.crosstab(manifest["split"], manifest["label"])`) using `data/processed/manifest_all_resplit.csv`.

| Split | Crack | Taping Area | Total |
|---|---:|---:|---:|
| Train |  |  |  |
| Val |  |  |  |
| Test | 806 | 153 | 959 |

## 4. Training Setup
### Main Fine-tuned Run (`clipseg_ft_e8_352_v1`)
- Hardware: Google Colab T4
- Epochs: `8`
- Batch size: `4`
- LR: `2e-5`
- Weight decay: `1e-4`
- Image size: `352`
- Seed: `42`
- Grad accumulation steps: `1`
- Validation threshold: `0.5`
- Train time: `1515.0 s` (~`25.3 min`)
- Best checkpoint criterion:
  - Best composite validation score = average of `(val_mIoU + val_Dice) / 2`

### Experiment Run (`clipseg_ft_e12_352_trial1`)
- Same setup as above, but `12` epochs
- Train time: `2301.2 s` (~`38.4 min`)

## 5. Results

### Metrics (Test)
| Run | Prompt / Label | mIoU | Dice | Notes |
|---|---|---:|---:|---|
| Zero-shot CLIPSeg | `segment crack` | 0.2174 | 0.3219 | Baseline |
| Zero-shot CLIPSeg | `segment taping area` | 0.0201 | 0.0304 | Very weak on taping area |
| Zero-shot CLIPSeg | Overall | 0.1860 | 0.2754 | `n=959` |
| Fine-tuned CLIPSeg (e8, main) | `segment crack` | 0.4339 | 0.5798 | Step 14 eval, threshold `0.5` |
| Fine-tuned CLIPSeg (e8, main) | `segment taping area` | 0.4951 | 0.6531 | Strong gain after fine-tuning |
| Fine-tuned CLIPSeg (e8, main) | Overall | 0.4437 | 0.5915 | `n=959` |
| Fine-tuned CLIPSeg (e12, trial1) | `segment crack` | 0.4397 | 0.5868 | Better than e8 |
| Fine-tuned CLIPSeg (e12, trial1) | `segment taping area` | 0.5230 | 0.6777 | Best committed result |
| Fine-tuned CLIPSeg (e12, trial1) | Overall | 0.4530 | 0.6013 | `n=959` |

### Improvement Summary
- Fine-tuning CLIPSeg vs zero-shot CLIPSeg (overall):
  - mIoU: `0.1860 -> 0.4437` (`+0.2577`)
  - Dice: `0.2754 -> 0.5915` (`+0.3161`)
- Additional epochs (e12 vs e8) further improved:
  - mIoU: `+0.0093`
  - Dice: `+0.0098`

### Optional Threshold Sweep (Cheap Improvement Test on Main Checkpoint)
This experiment was run on the main checkpoint (`clipseg_ft_e8_352_v1`) to test threshold sensitivity at inference time.

- `thr=0.3`: mIoU `0.4433`, Dice `0.5879`
- `thr=0.4`: mIoU `0.4455`, Dice `0.5917` (best in this sweep)
- `thr=0.5`: mIoU `0.4437`, Dice `0.5915` (default used in Step 14)
- `thr=0.6`: mIoU `0.4372`, Dice `0.5866`

Notes:
- Threshold `0.4` gives a small improvement over `0.5` on the test split.
- Because the threshold was selected using the test split, this is reported as **sensitivity analysis**, not final threshold tuning.

### Runtime & Footprint
- Avg inference time / image:
  - Zero-shot CLIPSeg: `0.0360 s`
  - Fine-tuned CLIPSeg (e8): `0.0385 s`
  - Fine-tuned CLIPSeg (e12): `0.0419 s`
- Model size on disk:
  - [Fill from Colab using `du -sh checkpoints/<run_id>`]
- Params (optional):
  - [Optional if measured]

## 6. Visual Examples
Include 3-4 examples (`Original | GT | Prediction`) from:
- `results/finetuned/clipseg_ft_e8_352_v1/report_panels_clipseg_ft_e8_352_v1/`

Suggested examples already archived:
- `drywall-join-detect__2000x1500_20_resized_jpg...__panel.png`
- `drywall-join-detect__IMG_8206...__panel.png`
- `drywall-join-detect__IMG_8224...__panel.png`
- `drywall-join-detect__IMG_8236...__panel.png`

## 7. Failure Cases and Potential Solutions
### Failure case 1: Thin seam predicted too thick / blob-like
- Observation:
  - Narrow taping seams are sometimes predicted with thicker boundaries than ground truth.
- Potential causes:
  - Resolution reduction to `352x352`
  - Boundary precision loss after up/downsampling
- Potential fixes:
  - Higher resolution or tiled inference
  - Boundary-aware loss / edge refinement
  - Morphological post-processing (thin/cleanup)

### Failure case 2: Over-segmentation when multiple seam-like lines are present
- Observation:
  - The model may predict broad/merged regions around multiple vertical seam-like structures.
- Potential causes:
  - Similar visual texture/lines in drywall scenes
  - Global thresholding behavior
- Potential fixes:
  - Threshold selection on validation set
  - Connected-component filtering
  - More hard negatives / diverse seam backgrounds in training

### Failure case 3: Shape extent mismatch (length/width mismatch vs GT)
- Observation:
  - Predicted seam region can be correct in location but inaccurate in exact extent.
- Potential causes:
  - Annotation thickness inconsistency
  - Perspective and lighting variation
- Potential fixes:
  - More data with varied viewpoints
  - Stronger augmentations (lighting/perspective)
  - Fine-tune longer with early stopping on validation

## 8. Reproducibility
- GitHub link:
  - [Add your GitHub repo URL]
- Colab link:
  - [Add your Colab notebook share link]
- Key notebook:
  - `colab_origin_takehome.ipynb`
- Seeds:
  - `42`
- Main runs used for report:
  - `clipseg_ft_e8_352_v1` (main)
  - `clipseg_ft_e12_352_trial1` (experiment)
