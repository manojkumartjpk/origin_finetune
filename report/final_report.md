# Prompted Segmentation for Drywall QA - Final Report

## 1. Goal Summary
Build a text-conditioned segmentation model that takes an image and a natural-language prompt, then outputs a binary mask (PNG, single-channel, same size as input, values `{0,255}`) for:
- `segment crack`
- `segment taping area`

## 2. Methodology
- Base approach: prompt-conditioned semantic segmentation with CLIPSeg (`CIDAS/clipseg-rd64-refined`).
- Baselines:
  - Zero-shot CLIPSeg.
  - SAM3 zero-shot subset baseline (`facebook/sam3`, 100 samples) as an exploratory comparison.
- Main training objective: weighted BCE + Dice (`bce_dice_loss` in `src/train_clipseg.py`).
- Validation/Test metrics: mIoU and Dice (overall + per label).
- Inference post-processing: sigmoid + binary threshold (`0.5` default, plus threshold sweep analysis).

Why CLIPSeg:
- It directly supports text-conditioned segmentation with a single model family for both prompts.
- It gave stable improvements after supervised fine-tuning on merged drywall data.

## 3. Data Preparation
## 3.1 Sources and Label Mapping
- Dataset 1: Drywall-Join-Detect -> mapped to label `taping_area`.
- Dataset 2: Cracks -> mapped to label `crack`.
- Export format: Roboflow COCO Segmentation export.

## 3.2 Processing Pipeline
1. Convert each Roboflow COCO export to unified manifest rows (`src/data/prepare_from_roboflow_coco_export.py`).
2. Merge manifests (`src/data/merge_manifests.py`).
3. Re-split by `image_id`, label-aware, seed `42` (`src/data/resplit_manifest.py`), to ensure both labels are present in train/val/test.
4. Use prompts from `configs/prompts.json`.

## 3.3 Prompt Mapping
- `crack`: `segment crack`, `segment wall crack`
- `taping_area`: `segment taping area`, `segment drywall seam`, `segment joint tape`, `segment joint/tape`

## 3.4 Final Split Counts (`manifest_all_resplit.csv`)
| Split | Crack | Taping Area | Total |
|---|---:|---:|---:|
| Train | 3758 | 714 | 4472 |
| Val | 805 | 153 | 958 |
| Test | 806 | 153 | 959 |
| Total | 5369 | 1020 | 6389 |

## 4. Training Setup
- Environment: Google Colab GPU (T4).
- Seed: `42`.
- Optimizer setup: LR `2e-5`, weight decay `1e-4`.
- Batch size: `16` for `352x352` runs.
- Main fine-tuned runs:
  - `clipseg_ft_e8_352_v1`: 8 epochs, image size 352.
  - `clipseg_ft_e12_352_trial1`: 12 epochs, image size 352.
- Additional experiments:
  - `clipseg_ft_e16_512_thr04_v1`: 16 epochs, image size 512, threshold 0.4.
  - `clipseg_ft_e4_1024_thr04_v1`: 4 epochs, image size 1024, threshold 0.4.
  - `clipseg_ft_e8_512_thr04_baltrain_dataset_v1`: 8 epochs, 512, balanced train by dataset.

## 5. Results
## 5.1 Core Metrics on Test Split
| Run | mIoU (overall) | Dice (overall) | Crack mIoU / Dice | Taping mIoU / Dice | n |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIPSeg | 0.1860 | 0.2754 | 0.2175 / 0.3219 | 0.0201 / 0.0304 | 959 |
| Fine-tuned CLIPSeg `e8_352` | 0.4149 | 0.5618 | 0.4155 / 0.5607 | 0.4115 / 0.5675 | 959 |
| Fine-tuned CLIPSeg `e12_352` | 0.4284 | 0.5770 | 0.4219 / 0.5686 | 0.4625 / 0.6216 | 959 |
| Fine-tuned CLIPSeg `e16_512_thr04` | **0.4701** | **0.6160** | 0.4608 / 0.6054 | 0.5194 / 0.6718 | 959 |
| Fine-tuned CLIPSeg `e4_1024_thr04` | 0.4430 | 0.5920 | 0.4465 / 0.5938 | 0.4247 / 0.5826 | 959 |
| Fine-tuned CLIPSeg `e8_512_thr04_baltrain` | 0.4096 | 0.5555 | 0.4057 / 0.5496 | 0.4300 / 0.5866 | 959 |
| SAM3 zero-shot subset100 | ~0.0000 | ~0.0000 | N/A | ~0.0000 / ~0.0000 | 100 |

## 5.2 Improvement Summary
- `e8_352` vs zero-shot CLIPSeg:
  - mIoU: `0.1860 -> 0.4149` (`+0.2289`)
  - Dice: `0.2754 -> 0.5618` (`+0.2864`)
- Best run (`e16_512_thr04`) vs zero-shot CLIPSeg:
  - mIoU: `0.1860 -> 0.4701` (`+0.2841`)
  - Dice: `0.2754 -> 0.6160` (`+0.3406`)

## 5.3 Threshold Sensitivity (Main `e8_352` Checkpoint)
- `thr=0.3`: mIoU `0.4134`, Dice `0.5570`
- `thr=0.4`: mIoU `0.4167`, Dice `0.5621` (best in this sweep)
- `thr=0.5`: mIoU `0.4149`, Dice `0.5618` (default)
- `thr=0.6`: mIoU `0.4082`, Dice `0.5564`

Note: This was run on the test split, so it is sensitivity analysis, not final threshold tuning protocol.

## 5.4 Runtime and Footprint
- Train time:
  - `e8_352`: `224.2 s` (~3.7 min)
  - `e12_352`: `334.9 s` (~5.6 min)
  - `e16_512`: `1020.7 s` (~17.0 min)
  - `e4_1024`: `1985.2 s` (~33.1 min)
- Avg inference time/image:
  - Zero-shot CLIPSeg: `0.0089 s`
  - `e8_352`: `0.0088 s`
  - `e12_352`: `0.0089 s`
  - `e16_512`: `0.0237 s`
  - `e4_1024`: `0.1438 s`
- Model size (`e8_352` checkpoint): `579M` total (`model.safetensors` ~`576M`).

## 6. Visual Examples
Side-by-side panels (`Original | GT | Prediction`) are archived in:
- `results/finetuned/clipseg_ft_e8_352_v1/report_panels_clipseg_ft_e8_352_v1/`

Examples:
- `drywall-join-detect__2000x1500_20_resized_jpg.rf.be61fbb154dd0954665ede3fa96236fe__panel.png`
- `drywall-join-detect__IMG_8206_JPG_jpg.rf.cd21944e2cf09139a3c040c48502e944__panel.png`
- `drywall-join-detect__IMG_8224_JPG_jpg.rf.9a400b4c54301ff9d5548148292723fb__panel.png`
- `drywall-join-detect__IMG_8236_JPG_jpg.rf.c257d1d85928d6e72efe38ab4d3a67e6__panel.png`

## 7. Failure Cases and Potential Solutions
1. Thin seam masks are often predicted thicker than ground truth.
   - Likely causes: spatial downsampling and boundary blur.
   - Potential fixes: higher-resolution training/inference, tiling, boundary-aware loss, lightweight edge refinement.

2. Over-segmentation in scenes with multiple seam-like lines/textures.
   - Likely causes: visually similar background patterns and global thresholding.
   - Potential fixes: validation-set threshold tuning, connected-component filtering, hard-negative mining.

3. Extent mismatch (right location, wrong length/width).
   - Likely causes: annotation style variance and perspective/illumination changes.
   - Potential fixes: targeted augmentation (lighting/perspective), consistency checks on mask thickness, more diverse labeled samples.

## 8. Final Model Choice and Tradeoff
- Recommended accuracy-first model: `clipseg_ft_e16_512_thr04_v1` (best mIoU/Dice).
- Recommended speed-first model: `clipseg_ft_e8_352_v1` or `clipseg_ft_e12_352_trial1` (much faster inference, still large gain over baseline).

## 9. Reproducibility
- Repository: `https://github.com/manojkumartjpk/origin_finetune`
- Colab notebook (with outputs): `https://colab.research.google.com/drive/1543t9QcbBRw2QnUsZIEXMZlZcmhcm4f-?usp=sharing`
- Key notebook files:
  - `colab_origin_takehome.ipynb`
  - `colab_origin_takehome_with_output.ipynb`
- Main artifacts in repo:
  - `results/baselines/*`
  - `results/finetuned/*`
  - `results/experiments/*`

## 10. Next Steps
1. Experiment full model training:
   - Run a full fine-tuning sweep on larger variants and/or longer schedules with strict validation-based checkpointing.

2. Post-processing to match box binary mask:
   - Add morphology + connected-components cleanup to align predictions with expected compact binary mask shapes.

3. Validation-driven threshold calibration:
   - Select threshold on validation split (not test), then lock it before final test evaluation.

4. Data quality and hard-negative expansion:
   - Add more seam-like non-target backgrounds and tighten annotation consistency for thin boundaries.

5. Productionization path:
   - Export reproducible inference pipeline with latency/throughput benchmarks and confidence monitoring hooks.
