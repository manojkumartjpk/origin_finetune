# Prompted Segmentation for Drywall QA - Report

## 1. Goal Summary
- Build a text-conditioned segmentation model that predicts binary masks for:
- `segment crack`
- `segment taping area`

## 2. Methodology
- Model(s) tried:
- Final model:
- Why this model:
- Training objective:

## 3. Data Preparation
- Dataset 1 (Drywall-Join-Detect) version/export format:
- Dataset 2 (Cracks) version/export format:
- Annotation type (polygon mask / bbox / other):
- Canonical labels:
- Prompt mapping used for training:
- Image preprocessing / resize:
- Split strategy (seed noted):

### Split Counts
| Split | Crack | Taping Area | Total |
|---|---:|---:|---:|
| Train |  |  |  |
| Val |  |  |  |
| Test |  |  |  |

## 4. Training Setup
- Hardware: Google Colab T4
- Epochs:
- Batch size:
- LR:
- Image size:
- Mixed precision:
- Seed:
- Train time:
- Best checkpoint criterion:

## 5. Results

### Metrics (Test)
| Prompt / Label | mIoU | Dice | Notes |
|---|---:|---:|---|
| `segment crack` |  |  |  |
| `segment taping area` |  |  |  |
| Overall |  |  |  |

### Runtime & Footprint
- Avg inference time / image:
- Model size on disk:
- Params (optional):

## 6. Visual Examples
Include 3-4 examples (`original | GT | prediction`) with diverse scenes.

## 7. Failure Cases and Potential Solutions
- Failure case 1:
- Potential fix:
- Failure case 2:
- Potential fix:
- Failure case 3:
- Potential fix:

## 8. Reproducibility
- GitHub/Colab link:
- Exact commands:
- Seeds:

