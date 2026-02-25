from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm


CANONICAL_MAP = {
    "drywall-join-detect": "taping_area",
    "drywall-join": "taping_area",
    "drywall join": "taping_area",
    "taping_area": "taping_area",
    "drywall seam": "taping_area",
    "seam-detect": "taping_area",
    "seam": "taping_area",
    "join": "taping_area",
    "joint": "taping_area",
    "tape": "taping_area",
    "cracks": "crack",
    "crack": "crack",
    "wall crack": "crack",
}


def polygon_or_rle_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and "counts" in segmentation:
        rle = segmentation
    else:
        raise ValueError("Unsupported segmentation format")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    return mask.astype(np.uint8)


def canonicalize_category(name: str) -> str:
    key = name.strip().lower()
    for k, v in CANONICAL_MAP.items():
        if k in key:
            return v
    raise ValueError(f"Could not map category '{name}' to canonical label")


def split_ids(image_ids: list[int], train_ratio: float, val_ratio: float, seed: int):
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = set(ids[:n_train])
    val = set(ids[n_train:n_train + n_val])
    test = set(ids[n_train + n_val :])
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO segmentation export to unified binary masks + manifest.")
    parser.add_argument("--coco-json", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--dataset-tag", required=True, help="Human-readable tag, e.g. cracks or drywall-join-detect")
    parser.add_argument("--out-dir", required=True, help="Output root, e.g. data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco = COCO(args.coco_json)
    cat_lookup = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    image_ids = coco.getImgIds()
    train_ids, val_ids, test_ids = split_ids(image_ids, args.train_ratio, args.val_ratio, args.seed)

    out_root = Path(args.out_dir)
    img_out_dir = out_root / "images"
    mask_out_dir = out_root / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    split_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for img_id in tqdm(image_ids, desc=f"Converting {args.dataset_tag}"):
        img = coco.loadImgs([img_id])[0]
        file_name = img["file_name"]
        h, w = img["height"], img["width"]

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        # Group masks by canonical label so one image can produce multiple prompt-target rows.
        masks_by_label = defaultdict(lambda: np.zeros((h, w), dtype=np.uint8))
        for ann in anns:
            cat_name = cat_lookup[ann["category_id"]]
            label = canonicalize_category(cat_name)
            if "segmentation" in ann and ann["segmentation"]:
                ann_mask = polygon_or_rle_to_mask(ann["segmentation"], h, w)
            elif "bbox" in ann:
                x, y, bw, bh = map(int, ann["bbox"])
                ann_mask = np.zeros((h, w), dtype=np.uint8)
                ann_mask[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)] = 1
            else:
                continue
            masks_by_label[label] = np.maximum(masks_by_label[label], ann_mask)

        src_img_path = Path(args.images_dir) / file_name
        if not src_img_path.exists():
            # Some exports keep images adjacent to JSON; fall back.
            alt = Path(args.coco_json).parent / file_name
            if alt.exists():
                src_img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {file_name}")

        split = "train" if img_id in train_ids else "val" if img_id in val_ids else "test"

        for label, mask_np in masks_by_label.items():
            if mask_np.sum() == 0:
                continue
            stem = Path(file_name).stem
            image_copy_name = f"{args.dataset_tag}__{stem}{Path(file_name).suffix}"
            mask_name = f"{args.dataset_tag}__{stem}__{label}.png"
            dst_img_path = img_out_dir / image_copy_name
            dst_mask_path = mask_out_dir / mask_name

            if not dst_img_path.exists():
                Image.open(src_img_path).convert("RGB").save(dst_img_path)
            Image.fromarray((mask_np > 0).astype(np.uint8) * 255, mode="L").save(dst_mask_path)

            rows.append(
                {
                    "image_id": f"{args.dataset_tag}__{stem}",
                    "dataset_tag": args.dataset_tag,
                    "label": label,
                    "split": split,
                    "image_path": str(dst_img_path.resolve()),
                    "mask_path": str(dst_mask_path.resolve()),
                }
            )
            split_counts[split] += 1
            label_counts[label] += 1

    if not rows:
        raise RuntimeError("No rows generated. Check export format and category names.")

    manifest_path = out_root / f"manifest_{args.dataset_tag}.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    summary = {
        "dataset_tag": args.dataset_tag,
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "manifest_path": str(manifest_path.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
