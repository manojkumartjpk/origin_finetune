from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from src.data.prepare_from_coco import canonicalize_category, polygon_or_rle_to_mask


SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def convert_split(export_root: Path, split_name: str, dataset_tag: str, out_root: Path):
    split_dir = export_root / split_name
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return []

    coco = COCO(str(ann_path))
    cat_lookup = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    img_out_dir = out_root / "images"
    mask_out_dir = out_root / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_id in tqdm(coco.getImgIds(), desc=f"{dataset_tag}:{split_name}"):
        img = coco.loadImgs([img_id])[0]
        h, w = img["height"], img["width"]
        file_name = img["file_name"]
        src_img_path = split_dir / file_name
        if not src_img_path.exists():
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        masks_by_label = defaultdict(lambda: np.zeros((h, w), dtype=np.uint8))
        for ann in anns:
            cat_name = cat_lookup[ann["category_id"]]
            label = canonicalize_category(cat_name)
            seg = ann.get("segmentation")
            if seg:
                ann_mask = polygon_or_rle_to_mask(seg, h, w)
            elif "bbox" in ann:
                x, y, bw, bh = map(int, ann["bbox"])
                ann_mask = np.zeros((h, w), dtype=np.uint8)
                ann_mask[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)] = 1
            else:
                continue
            masks_by_label[label] = np.maximum(masks_by_label[label], ann_mask)

        stem = Path(file_name).stem
        canonical_split = SPLIT_MAP[split_name]
        for label, mask_np in masks_by_label.items():
            if mask_np.sum() == 0:
                continue
            image_copy_name = f"{dataset_tag}__{stem}{Path(file_name).suffix}"
            mask_name = f"{dataset_tag}__{stem}__{label}.png"
            dst_img_path = img_out_dir / image_copy_name
            dst_mask_path = mask_out_dir / mask_name
            if not dst_img_path.exists():
                Image.open(src_img_path).convert("RGB").save(dst_img_path)
            Image.fromarray((mask_np > 0).astype(np.uint8) * 255, mode="L").save(dst_mask_path)
            rows.append(
                {
                    "image_id": f"{dataset_tag}__{stem}",
                    "dataset_tag": dataset_tag,
                    "label": label,
                    "split": canonical_split,
                    "image_path": str(dst_img_path.resolve()),
                    "mask_path": str(dst_mask_path.resolve()),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO export (train/valid/test dirs) to unified binary mask manifest.")
    parser.add_argument("--export-root", required=True, help="Path to Roboflow COCO export root")
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    export_root = Path(args.export_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split_name in ["train", "valid", "val", "test"]:
        if (export_root / split_name).exists():
            all_rows.extend(convert_split(export_root, split_name, args.dataset_tag, out_root))

    if not all_rows:
        raise RuntimeError(
            f"No rows generated from {export_root}. Expected train/valid/test folders with _annotations.coco.json."
        )

    manifest_path = out_root / f"manifest_{args.dataset_tag}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(manifest_path, index=False)

    print("Saved manifest:", manifest_path)
    print(df.groupby(["split", "label"]).size().to_string())

    summary_path = out_root / f"summary_{args.dataset_tag}.json"
    summary = {
        "dataset_tag": args.dataset_tag,
        "num_rows": len(df),
        "split_label_counts": {
            f"{split}:{label}": int(count)
            for (split, label), count in df.groupby(["split", "label"]).size().items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()

