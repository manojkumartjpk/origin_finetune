from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create side-by-side report panels (Original | GT | Pred).")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--pred-dir", required=True, help="Directory containing <image_id>__pred.png and __gt.png")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--group-col", default="dataset_tag", help="Group key to cap examples per group")
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--max-total", type=int, default=None, help="Optional hard cap across all groups")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    if args.split:
        manifest = manifest[manifest["split"] == args.split].copy()
    if manifest.empty:
        raise ValueError("No rows found after filtering manifest")
    if args.group_col not in manifest.columns:
        raise ValueError(f"Manifest missing group column: {args.group_col}")

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_per_group: dict[str, int] = {}
    saved_total = 0

    # Deterministic order for reproducible panels.
    sort_cols = [c for c in [args.group_col, "label", "image_id"] if c in manifest.columns]
    manifest = manifest.sort_values(sort_cols).reset_index(drop=True)

    for _, row in manifest.iterrows():
        group = str(row[args.group_col])
        if saved_per_group.get(group, 0) >= args.per_group:
            continue
        if args.max_total is not None and saved_total >= args.max_total:
            break

        image_id = str(row["image_id"])
        pred_path = pred_dir / f"{image_id}__pred.png"
        gt_path = pred_dir / f"{image_id}__gt.png"
        if not pred_path.exists() or not gt_path.exists():
            continue

        img = Image.open(row["image_path"]).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        pred = Image.open(pred_path).convert("L")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("GT")
        axes[1].axis("off")
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Pred")
        axes[2].axis("off")
        title_parts = [image_id]
        if "dataset_tag" in row:
            title_parts.append(str(row["dataset_tag"]))
        if "label" in row:
            title_parts.append(str(row["label"]))
        fig.suptitle(" | ".join(title_parts))

        out_path = out_dir / f"{image_id}__panel.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved_per_group[group] = saved_per_group.get(group, 0) + 1
        saved_total += 1
        print(f"saved {out_path}")

    print("Saved panels total:", saved_total)
    print("Saved per group:", saved_per_group)


if __name__ == "__main__":
    main()
