from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _split_ids(ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * train_ratio)) if n > 0 else 0
    n_val = int(n * val_ratio) if n > 2 else max(0, n - n_train - 1)
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    if not test and ids:
        # Guarantee a test sample when possible for reporting/metrics.
        moved = next(iter(train or val))
        if moved in train:
            train.remove(moved)
        if moved in val:
            val.remove(moved)
        test.add(moved)
    if not val and len(ids) >= 3:
        moved = next(iter(train))
        train.remove(moved)
        val.add(moved)
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Resplit a merged manifest by image_id within each label.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    required_cols = {"image_id", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    out_df = df.copy()
    assignments: dict[str, str] = {}
    summary = {}

    for label, label_df in df.groupby("label"):
        image_ids = sorted(label_df["image_id"].unique().tolist())
        train_ids, val_ids, test_ids = _split_ids(image_ids, args.train_ratio, args.val_ratio, args.seed)
        summary[label] = {
            "num_images": len(image_ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        }
        for image_id in train_ids:
            assignments[image_id] = "train"
        for image_id in val_ids:
            assignments[image_id] = "val"
        for image_id in test_ids:
            assignments[image_id] = "test"

    out_df["split"] = out_df["image_id"].map(assignments)
    if out_df["split"].isna().any():
        missing_ids = out_df.loc[out_df["split"].isna(), "image_id"].unique().tolist()[:10]
        raise RuntimeError(f"Missing split assignments for some rows, examples: {missing_ids}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    split_label_counts = {
        f"{split}:{label}": int(count)
        for (split, label), count in out_df.groupby(["split", "label"]).size().items()
    }
    print(json.dumps({"summary_by_label_images": summary, "split_label_row_counts": split_label_counts}, indent=2, default=int))


if __name__ == "__main__":
    main()
