from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample one split to equal counts across groups (e.g. dataset_tag/label), keeping other splits unchanged."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--group-col", default="dataset_tag", help="Column used to balance counts (e.g. dataset_tag or label)")
    parser.add_argument("--unit-col", default="image_id", help="Sample units by this column to avoid splitting same image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Optional per-group target. Defaults to min count among groups in the selected split.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    required = {"split", args.group_col, args.unit_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    split_df = df[df["split"] == args.split].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split='{args.split}'")

    unit_groups = (
        split_df[[args.unit_col, args.group_col]]
        .drop_duplicates()
        .groupby(args.group_col)[args.unit_col]
        .apply(list)
        .to_dict()
    )
    if len(unit_groups) < 2:
        raise ValueError(f"Need at least 2 groups in column '{args.group_col}' to balance")

    counts = {g: len(ids) for g, ids in unit_groups.items()}
    target = args.target_count if args.target_count is not None else min(counts.values())
    if target <= 0:
        raise ValueError("Computed target_count <= 0")

    keep_units: set[str] = set()
    for group_name, ids in unit_groups.items():
        s = pd.Series(sorted(set(ids)))
        if len(s) < target:
            raise ValueError(
                f"Group '{group_name}' has only {len(s)} units; cannot sample target_count={target}. "
                "Use a smaller target or default min-count behavior."
            )
        sampled = s.sample(n=target, random_state=args.seed, replace=False).tolist()
        keep_units.update(map(str, sampled))

    keep_mask = (df["split"] != args.split) | (df[args.unit_col].astype(str).isin(keep_units))
    out_df = df[keep_mask].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved balanced manifest to {out_path}")
    print(f"Balanced split: {args.split}")
    print(f"group_col: {args.group_col}")
    print(f"unit_col: {args.unit_col}")
    print(f"target_count_per_group: {target}")
    print("\nBefore (unique units):")
    before = (
        split_df[[args.unit_col, args.group_col]]
        .drop_duplicates()
        .groupby(args.group_col)[args.unit_col]
        .nunique()
        .sort_index()
    )
    print(before.to_string())
    print("\nAfter (unique units):")
    after_split = out_df[out_df["split"] == args.split]
    after = (
        after_split[[args.unit_col, args.group_col]]
        .drop_duplicates()
        .groupby(args.group_col)[args.unit_col]
        .nunique()
        .sort_index()
    )
    print(after.to_string())

    for cols in (["split", "label"], ["split", "dataset_tag"]):
        if all(c in out_df.columns for c in cols):
            print(f"\nRow counts by {cols}:")
            print(out_df.groupby(cols).size().to_string())


if __name__ == "__main__":
    main()
