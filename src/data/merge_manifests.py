from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge per-dataset manifests into one unified manifest.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Manifest CSV paths")
    parser.add_argument("--out", required=True, help="Output merged manifest CSV")
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    merged = pd.concat(dfs, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print("Merged rows:", len(merged))
    print(merged.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()

