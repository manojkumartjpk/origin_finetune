from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def _copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Archive small experiment artifacts into the tracked results/ tree.")
    parser.add_argument("--run-id", required=True, help="Unique run ID, e.g. clipseg_zeroshot_v1")
    parser.add_argument("--category", required=True, help="baselines | finetuned | experiments")
    parser.add_argument("--summary-json", default=None, help="Path to a metrics/summary JSON file to copy as metrics.json")
    parser.add_argument("--notes", default="", help="Short note saved to notes.txt")
    parser.add_argument("--copy", nargs="*", default=[], help="Extra files/dirs to copy into the run directory")
    parser.add_argument("--out-root", default="results", help="Tracked results root")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / args.category / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": args.run_id,
        "category": args.category,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_json": None,
        "copied": [],
    }

    if args.summary_json:
        src = Path(args.summary_json)
        if src.exists():
            dst = out_dir / "metrics.json"
            _copy_path(src, dst)
            manifest["summary_json"] = str(dst)

    for path_str in args.copy:
        src = Path(path_str)
        if not src.exists():
            continue
        dst = out_dir / src.name
        _copy_path(src, dst)
        manifest["copied"].append(str(dst))

    (out_dir / "notes.txt").write_text(args.notes.strip() + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Archived run to {out_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

