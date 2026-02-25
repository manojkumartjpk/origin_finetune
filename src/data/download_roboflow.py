from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from roboflow import Roboflow


API_URL = "https://api.roboflow.com"


def fetch_project_meta(api_key: str, workspace: str, project: str) -> dict[str, Any]:
    url = f"{API_URL}/{workspace}/{project}?api_key={api_key}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def choose_version(meta: dict[str, Any], explicit_version: int | None) -> int:
    versions = meta.get("versions", [])
    if explicit_version is not None:
        return explicit_version
    if not versions:
        raise RuntimeError(
            "Project has no published versions available for SDK download. "
            "Ask for a manual Roboflow export ZIP (COCO preferred) or a versioned project link."
        )
    latest = max(int(v["id"].rstrip("/").split("/")[-1]) for v in versions)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset version using the Python SDK.")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--format", default="coco", help="Roboflow export format (default: coco)")
    parser.add_argument("--version", type=int, default=None, help="Specific version to download (default: latest published)")
    parser.add_argument("--out-dir", required=True, help="Destination directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta = fetch_project_meta(args.api_key, args.workspace, args.project)
    if "project" not in meta:
        raise RuntimeError(
            "Roboflow API response did not contain 'project'. "
            f"Check workspace/project names and API key. Response keys={list(meta.keys())}, body={json.dumps(meta)[:500]}"
        )
    project_meta = meta["project"]
    versions = meta.get("versions", [])

    summary = {
        "id": project_meta.get("id"),
        "name": project_meta.get("name"),
        "type": project_meta.get("type"),
        "published_versions": len(versions),
        "splits": project_meta.get("splits"),
        "classes": project_meta.get("classes"),
    }
    print("Project metadata:")
    print(json.dumps(summary, indent=2))

    version_num = choose_version(meta, args.version)
    print(f"Using version: {version_num}")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(version_num)

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds = version.download(args.format, location=str(out_dir), overwrite=args.overwrite)
    print("Downloaded to:", ds.location)


if __name__ == "__main__":
    main()
