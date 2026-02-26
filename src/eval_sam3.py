from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.metrics import batch_iou_dice_from_logits


def _load_prompt_map(prompts_json: str) -> dict[str, str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("eval_prompts", {})


def _load_mask(mask_path: str) -> np.ndarray:
    return (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def _union_instance_masks(results: dict, height: int, width: int) -> np.ndarray:
    masks = results.get("masks")
    if masks is None:
        return np.zeros((height, width), dtype=np.uint8)
    if torch.is_tensor(masks):
        if masks.numel() == 0:
            return np.zeros((height, width), dtype=np.uint8)
        m = masks.detach().cpu().numpy()
    else:
        m = np.asarray(masks)
    if m.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if m.ndim == 2:
        return (m > 0).astype(np.uint8)
    return (np.any(m > 0, axis=0)).astype(np.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--prompts-json", default="configs/prompts.json")
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Instance confidence threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--save-vis-dir", default=None)
    parser.add_argument("--max-vis", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick subset eval")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()

    from transformers import Sam3Model, Sam3Processor  # imported lazily for environments without SAM3-capable transformers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = os.environ.get(args.hf_token_env)
    model = Sam3Model.from_pretrained(args.model_id, token=hf_token).to(device)
    processor = Sam3Processor.from_pretrained(args.model_id, token=hf_token)
    model.eval()

    df = pd.read_csv(args.manifest_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_samples:
        df = df.iloc[: args.max_samples].copy()
    if df.empty:
        raise ValueError(f"No rows found for split='{args.split}' in {args.manifest_csv}")

    eval_prompts = _load_prompt_map(args.prompts_json)
    per_label = defaultdict(lambda: {"iou_sum": 0.0, "dice_sum": 0.0, "n": 0})
    overall = {"iou_sum": 0.0, "dice_sum": 0.0, "n": 0}
    infer_times = []
    vis_saved = 0
    vis_dir = Path(args.save_vis_dir) if args.save_vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    iterable = tqdm(df.itertuples(index=False), total=len(df), desc=f"SAM3 Eval {args.split}")
    for row in iterable:
        image = Image.open(row.image_path).convert("RGB")
        gt = _load_mask(row.mask_path)
        prompt = eval_prompts.get(row.label, f"segment {row.label.replace('_', ' ')}")

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        t0 = time.perf_counter()
        outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t0)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").detach().cpu().tolist(),
        )[0]

        pred = _union_instance_masks(results, gt.shape[0], gt.shape[1])

        pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        logits_proxy = torch.where(pred_t > 0, torch.tensor(10.0), torch.tensor(-10.0))
        iou, dice = batch_iou_dice_from_logits(logits_proxy, gt_t, threshold=0.5)
        iou_val = float(iou[0].item())
        dice_val = float(dice[0].item())

        per_label[row.label]["iou_sum"] += iou_val
        per_label[row.label]["dice_sum"] += dice_val
        per_label[row.label]["n"] += 1
        overall["iou_sum"] += iou_val
        overall["dice_sum"] += dice_val
        overall["n"] += 1

        if vis_dir and vis_saved < args.max_vis:
            Image.fromarray((pred * 255).astype("uint8"), mode="L").save(vis_dir / f"{row.image_id}__pred.png")
            Image.fromarray((gt * 255).astype("uint8"), mode="L").save(vis_dir / f"{row.image_id}__gt.png")
            vis_saved += 1

    per_label_metrics = {
        label: {
            "miou": agg["iou_sum"] / max(agg["n"], 1),
            "dice": agg["dice_sum"] / max(agg["n"], 1),
            "n": agg["n"],
        }
        for label, agg in per_label.items()
    }
    result = {
        "split": args.split,
        "overall": {
            "miou": overall["iou_sum"] / max(overall["n"], 1),
            "dice": overall["dice_sum"] / max(overall["n"], 1),
            "n": overall["n"],
        },
        "per_label": per_label_metrics,
        "avg_inference_time_sec_per_image": sum(infer_times) / max(len(infer_times), 1),
        "model_dir": args.model_id,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "max_samples": args.max_samples,
    }
    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

