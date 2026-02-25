from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from src.data.dataset import PromptSegmentationDataset, build_collate_fn
from src.utils.metrics import batch_iou_dice_from_logits


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--prompts-json", default="configs/prompts.json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-vis-dir", default=None)
    parser.add_argument("--max-vis", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained(args.model_dir)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_dir).to(device)
    model.eval()

    ds = PromptSegmentationDataset(args.manifest_csv, args.prompts_json, split=args.split, deterministic_eval_prompt=True)
    collate_fn = build_collate_fn(processor, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    per_label = defaultdict(lambda: {"iou_sum": 0.0, "dice_sum": 0.0, "n": 0})
    overall = {"iou_sum": 0.0, "dice_sum": 0.0, "n": 0}
    vis_saved = 0
    vis_dir = Path(args.save_vis_dir) if args.save_vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    infer_times = []
    for batch in tqdm(loader, desc=f"Eval {args.split}"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["masks"].to(device)

        t0 = time.perf_counter()
        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append((time.perf_counter() - t0) / pixel_values.shape[0])

        logits = outputs.logits.unsqueeze(1) if outputs.logits.ndim == 3 else outputs.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        ious, dices = batch_iou_dice_from_logits(logits, masks, threshold=args.threshold)
        probs = torch.sigmoid(logits)
        preds = (probs >= args.threshold).float()

        for i in range(masks.shape[0]):
            label = batch["meta"][i]["label"]
            iou_val = float(ious[i].item())
            dice_val = float(dices[i].item())
            per_label[label]["iou_sum"] += iou_val
            per_label[label]["dice_sum"] += dice_val
            per_label[label]["n"] += 1
            overall["iou_sum"] += iou_val
            overall["dice_sum"] += dice_val
            overall["n"] += 1

            if vis_dir and vis_saved < args.max_vis:
                pred_img = (preds[i, 0].detach().cpu().numpy() * 255).astype("uint8")
                gt_img = (masks[i, 0].detach().cpu().numpy() * 255).astype("uint8")
                Image.fromarray(pred_img).save(vis_dir / f"{batch['meta'][i]['image_id']}__pred.png")
                Image.fromarray(gt_img).save(vis_dir / f"{batch['meta'][i]['image_id']}__gt.png")
                vis_saved += 1

    per_label_metrics = {}
    for label, agg in per_label.items():
        per_label_metrics[label] = {
            "miou": agg["iou_sum"] / max(agg["n"], 1),
            "dice": agg["dice_sum"] / max(agg["n"], 1),
            "n": agg["n"],
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
        "model_dir": str(Path(args.model_dir).resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

