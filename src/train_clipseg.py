from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from src.data.dataset import PromptSegmentationDataset, build_collate_fn
from src.utils.metrics import batch_iou_dice_from_logits, dice_loss_from_logits
from src.utils.seed import set_seed


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.7) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["masks"].to(device)

        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.unsqueeze(1) if outputs.logits.ndim == 3 else outputs.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = bce_dice_loss(logits, masks)
        iou, dice = batch_iou_dice_from_logits(logits, masks, threshold=threshold)
        n = masks.shape[0]
        total_loss += loss.item() * n
        total_iou += iou.mean().item() * n
        total_dice += dice.mean().item() * n
        total_n += n

    return {
        "loss": total_loss / max(total_n, 1),
        "miou": total_iou / max(total_n, 1),
        "dice": total_dice / max(total_n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--prompts-json", default="configs/prompts.json")
    parser.add_argument("--model-name", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--output-dir", default="checkpoints/clipseg_takehome")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--no-processor-resize", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_memory = (device.type == "cuda") and (not args.no_pin_memory)
    if args.pin_memory:
        pin_memory = True

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    processor = CLIPSegProcessor.from_pretrained(args.model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_name).to(device)

    train_ds = PromptSegmentationDataset(args.manifest_csv, args.prompts_json, split="train", deterministic_eval_prompt=False)
    val_ds = PromptSegmentationDataset(args.manifest_csv, args.prompts_json, split="val", deterministic_eval_prompt=True)
    collate_fn = build_collate_fn(processor, args.image_size, processor_resize=not args.no_processor_resize)

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dataloader_kwargs)

    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []

    start_train = time.time()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=pin_memory)
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
            masks = batch["masks"].to(device, non_blocking=pin_memory)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.unsqueeze(1) if outputs.logits.ndim == 3 else outputs.logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = bce_dice_loss(logits, masks) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_n = masks.shape[0]
            seen += batch_n
            running_loss += loss.item() * args.grad_accum_steps * batch_n
            pbar.set_postfix(loss=f"{running_loss/max(seen,1):.4f}")

        if len(train_loader) > 0 and (len(train_loader) % args.grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)
        train_loss = running_loss / max(seen, 1)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        score = (val_metrics["miou"] + val_metrics["dice"]) / 2.0
        if score > best_score:
            best_score = score
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(epoch_record, f, indent=2)

    total_train_seconds = time.time() - start_train
    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "history": history,
                "train_time_seconds": total_train_seconds,
                "best_score": best_score,
            },
            f,
            indent=2,
        )
    print(f"Training complete. Best composite score: {best_score:.4f}")
    print(f"Train time (s): {total_train_seconds:.1f}")


if __name__ == "__main__":
    main()
