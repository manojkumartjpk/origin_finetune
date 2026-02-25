from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    image: Image.Image
    mask: Image.Image
    prompt: str
    image_id: str
    label: str
    image_path: str
    mask_path: str


class PromptSegmentationDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, prompts_json: str | Path, split: str, deterministic_eval_prompt: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}' in {manifest_csv}")

        with open(prompts_json, "r", encoding="utf-8") as f:
            prompt_cfg = json.load(f)
        self.prompts = {k: v for k, v in prompt_cfg.items() if k != "eval_prompts"}
        self.eval_prompts = prompt_cfg.get("eval_prompts", {})
        self.deterministic_eval_prompt = deterministic_eval_prompt

    def __len__(self) -> int:
        return len(self.df)

    def _pick_prompt(self, label: str) -> str:
        if self.deterministic_eval_prompt and label in self.eval_prompts:
            return self.eval_prompts[label]
        choices = self.prompts.get(label)
        if not choices:
            raise KeyError(f"No prompt mapping found for label '{label}'")
        return random.choice(choices)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        prompt = self._pick_prompt(row["label"])
        return Sample(
            image=image,
            mask=mask,
            prompt=prompt,
            image_id=str(row["image_id"]),
            label=str(row["label"]),
            image_path=str(row["image_path"]),
            mask_path=str(row["mask_path"]),
        )


def build_collate_fn(processor, image_size: int):
    def collate(samples: list[Sample]):
        images = [s.image.resize((image_size, image_size), resample=Image.BILINEAR) for s in samples]
        prompts = [s.prompt for s in samples]
        enc = processor(
            text=prompts,
            images=images,
            padding="max_length",
            return_tensors="pt",
        )

        masks = []
        meta = []
        for s in samples:
            resized_mask = s.mask.resize((image_size, image_size), resample=Image.NEAREST)
            mask_np = (np.array(resized_mask, dtype=np.uint8) > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np)
            masks.append(mask_tensor.unsqueeze(0))
            meta.append(
                {
                    "image_id": s.image_id,
                    "label": s.label,
                    "prompt": s.prompt,
                    "orig_size": s.image.size[::-1],  # (H, W)
                    "image_path": s.image_path,
                    "mask_path": s.mask_path,
                }
            )
        batch = {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "masks": torch.stack(masks, dim=0),  # B,1,H,W
            "meta": meta,
        }
        return batch

    return collate
