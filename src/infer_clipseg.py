from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


def slugify_prompt(prompt: str) -> str:
    return prompt.strip().lower().replace("/", "_").replace(" ", "_")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", default="outputs/pred_masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--image-id", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained(args.model_dir)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_dir).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = image.size
    inputs = processor(text=[args.prompt], images=[image], return_tensors="pt", padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits.unsqueeze(1) if outputs.logits.ndim == 3 else outputs.logits
    logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred = (torch.sigmoid(logits)[0, 0] >= args.threshold).detach().cpu().numpy().astype(np.uint8) * 255

    image_id = args.image_id or Path(args.image).stem
    out_name = f"{image_id}__{slugify_prompt(args.prompt)}.png"
    out_path = out_dir / out_name
    Image.fromarray(pred, mode="L").save(out_path)
    print(out_path.resolve())


if __name__ == "__main__":
    main()

