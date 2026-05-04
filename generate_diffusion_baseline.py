from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from imggen import collect_image_paths, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diffusion baseline images from prompts.")
    parser.add_argument("--input", type=str, required=True, help="Input image file or folder.")
    parser.add_argument("--styles-file", type=str, required=True, help="JSON file defining style prompts.")
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_baseline", help="Output directory.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusers model id, or use a local path if you downloaded it already.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--image-size", type=int, default=768, help="Resize long side before img2img generation.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    width, height = image.size
    current = max(width, height)
    if current == long_side:
        return image
    scale = long_side / current
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)


def load_styles(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not data:
        raise ValueError("styles file must be a non-empty JSON object.")
    return data


def load_pipeline(model_id: str, device: torch.device):
    from diffusers import AutoPipelineForImage2Image

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device.type == "cuda":
        pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    paths = collect_image_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    styles = load_styles(args.styles_file)
    pipeline = load_pipeline(args.model_id, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "model_id": args.model_id,
            "input": args.input,
            "styles_file": args.styles_file,
            "seed": args.seed,
            "image_size": args.image_size,
            "styles": styles,
        },
        output_dir / "run_config.json",
    )

    for image_index, image_path in enumerate(paths):
        base_image = resize_long_side(Image.open(image_path).convert("RGB"), args.image_size)
        for style_index, (style_name, style_config) in enumerate(styles.items()):
            generator = torch.Generator(device="cpu").manual_seed(args.seed + image_index * 1000 + style_index)
            result = pipeline(
                prompt=style_config["prompt"],
                negative_prompt=style_config.get("negative_prompt"),
                image=base_image,
                strength=style_config.get("strength", 0.7),
                guidance_scale=style_config.get("guidance_scale", 7.5),
                num_inference_steps=style_config.get("num_inference_steps", 30),
                generator=generator,
            ).images[0]
            style_dir = output_dir / style_name
            style_dir.mkdir(parents=True, exist_ok=True)
            result.save(style_dir / image_path.name)
            print(f"[{style_name}] saved {style_dir / image_path.name}")


if __name__ == "__main__":
    main()
