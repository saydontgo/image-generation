from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_image_paths(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    return sorted(path for path in root_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    tensor = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = ((tensor + 1.0) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(array)


def save_image_tensor(image_tensor: torch.Tensor, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image_tensor).save(output)


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_yaml_like_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
