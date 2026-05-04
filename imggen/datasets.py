from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import collect_image_paths


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        root_a: str | Path,
        root_b: str | Path,
        image_size: int = 256,
        random_flip: bool = True,
    ) -> None:
        self.paths_a = collect_image_paths(root_a)
        self.paths_b = collect_image_paths(root_b)
        if not self.paths_a:
            raise FileNotFoundError(f"No images found in domain A: {root_a}")
        if not self.paths_b:
            raise FileNotFoundError(f"No images found in domain B: {root_b}")

        transform_steps: list[transforms.Transform] = [
            transforms.Resize(int(image_size * 1.12), Image.Resampling.BICUBIC),
            transforms.RandomCrop(image_size),
        ]
        if random_flip:
            transform_steps.append(transforms.RandomHorizontalFlip())
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, index: int) -> dict[str, object]:
        path_a = self.paths_a[index % len(self.paths_a)]
        path_b = self.paths_b[random.randrange(len(self.paths_b))]
        image_a = self.transform(Image.open(path_a).convert("RGB"))
        image_b = self.transform(Image.open(path_b).convert("RGB"))
        return {
            "A": image_a,
            "B": image_b,
            "A_path": str(path_a),
            "B_path": str(path_b),
        }
