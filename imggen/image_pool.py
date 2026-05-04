from __future__ import annotations

import random

import torch


class ImagePool:
    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 0:
            return images

        returned: list[torch.Tensor] = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                returned.append(image)
                continue

            if random.random() > 0.5:
                index = random.randrange(len(self.images))
                cached = self.images[index].clone()
                self.images[index] = image
                returned.append(cached)
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)
