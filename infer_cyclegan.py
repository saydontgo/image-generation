from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from imggen import CycleGANModel, collect_image_paths, save_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CycleGAN generators on a photo folder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a generator checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="Input image file or folder.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for generated outputs.")
    parser.add_argument(
        "--direction",
        type=str,
        default="A2B",
        choices=["A2B", "B2A"],
        help="A2B means photo->art if train-a was photo and train-b was artwork.",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Resize shorter side before center crop.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--generator-channels", type=int, default=64, help="Must match training config.")
    parser.add_argument("--discriminator-channels", type=int, default=64, help="Must match training config.")
    parser.add_argument("--res-blocks", type=int, default=9, help="Must match training config.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, Image.Resampling.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_generator(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = CycleGANModel(
        generator_channels=args.generator_channels,
        discriminator_channels=args.discriminator_channels,
        res_blocks=args.res_blocks,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Compatible with two common formats:
    # 1) our bundled checkpoint containing both generators
    # 2) official CycleGAN-style single-generator state dict such as latest_net_G_A.pth
    if isinstance(checkpoint, dict) and any(key in checkpoint for key in ("netG_A", "G_A", "netG_B", "G_B")):
        model.load_generators_only(checkpoint)
        generator = model.netG_A if args.direction == "A2B" else model.netG_B
    elif isinstance(checkpoint, dict):
        generator = model.netG_A if args.direction == "A2B" else model.netG_B
        generator.load_state_dict(checkpoint)
    else:
        raise TypeError("Unsupported checkpoint format.")
    generator.eval()
    return generator


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    input_paths = collect_image_paths(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(args.image_size)
    generator = load_generator(args, device)

    for image_path in tqdm(input_paths, desc="infer", ncols=100):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            generated = generator(tensor)
        save_image_tensor(generated, output_dir / image_path.name)

    print(f"Finished inference for {len(input_paths)} images. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
