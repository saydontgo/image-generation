from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm

from imggen import build_official_generator_from_state_dict, collect_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CycleGAN inference and optionally export side-by-side comparison sheets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a generator checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="Input image file or folder.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for generated outputs.")
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="",
        help="Optional public/pretrained generator checkpoint used for side-by-side comparison.",
    )
    parser.add_argument(
        "--baseline-output-dir",
        type=str,
        default="",
        help="Optional directory to save outputs from the public/pretrained model.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="A2B",
        choices=["A2B", "B2A"],
        help="A2B means photo->art if train-a was photo and train-b was artwork.",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Generator input size. Inference keeps original output size after restoration.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--generator-channels", type=int, default=64, help="Must match training config.")
    parser.add_argument("--discriminator-channels", type=int, default=64, help="Must match training config.")
    parser.add_argument("--res-blocks", type=int, default=9, help="Must match training config.")
    parser.add_argument(
        "--official-norm",
        type=str,
        default="instance",
        choices=["instance", "batch", "none"],
        help="Normalization used when loading official pytorch-CycleGAN-and-pix2pix weights.",
    )
    parser.add_argument(
        "--official-use-dropout",
        action="store_true",
        help="Enable dropout when loading official pytorch-CycleGAN-and-pix2pix generators.",
    )
    parser.add_argument("--comparison-dir", type=str, default="", help="Optional directory for side-by-side comparison images.")
    parser.add_argument("--generated-label", type=str, default="generated", help="Label shown on the generated panel.")
    parser.add_argument("--baseline-label", type=str, default="pretrained", help="Label shown on the public/pretrained panel.")
    parser.add_argument("--label-height", type=int, default=36, help="Label area height for comparison sheets.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_label(image: Image.Image, label: str, label_height: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((12, 10), label, fill=(0, 0, 0))
    return canvas


def compose_row(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    tensor = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = ((tensor + 1.0) * 127.5).permute(1, 2, 0).numpy().astype("uint8")
    return Image.fromarray(array)


def preprocess_image_official(image: Image.Image, image_size: int) -> tuple[torch.Tensor, dict[str, int]]:
    width, height = image.size
    resized = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = transforms.ToTensor()(resized)
    tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor)
    meta = {
        "orig_width": width,
        "orig_height": height,
    }
    return tensor.unsqueeze(0), meta


def postprocess_image_official(image_tensor: torch.Tensor, meta: dict[str, int]) -> Image.Image:
    image = tensor_to_pil(image_tensor)
    return image.resize((meta["orig_width"], meta["orig_height"]), Image.Resampling.LANCZOS)


def extract_generator_state_dict(checkpoint: dict[str, object], direction: str) -> dict[str, torch.Tensor]:
    if "netG_A" in checkpoint and "netG_B" in checkpoint:
        selected = checkpoint["netG_A"] if direction == "A2B" else checkpoint["netG_B"]
        return selected  # type: ignore[return-value]
    if "G_A" in checkpoint and "G_B" in checkpoint:
        selected = checkpoint["G_A"] if direction == "A2B" else checkpoint["G_B"]
        return selected  # type: ignore[return-value]
    return checkpoint  # type: ignore[return-value]


def load_generator_from_checkpoint(
    checkpoint_path: str,
    direction: str,
    official_norm: str,
    official_use_dropout: bool,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")
    state_dict = extract_generator_state_dict(checkpoint, direction)
    if not isinstance(state_dict, dict):
        raise TypeError("Unsupported generator state_dict format.")
    generator = build_official_generator_from_state_dict(
        state_dict,
        norm_type=official_norm,
        use_dropout=official_use_dropout,
    ).to(device)
    generator.eval()
    return generator


def load_generator(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return load_generator_from_checkpoint(
        checkpoint_path=args.checkpoint,
        direction=args.direction,
        official_norm=args.official_norm,
        official_use_dropout=args.official_use_dropout,
        device=device,
    )


def run_generator(
    image: Image.Image,
    generator: torch.nn.Module,
    image_size: int,
    device: torch.device,
) -> Image.Image:
    tensor, meta = preprocess_image_official(image, image_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        generated = generator(tensor)
    return postprocess_image_official(generated, meta)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    input_paths = collect_image_paths(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_output_dir = Path(args.baseline_output_dir) if args.baseline_output_dir else None
    if baseline_output_dir is not None:
        baseline_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = Path(args.comparison_dir) if args.comparison_dir else None
    if comparison_dir is not None:
        comparison_dir.mkdir(parents=True, exist_ok=True)
    generator = load_generator(args, device)
    baseline_generator = None
    if args.baseline_checkpoint:
        baseline_generator = load_generator_from_checkpoint(
            checkpoint_path=args.baseline_checkpoint,
            direction=args.direction,
            official_norm=args.official_norm,
            official_use_dropout=args.official_use_dropout,
            device=device,
        )

    for image_path in tqdm(input_paths, desc="infer", ncols=100):
        image = Image.open(image_path).convert("RGB")
        baseline_restored = None
        if baseline_generator is not None:
            baseline_restored = run_generator(
                image=image,
                generator=baseline_generator,
                image_size=args.image_size,
                device=device,
            )
            if baseline_output_dir is not None:
                baseline_restored.save(baseline_output_dir / image_path.name)
        restored = run_generator(
            image=image,
            generator=generator,
            image_size=args.image_size,
            device=device,
        )
        restored.save(output_dir / image_path.name)

        if comparison_dir is not None:
            panels = [add_label(image, "input", args.label_height)]
            if baseline_restored is not None:
                panels.append(add_label(baseline_restored, args.baseline_label, args.label_height))
            panels.append(add_label(restored, args.generated_label, args.label_height))
            compose_row(panels).save(comparison_dir / image_path.name)

    print(f"Finished inference for {len(input_paths)} images. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
