from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from imggen import collect_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side comparison sheets.")
    parser.add_argument("--input", type=str, required=True, help="Original input image folder.")
    parser.add_argument("--columns", type=str, nargs="+", required=True, help="label=folder pairs, e.g. input=data/photos monet=outputs/monet")
    parser.add_argument("--output-dir", type=str, default="outputs/comparison_sheet", help="Output directory.")
    parser.add_argument("--label-height", type=int, default=36, help="Label area height.")
    return parser.parse_args()


def add_label(image: Image.Image, label: str, label_height: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((12, 10), label, fill=(0, 0, 0))
    return canvas


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    scale = target_height / image.height
    return image.resize((max(1, round(image.width * scale)), target_height), Image.Resampling.LANCZOS)


def parse_columns(raw_columns: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in raw_columns:
        if "=" not in item:
            raise ValueError(f"Invalid column spec: {item}. Expected label=folder.")
        label, folder = item.split("=", 1)
        parsed.append((label, Path(folder)))
    return parsed


def compose_row(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def main() -> None:
    args = parse_args()
    input_paths = collect_image_paths(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No images found under: {args.input}")
    columns = parse_columns(args.columns)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in input_paths:
        original = Image.open(input_path).convert("RGB")
        panels = [add_label(original, "input", args.label_height)]
        for label, folder in columns:
            generated_path = folder / input_path.name
            if not generated_path.exists():
                continue
            generated = Image.open(generated_path).convert("RGB")
            generated = resize_to_height(generated, original.height)
            panels.append(add_label(generated, label, args.label_height))
        sheet = compose_row(panels)
        sheet.save(output_dir / input_path.name)
        print(f"saved {output_dir / input_path.name}")


if __name__ == "__main__":
    main()
