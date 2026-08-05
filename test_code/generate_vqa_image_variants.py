#!/usr/bin/env python3
"""Generate controlled image variants for VQA rows.

This file is standalone: it imports no project code and only requires Pillow.
Input may be JSONL, a JSON list, or a JSON object containing a ``data`` list.
Every output row keeps the original VQA fields while replacing its image path.

Example:
    python generate_vqa_image_variants.py \
      --input vqa.jsonl \
      --output-dir generated_vqa
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

try:
    from PIL import Image, ImageFilter
except ImportError as error:
    raise SystemExit("Pillow is required: pip install pillow") from error


RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
IMAGE_FIELDS = ("media_path", "image_path", "media_url", "image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="VQA JSON or JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Default: <output-dir>/variants.jsonl",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--blur-radius", type=float, default=3.0)
    parser.add_argument("--crop-fraction", type=float, default=0.85)
    parser.add_argument("--limit", type=int, help="Process at most this many input rows")
    parser.add_argument("--include-native", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("data")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("input must be JSONL rows, a JSON list, or {'data': [...]} ")
    return value


def source_path(row: dict[str, Any], input_dir: Path) -> tuple[str, Path]:
    for field in IMAGE_FIELDS:
        raw = row.get(field)
        if not raw:
            continue
        parsed = urlparse(str(raw))
        if parsed.scheme not in ("", "file"):
            raise ValueError(f"only local images are supported, got {raw!r}")
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else str(raw))
        if not path.is_absolute():
            path = input_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return field, path.resolve()
    raise ValueError(f"row has none of the image fields: {IMAGE_FIELDS}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_long_edge(image: Image.Image, size: int, resampling: int) -> Image.Image:
    scale = size / max(image.size)
    shape = tuple(max(1, round(value * scale)) for value in image.size)
    return image.resize(shape, resampling)


def variants(
    image: Image.Image,
    *,
    size: int,
    blur_radius: float,
    crop_fraction: float,
) -> Iterable[tuple[str, Image.Image, dict[str, Any]]]:
    rgb = image.convert("RGB")
    for method, resampling in RESAMPLING.items():
        yield (
            f"{method}_{size}",
            resize_long_edge(rgb, size, resampling),
            {"type": "resize", "interpolation": method, "target_long_edge": size},
        )

    square = (size, size)
    yield (
        f"grayscale_{size}",
        rgb.convert("L").convert("RGB").resize(square, Image.Resampling.BICUBIC),
        {"type": "grayscale", "target_size": [size, size]},
    )
    yield (
        f"blur_{size}",
        rgb.resize(square, Image.Resampling.BICUBIC).filter(
            ImageFilter.GaussianBlur(blur_radius)
        ),
        {
            "type": "gaussian_blur",
            "radius": blur_radius,
            "target_size": [size, size],
        },
    )

    removed_fraction = (1.0 - crop_fraction) / 2.0
    dx = int(rgb.width * removed_fraction)
    dy = int(rgb.height * removed_fraction)
    crop_box = (dx, dy, rgb.width - dx, rgb.height - dy)
    crop_name = f"center_crop_{round(crop_fraction * 100):02d}"
    yield (
        crop_name,
        rgb.crop(crop_box).resize(square, Image.Resampling.BICUBIC),
        {
            "type": "center_crop",
            "crop_fraction": crop_fraction,
            "crop_box": list(crop_box),
            "target_size": [size, size],
        },
    )


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def main() -> None:
    args = parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")
    if args.blur_radius < 0:
        raise SystemExit("--blur-radius must be non-negative")
    if not 0 < args.crop_fraction <= 1:
        raise SystemExit("--crop-fraction must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    rows = [row for row in read_rows(args.input) if row.get("_type") != "metadata"]
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = args.output_dir.resolve()
    output_jsonl = (args.output_jsonl or output_dir / "variants.jsonl").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    materialized: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows):
        image_field, path = source_path(row, args.input.resolve().parent)
        digest = file_sha256(path)
        with Image.open(path) as loaded:
            rgb = loaded.convert("RGB")
        source_id = str(row.get("media_id") or row.get("image_id") or digest)

        items = list(
            variants(
                rgb,
                size=args.size,
                blur_radius=args.blur_radius,
                crop_fraction=args.crop_fraction,
            )
        )
        if args.include_native:
            items.insert(0, ("native", rgb, {"type": "native"}))

        for variant_name, image, parameters in items:
            destination = output_dir / "images" / variant_name / f"{digest[:20]}.png"
            materialization_key = (digest, variant_name)
            if materialization_key not in materialized:
                save_png(image, destination)
                materialized.add(materialization_key)

            output = dict(row)
            output[image_field] = str(destination)
            if image_field == "media_url":
                output[image_field] = destination.as_uri()
            output["media_path"] = str(destination)
            output["media_url"] = destination.as_uri()
            output["source_media_id"] = source_id
            output["source_media_path"] = str(path)
            output["source_media_sha256"] = digest
            output["variant_name"] = variant_name
            output["variant"] = parameters
            output["request_id"] = (
                f"{row.get('request_id', row.get('question_id', row_index))}-{variant_name}"
            )
            generated.append(output)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "output_rows": len(generated),
                "variants_per_row": 8 if args.include_native else 7,
                "unique_images_written": len(materialized),
                "output_jsonl": str(output_jsonl),
                "image_root": str(output_dir / "images"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()