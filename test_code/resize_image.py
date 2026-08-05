# -*- coding: utf-8 -*-
"""
把一张图片转换成多种分辨率，每种分辨率生成一个新图片。

用法:
    python resize_image.py <输入图片> [-o 输出目录]

需要安装依赖:
    pip install Pillow
"""

import argparse
import os

from PIL import Image

# 目标分辨率: (名称, 目标宽度)
# 保持原图宽高比，高度按比例自动计算
TARGET_SIZES = [
    ("tiny", 320),
    ("small", 480),
    ("medium", 960),
    ("large", 1440),
    ("xlarge", 1920),
    ("huge", 2560),
]


def resize_image(input_path: str, output_dir: str = ".",
                 target_sizes=TARGET_SIZES) -> list[str]:
    """把 input_path 指向的图片缩放到多种分辨率，返回生成的文件路径列表。"""
    img = Image.open(input_path)
    orig_w, orig_h = img.size
    print(f"原图: {input_path}  ({orig_w}x{orig_h})")

    name, ext = os.path.splitext(os.path.basename(input_path))
    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    for label, target_w in target_sizes:
        # 等比缩放
        target_h = round(orig_h * target_w / orig_w)
        resized = img.resize((target_w, target_h), Image.LANCZOS)

        out_path = os.path.join(output_dir, f"{name}_{label}_{target_w}x{target_h}{ext}")
        resized.save(out_path)
        output_files.append(out_path)
        print(f"已生成: {out_path}")

    return output_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="把图片转换成多种分辨率")
    parser.add_argument("input", nargs="?", default="C:\\files\\vllm_notes\\test_code\\20260805221917_467_29.jpg", help="输入图片路径")
    parser.add_argument("-o", "--output-dir", default=".", help="输出目录 (默认当前目录)")
    args = parser.parse_args()

    resize_image(args.input, args.output_dir)
