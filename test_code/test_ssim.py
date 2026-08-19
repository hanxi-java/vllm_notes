# -*- coding: utf-8 -*-
"""
计算 resized_out 目录下各图片与参考图 qiaoliang.jpg 的 SSIM 差异，并输出每次计算的耗时。

说明：
- SSIM 要求两图尺寸一致，这里将待比较图统一 resize 到参考图尺寸（4096x3072, PIL LANCZOS）
- 计时拆分为两部分：图像加载+resize 耗时、纯 SSIM 计算耗时
- 参考图自身也会参与比较（预期 SSIM = 1.0），作为正确性校验

运行（系统 Python 3.12，venv 的 numpy 已损坏勿用）：
    C:\\program\\python.exe test_ssim.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Windows 控制台默认 GBK，强制 UTF-8 避免中文输出乱码
sys.stdout.reconfigure(encoding="utf-8")

IMG_DIR = Path(__file__).parent / "resized_out"
REF_NAME = "qiaoliang.jpg"


def load_as_float(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """加载图片为 float32 [0,1] 的 HWC 数组；size 为 (width, height) 时先 resize。"""
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def main() -> None:
    ref_path = IMG_DIR / REF_NAME
    ref_size = Image.open(ref_path).size  # (width, height)
    print(f"参考图: {REF_NAME}  尺寸: {ref_size[0]}x{ref_size[1]}")
    print("-" * 72)

    t0 = time.perf_counter()
    ref = load_as_float(ref_path)
    print(f"参考图加载耗时: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print("-" * 72)
    print(f"{'图片':<38}{'SSIM':>8}{'resize(ms)':>12}{'ssim(ms)':>10}")
    print("-" * 72)

    total_ssim_ms = 0.0
    count = 0
    for path in sorted(IMG_DIR.glob("*.jpg")):
        t0 = time.perf_counter()
        img = load_as_float(path, size=ref_size)
        load_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        score = ssim(ref, img, channel_axis=-1, data_range=1.0)
        ssim_ms = (time.perf_counter() - t0) * 1000

        total_ssim_ms += ssim_ms
        count += 1
        print(f"{path.name:<38}{score:>8.4f}{load_ms:>12.1f}{ssim_ms:>10.1f}")

    print("-" * 72)
    print(f"共 {count} 对, SSIM 计算总耗时 {total_ssim_ms:.1f} ms, "
          f"平均 {total_ssim_ms / max(count, 1):.1f} ms/pair")


if __name__ == "__main__":
    main()
