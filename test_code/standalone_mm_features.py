"""
standalone_mm_features.py

不启动 vLLM 引擎进程，复现 P0 前端对图片的多模态处理，
得到与 request.mm_features[i].data 等价的 MultiModalKwargsItem。

原理（vllm/v1/engine/input_processor.py:364-378）:
    mm_features[i].data == processor.apply(...)["mm_kwargs"][modality][i]
即 BaseMultiModalProcessor.apply() 返回的 mm_kwargs 中按模态/位置取出的
MultiModalKwargsItem（对 Qwen-VL 图片: pixel_values + image_grid_thw）。

运行方式（Windows，使用 vllm 仓库的虚拟环境）:
    ./.venv/Scripts/python standalone_mm_features.py
首次运行会下载 HF config 和 tokenizer（不加载模型权重）。
"""

from PIL import Image

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # 或本地模型路径
IMAGE_PATH = "test.jpg"


def main():
    # 1. 只需要 ModelConfig：读 HF config + tokenizer，不加载权重、不起进程
    model_config = ModelConfig(model=MODEL_ID)

    # 2. 从注册表创建 Qwen3VLMultiModalProcessor
    #    cache=None：禁用 processor cache，结果 = 缓存未命中时的全新处理，
    #    拿到的永远是含真实张量的 item（而不是 shm 缓存下的 address 条目）。
    processor = MULTIMODAL_REGISTRY.create_processor(model_config, cache=None)

    # 3. 原始图片 → MultiModalDataItems
    #    （与 P0 renderer 里 base.py:745 的 parse_mm_data 是同一步骤）
    image = Image.open(IMAGE_PATH).convert("RGB")
    mm_items = processor.info.parse_mm_data({"image": image})

    # 4. processor.__call__ 内部就是 apply()，与 P0 走完全相同的代码路径。
    #    注意 prompt 必须包含与图片数一致的占位符，否则 placeholder 校验会报错。
    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>描述这张图"
        "<|im_end|>\n"
    )
    mm_inputs = processor(prompt, mm_items)

    # 5. 取图片对应的 item —— 就是 request.mm_features[0].data
    data = mm_inputs["mm_kwargs"]["image"][0]

    print(type(data))  # <class 'vllm.multimodal.inputs.MultiModalKwargsItem'>
    for key, elem in data.items():
        print(f"{key}: shape={tuple(elem.data.shape)}, dtype={elem.data.dtype}")
    # 典型输出：
    # pixel_values:   shape=(N, 1176), dtype=torch.float32  # N=grid_t*h*w 个 patch
    # image_grid_thw: shape=(1, 3),    dtype=torch.int64

    # 顺便可以看到与之一配套的其它产物（P0 也会一并算出）：
    print("hash:", mm_inputs["mm_hashes"]["image"][0])
    print("placeholder:", mm_inputs["mm_placeholders"]["image"][0])

    return data


if __name__ == "__main__":
    main()
