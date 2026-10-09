"""List-aware image compressor, adapted from ComfyUI-Image-Compressor.

Upstream: https://github.com/liuqianhonga/ComfyUI-Image-Compressor
MIT License
Copyright (c) 2024 liuqianhong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import io
import os
from datetime import datetime
from uuid import uuid4

import folder_paths
import numpy as np
import torch
from PIL import Image


def _list_values(value):
    """Unwrap ComfyUI list inputs, including a list-valued STRING output."""
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in _list_values(part)]
    return [value]


def _image_samples(images):
    for value in _list_values(images):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == 3:
            yield tensor
        elif tensor.ndim == 4:
            yield from tensor
        else:
            raise ValueError("images 必须是 IMAGE 张量、图片批次或图片列表")


def _size_label(size):
    return f"{size / 1048576:.2f}MB" if size >= 1048576 else f"{size / 1024:.2f}KB"


class MixlabImageCompressor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "WEBP", "JPEG"],),
                "quality": ("INT", {"default": 85, "min": 1, "max": 100, "display": "slider"}),
                "resize_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1, "display": "slider"}),
                "compression_level": ("INT", {"default": 6, "min": 0, "max": 9, "display": "slider"}),
                "save_image": ("BOOLEAN", {"default": True}),
                "output_prefix": ("STRING", {"default": "compressed_"}),
            },
            "optional": {"output_path": ("STRING", {"default": ""})},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("compression_info", "images")
    OUTPUT_IS_LIST = (False, True)
    OUTPUT_NODE = True
    FUNCTION = "compress_image"
    CATEGORY = "image/Mixlab"

    def compress_image(self, images, format, quality=85, resize_factor=1.0,
                       compression_level=6, save_image=True,
                       output_prefix="compressed_", output_path=""):
        samples = list(_image_samples(images))
        prefixes = _list_values(output_prefix)
        if len(prefixes) not in (1, len(samples)):
            raise ValueError(
                f"output_prefix 有 {len(prefixes)} 项，图片有 {len(samples)} 张；"
                "请提供一个共用前缀，或为每张图片提供一个前缀"
            )
        if any(not isinstance(prefix, str) for prefix in prefixes):
            raise ValueError("output_prefix 的每一项必须是文本")

        settings = {
            "format": _list_values(format),
            "quality": _list_values(quality),
            "resize_factor": _list_values(resize_factor),
            "compression_level": _list_values(compression_level),
            "save_image": _list_values(save_image),
            "output_path": _list_values(output_path),
        }
        for name, values in settings.items():
            if not values:
                raise ValueError(f"{name} 不能为空列表")

        base_output_dir = os.path.abspath(folder_paths.get_output_directory())
        outputs, infos, ui_images = [], [], []
        for index, sample in enumerate(samples):
            # Match ComfyUI's normal mapping behavior for non-prefix settings.
            options = {name: values[min(index, len(values) - 1)] for name, values in settings.items()}
            image_format = str(options["format"]).upper()
            if image_format not in ("PNG", "WEBP", "JPEG"):
                raise ValueError(f"不支持的压缩格式: {image_format}")
            array = sample.detach().cpu().numpy()
            if array.shape[-1] not in (1, 3, 4):
                raise ValueError("IMAGE 张量需要 1、3 或 4 个颜色通道")
            if array.size and array.max() > 1:
                array = array / 255.0
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)

            original_buffer = io.BytesIO()
            image.save(original_buffer, format="PNG")
            factor = float(options["resize_factor"])
            if not 0.1 <= factor <= 1.0:
                raise ValueError("resize_factor 必须在 0.1 到 1.0 之间")
            if factor < 1:
                image = image.resize(tuple(max(1, int(size * factor)) for size in image.size), Image.Resampling.LANCZOS)
            if image_format == "JPEG" and image.mode == "RGBA":
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background

            if image_format == "PNG":
                save_options = {"optimize": True, "compress_level": int(options["compression_level"])}
            elif image_format == "JPEG":
                save_options = {"quality": int(options["quality"]), "optimize": True, "subsampling": 1}
            else:
                save_options = {"quality": int(options["quality"]), "method": 6, "lossless": False, "alpha_quality": int(options["quality"])}
            buffer = io.BytesIO()
            image.save(buffer, format=image_format, **save_options)

            save_path_str = "File not saved"
            if options["save_image"]:
                path = str(options["output_path"] or "")
                output_dir = os.path.abspath(path if os.path.isabs(path) else os.path.join(base_output_dir, path.strip("/\\") or "compressed"))
                prefix = prefixes[0] if len(prefixes) == 1 else prefixes[index]
                # Preserve prefix subfolders while keeping the filename itself
                # separate for ComfyUI's preview endpoint.
                stem = f"{prefix}{datetime.now():%Y%m%d_%H%M%S_%f}_{index:04d}_{uuid4().hex[:8]}"
                save_path = os.path.join(output_dir, f"{stem}.{image_format.lower()}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "xb") as file:
                    file.write(buffer.getvalue())
                save_path_str = save_path
                try:
                    if os.path.commonpath([base_output_dir, os.path.abspath(save_path)]) == base_output_dir:
                        subfolder = os.path.relpath(os.path.dirname(save_path), base_output_dir)
                        ui_images.append({"filename": os.path.basename(save_path), "subfolder": "" if subfolder == "." else subfolder, "type": "output"})
                except ValueError:
                    pass  # An absolute output path can be on another drive.

            buffer.seek(0)
            with Image.open(buffer) as compressed:
                decoded = np.asarray(compressed.convert("RGBA" if compressed.mode == "RGBA" else "RGB"), dtype=np.float32).copy() / 255.0
            outputs.append(torch.from_numpy(decoded).unsqueeze(0).to(sample.device))
            infos.append(f"{save_path_str}: {_size_label(original_buffer.tell())} -> {_size_label(len(buffer.getvalue()))}")

        result = {"result": ("Compression results:\n\n" + "\n".join(infos), outputs)}
        if ui_images:
            result["ui"] = {"images": ui_images}
        return result


NODE_CLASS_MAPPINGS = {"MixlabImageCompressor": MixlabImageCompressor}
NODE_DISPLAY_NAME_MAPPINGS = {"MixlabImageCompressor": "🐟 Image Compressor · Mixlab"}
