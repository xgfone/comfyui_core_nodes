"""OpenCV CPU implementation of ImageScaleByAspectRatio V2."""

import math

import cv2
import numpy as np
import torch

# from .imagefunc import is_valid_mask, log, num_round_up_to_multiple


def is_valid_mask(tensor: torch.Tensor) -> bool:
    return not bool(torch.all(tensor == 0).item())


def log(message: str, message_type: str = "info"):
    name = "LayerStyle"

    if message_type == "error":
        message = "\033[1;41m" + message + "\033[m"
    elif message_type == "warning":
        message = "\033[1;31m" + message + "\033[m"
    elif message_type == "finish":
        message = "\033[1;32m" + message + "\033[m"
    else:
        message = "\033[1;33m" + message + "\033[m"
    print(f"# 😺dzNodes: {name} -> {message}")


try:
    from cv2.ximgproc import guidedFilter
except ImportError:
    # print(e)
    log(
        "Cannot import name 'guidedFilter' from 'cv2.ximgproc'"
        "\nA few nodes cannot works properly, while most nodes are not affected. Please REINSTALL package 'opencv-contrib-python'."
        "\nFor detail refer to \033[4mhttps://github.com/chflame163/ComfyUI_LayerStyle/issues/5\033[0m"
    )


# 向上取整数倍
def num_round_up_to_multiple(number: int, multiple: int) -> int:
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        factor = (number + multiple - 1) // multiple  # 向上取整的计算方式
        return factor * multiple


def _hex_to_rgb(color):
    """Convert #RGB or #RRGGBB to an RGB float32 array in ComfyUI's 0..1 range."""
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    try:
        return np.asarray(
            [int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4)],
            dtype=np.float32,
        )
    except (TypeError, ValueError, IndexError):
        return np.zeros(3, dtype=np.float32)


def _resize_with_fit(source, target_width, target_height, fit, interpolation, background=None):
    """Resize one HWC image or HW mask using OpenCV."""
    source_height, source_width = source.shape[:2]

    if fit == "fill":
        return cv2.resize(
            source,
            (target_width, target_height),
            interpolation=interpolation,
        )

    if fit == "crop":
        scale = max(target_width / source_width, target_height / source_height)
    else:  # letterbox
        scale = min(target_width / source_width, target_height / source_height)

    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        source,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    if fit == "crop":
        left = max(0, (resized_width - target_width) // 2)
        top = max(0, (resized_height - target_height) // 2)
        return np.ascontiguousarray(resized[top : top + target_height, left : left + target_width])

    if source.ndim == 2:
        canvas = np.zeros((target_height, target_width), dtype=source.dtype)
    else:
        channels = source.shape[2]
        color = np.zeros(channels, dtype=source.dtype)
        if background is not None:
            count = min(channels, background.size)
            color[:count] = background[:count].astype(source.dtype, copy=False)
        canvas = np.empty((target_height, target_width, channels), dtype=source.dtype)
        canvas[...] = color

    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


class ImageScaleByAspectRatioOpenCV:
    def __init__(self):
        self.NODE_NAME = "ImageScaleByAspectRatio OpenCV"

    @classmethod
    def INPUT_TYPES(cls):
        ratio_list = ["original", "custom", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"]
        fit_mode = ["letterbox", "crop", "fill"]
        method_mode = ["INTER_AREA", "INTER_LANCZOS4"]
        multiple_list = ["8", "16", "32", "64", "128", "256", "512", "None"]
        scale_to_list = [
            "None",
            "longest",
            "shortest",
            "width",
            "height",
            "total_pixel(kilo pixel)",
        ]
        return {
            "required": {
                "aspect_ratio": (ratio_list,),
                "proportional_width": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100000000, "step": 1},
                ),
                "proportional_height": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100000000, "step": 1},
                ),
                "fit": (fit_mode,),
                "method": (method_mode,),
                "round_to_multiple": (multiple_list,),
                "scale_to_side": (scale_to_list,),
                "scale_to_length": (
                    "INT",
                    {"default": 1024, "min": 4, "max": 100000000, "step": 1},
                ),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height")
    FUNCTION = "image_scale_by_aspect_ratio"
    CATEGORY = "😺dzNodes/LayerUtility"

    def image_scale_by_aspect_ratio(
        self,
        aspect_ratio,
        proportional_width,
        proportional_height,
        fit,
        method,
        round_to_multiple,
        scale_to_side,
        scale_to_length,
        background_color,
        image=None,
        mask=None,
    ):
        if image is None and mask is None:
            log(
                f"Error: {self.NODE_NAME} skipped because no image or mask was provided.",
                message_type="error",
            )
            return (None, None, None, 0, 0)

        if image is not None:
            orig_height, orig_width = image.shape[1:3]
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            orig_height, orig_width = mask.shape[-2:]

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape[-2:]) != (orig_height, orig_width):
                log(
                    f"Error: {self.NODE_NAME} mask size does not match image size.",
                    message_type="error",
                )
                return (None, None, None, 0, 0)

        if aspect_ratio == "original":
            ratio = orig_width / orig_height
        elif aspect_ratio == "custom":
            ratio = proportional_width / proportional_height
        else:
            ratio_width, ratio_height = map(int, aspect_ratio.split(":"))
            ratio = ratio_width / ratio_height

        if scale_to_side == "total_pixel(kilo pixel)":
            target_width = int(math.sqrt(ratio * scale_to_length * 1000))
            target_height = int(target_width / ratio)
        elif (
            scale_to_side == "width"
            or (scale_to_side == "longest" and ratio > 1)
            or (scale_to_side == "shortest" and ratio <= 1)
        ):
            target_width = scale_to_length
            target_height = int(target_width / ratio)
        elif (
            scale_to_side == "height"
            or (scale_to_side == "longest" and ratio <= 1)
            or (scale_to_side == "shortest" and ratio > 1)
        ):
            target_height = scale_to_length
            target_width = int(target_height * ratio)
        elif ratio > 1:
            target_width = orig_width
            target_height = int(target_width / ratio)
        else:
            target_height = orig_height
            target_width = int(target_height * ratio)

        if round_to_multiple != "None":
            multiple = int(round_to_multiple)
            target_width = num_round_up_to_multiple(target_width, multiple)
            target_height = num_round_up_to_multiple(target_height, multiple)

        interpolation = {
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
        }[method]

        result_image = None
        if image is not None:
            image_device = image.device
            image_dtype = image.dtype
            # A CPU contiguous ComfyUI tensor exposes a zero-copy NumPy view here.
            image_array = image.detach().to("cpu").contiguous().numpy()
            background = _hex_to_rgb(background_color)
            resized_images = [
                _resize_with_fit(
                    source,
                    target_width,
                    target_height,
                    fit,
                    interpolation,
                    background,
                )
                for source in image_array
            ]
            result_image = torch.from_numpy(np.stack(resized_images, axis=0))
            result_image = result_image.clamp_(0, 1).to(dtype=image_dtype, device=image_device)

        result_mask = None
        if mask is not None:
            mask_device = mask.device
            mask_dtype = mask.dtype
            mask_array = mask.detach().to("cpu").contiguous().numpy()
            resized_masks = []
            for source_tensor, source_array in zip(mask, mask_array):
                if source_array.shape == (64, 64) and not is_valid_mask(source_tensor.unsqueeze(0)):
                    log(
                        f"Warning: {self.NODE_NAME} input mask is empty; ignoring it.",
                        message_type="warning",
                    )
                    continue
                resized_masks.append(
                    _resize_with_fit(
                        source_array,
                        target_width,
                        target_height,
                        fit,
                        interpolation,
                    )
                )
            if resized_masks:
                result_mask = torch.from_numpy(np.stack(resized_masks, axis=0))
                result_mask = result_mask.clamp_(0, 1).to(dtype=mask_dtype, device=mask_device)

        processed_count = image.shape[0] if image is not None else mask.shape[0]
        log(
            f"{self.NODE_NAME} processed {processed_count} item(s) with {method}.",
            message_type="finish",
        )
        return (
            result_image,
            result_mask,
            [orig_width, orig_height],
            target_width,
            target_height,
        )


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageScaleByAspectRatio OpenCV": ImageScaleByAspectRatioOpenCV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageScaleByAspectRatio OpenCV": "LayerUtility: ImageScaleByAspectRatio OpenCV"
}
