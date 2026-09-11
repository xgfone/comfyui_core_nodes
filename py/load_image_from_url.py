"""Standalone ComfyUI URL image loader, compatible with the original Mixlab node.

IMAGE contains straight RGB (including colors under fully transparent pixels).
MASK contains alpha: 1 = opaque, 0 = transparent, as in the original node.
retry_count is EXTRA attempts; timeout is the requests connect/read inactivity
timeout in seconds, not an overall download deadline. Successful URLs are cached.
Install by replacing the original module, or place this file in custom_nodes.
"""

import logging
import math
import time
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
urls_image = {}


def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _validate_settings(retry_count, retry_interval, timeout):
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise ValueError("retry_count must be a non-negative integer")
    if not math.isfinite(retry_interval) or retry_interval < 0:
        raise ValueError("retry_interval must be finite and >= 0 seconds")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and > 0 seconds")


def load_image_and_mask_from_url(url, timeout=10, retry_count=3, retry_interval=1.0):
    """Download and fully decode; retry HTTP, network and image decode failures."""
    _validate_settings(retry_count, retry_interval, timeout)
    for attempt in range(retry_count + 1):
        try:
            with requests.get(url, timeout=timeout) as response:
                response.raise_for_status()
                with Image.open(BytesIO(response.content)) as source:
                    # Apply rotations AND reflections before separating color/alpha.
                    oriented = ImageOps.exif_transpose(source)
                    oriented.load()
                    # Also expands palette transparency and RGB PNG color keys.
                    rgba = oriented.convert("RGBA")
                    # Dropping alpha does not composite or premultiply RGB values.
                    return rgba.convert("RGB"), rgba.getchannel("A")
        except (requests.RequestException, OSError, SyntaxError) as exc:
            if attempt >= retry_count:
                raise RuntimeError(
                    "Image download/decode failed after "
                    f"{retry_count + 1} attempt(s) ({type(exc).__name__})"
                ) from exc
            logger.warning(
                "Image download/decode failed (%s); retry %d/%d in %s seconds",
                type(exc).__name__,
                attempt + 1,
                retry_count,
                retry_interval,
            )
            time.sleep(retry_interval)


class LoadImageAndMaskFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "https://",
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": {
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Extra retries after the first attempt; 0 disables retries.",
                    },
                ),
                "retry_interval": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.1,
                        "tooltip": "Seconds between failed attempts.",
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.1,
                        "max": 3600.0,
                        "step": 0.1,
                        "tooltip": "Connect/read inactivity timeout per request, in seconds.",
                    },
                ),
            },
        }

    FUNCTION = "run"
    CATEGORY = "Mixlab/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True, True)

    def run(self, url, seed=0, retry_count=3, retry_interval=1.0, timeout=10.0):
        # Keep seed's position for existing callers. Like the original it is unused.
        _validate_settings(retry_count, retry_interval, timeout)
        filtered_urls = [
            line.strip()
            for line in url.splitlines()
            if line.strip().lower().startswith(("http://", "https://"))
        ]
        images, masks = [], []
        for index, img_url in enumerate(filtered_urls, 1):
            try:
                if img_url in urls_image:
                    img, mask = urls_image[img_url]
                else:
                    img, mask = load_image_and_mask_from_url(
                        img_url,
                        timeout=timeout,
                        retry_count=retry_count,
                        retry_interval=retry_interval,
                    )
                    urls_image[img_url] = (img, mask)
                images.append(pil2tensor(img))
                masks.append(pil2tensor(mask))
            except (RuntimeError, requests.RequestException, OSError, SyntaxError) as exc:
                # Preserve the original batch behavior: skip failed URLs.
                logger.error("URL item %d skipped: %s", index, exc)
        return images, masks


NODE_CLASS_MAPPINGS = {"LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageAndMaskFromUrl": "Load Image And Mask From Url"}
