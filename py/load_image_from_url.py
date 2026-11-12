"""Standalone ComfyUI URL image loader, compatible with the original Mixlab node.

IMAGE contains straight RGB (including colors under fully transparent pixels).
MASK contains alpha: 1 = opaque, 0 = transparent, as in the original node.
retry_count is EXTRA attempts; timeout is the requests connect/read inactivity
timeout in seconds, not an overall download deadline. Images are cached on disk.
Install by replacing the original module, or place this file in custom_nodes.
"""

import logging
import math
import os
import time
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif"}


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


def _cache_path(url):
    import folder_paths

    digest = sha256(url.encode("utf-8")).hexdigest()
    try:
        suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    except ValueError:
        suffix = ""
    if suffix not in _IMAGE_SUFFIXES:
        suffix = ".img"
    return Path(folder_paths.get_input_directory()) / "url_images" / (digest + suffix)


def _decode_image(file):
    with Image.open(file) as source:
        # Apply rotations and reflections before separating color and alpha.
        oriented = ImageOps.exif_transpose(source)
        oriented.load()
        # Expand palette transparency without compositing or premultiplying RGB.
        rgba = oriented.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")


def _write_cache(path, content):
    """Publish only complete files, even if two workers download the same URL."""
    temp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(content)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_image_and_mask_from_url(url, timeout=10, retry_count=3, retry_interval=1.0):
    """Read a cached image first; retry HTTP, network and image decode failures."""
    _validate_settings(retry_count, retry_interval, timeout)
    cache_path = _cache_path(url)
    try:
        if not cache_path.is_symlink() and cache_path.is_file():
            return _decode_image(cache_path)
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        logger.warning(
            "Cached image could not be decoded (%s); downloading again", type(exc).__name__
        )

    for attempt in range(retry_count + 1):
        try:
            with requests.get(url, timeout=timeout) as response:
                response.raise_for_status()
                content = response.content
            image_and_mask = _decode_image(BytesIO(content))
        except (
            requests.RequestException,
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as exc:
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
            continue

        try:
            _write_cache(cache_path, content)
        except OSError as exc:
            logger.warning("Could not save image to the input cache (%s)", type(exc).__name__)
        return image_and_mask


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
                img, mask = load_image_and_mask_from_url(
                    img_url,
                    timeout=timeout,
                    retry_count=retry_count,
                    retry_interval=retry_interval,
                )
                images.append(pil2tensor(img))
                masks.append(pil2tensor(mask))
            except (RuntimeError, requests.RequestException, OSError, SyntaxError) as exc:
                # Preserve the original batch behavior: skip failed URLs.
                logger.error("URL item %d skipped: %s", index, exc)
        return images, masks


NODE_CLASS_MAPPINGS = {"LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageAndMaskFromUrl": "Load Image And Mask From Url"}
