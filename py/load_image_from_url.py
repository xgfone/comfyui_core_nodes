from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_image_and_mask_from_url(url, timeout=10):
    response = requests.get(url, timeout=timeout)
    image = Image.open(BytesIO(response.content))

    # Create a mask from the image's alpha channel
    mask = image.convert("RGBA").split()[-1]

    # Convert the mask to a black and white image
    mask = mask.convert("L")

    image = image.convert("RGB")
    return (image, mask)


class LoadImageAndMaskFromUrl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": (
                    "STRING",
                    {"multiline": True, "default": "https://", "dynamicPrompts": False},
                ),
            },
        }

    FUNCTION = "run"
    CATEGORY = "Mixlab/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True, True)

    global urls_image
    urls_image = {}

    def run(self, url, seed=0):
        global urls_image

        def filter_http_urls(urls):
            filtered_urls = []
            for url in urls.split("\n"):
                if url.startswith("http"):
                    filtered_urls.append(url)
            return filtered_urls

        filtered_urls = filter_http_urls(url)

        images = []
        masks = []

        for img_url in filtered_urls:
            try:
                if img_url in urls_image:
                    img, mask = urls_image[img_url]
                else:
                    img, mask = load_image_and_mask_from_url(img_url)
                    urls_image[img_url] = (img, mask)

                img1 = pil2tensor(img)
                mask1 = pil2tensor(mask)

                images.append(img1)
                masks.append(mask1)
            except Exception as e:
                print("wrap an exception:", str(e))

        return (
            images,
            masks,
        )
