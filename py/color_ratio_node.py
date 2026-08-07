import numpy as np


class ColorRatioCalculator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255}),
                "tolerance": ("INT", {"default": 10, "min": 0, "max": 255}),
            }
        }

    FUNCTION = "calculate"
    CATEGORY = "Conchshell Image Analysis"

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("color_ratio",)

    def calculate(self, image, r, g, b, tolerance):
        # image shape is [Batch, Height, Width, Channel]
        # We take the first image in the batch
        img_tensor = image[0]  # shape is already [H, W, C]

        # Get dimensions correctly
        h, w, c = img_tensor.shape

        # Convert tensor to numpy and scale to 0-255
        # No permute is needed because ComfyUI uses [H, W, C] for individual tensors
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # [H, W, C]

        target = np.array([r, g, b])

        # Calculate absolute difference
        # Ensure we only check the first 3 channels (ignore Alpha if present)
        diff = np.abs(img_np[:, :, :3] - target)

        # Check if difference is within tolerance for all channels
        mask = np.all(diff <= tolerance, axis=2)

        color_pixels = np.sum(mask)
        total_pixels = h * w
        ratio = color_pixels / total_pixels

        return (float(round(ratio, 4)),)


NODE_CLASS_MAPPINGS = {"ColorRatioCalculator": ColorRatioCalculator}
NODE_DISPLAY_NAME_MAPPINGS = {"ColorRatioCalculator": "Color Ratio Calculator"}
