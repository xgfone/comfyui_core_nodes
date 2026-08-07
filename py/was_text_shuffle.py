import random


class WASTextShuffle:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "text"}),
                "separator": ("STRING", {"default": ",", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "shuffle"
    CATEGORY = "WAS Suite/Text/Operations"

    def shuffle(self, text, separator, seed):
        if seed is not None:
            random.seed(seed)

        text_list = text.split(separator)
        random.shuffle(text_list)
        new_text = separator.join(text_list)

        return (new_text,)
