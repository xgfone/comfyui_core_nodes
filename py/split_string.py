from .py import any


class SplitString:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": "text"}),
            },
            "optional": {
                "delimiter": ("STRING", {"multiline": False, "default": ","}),
            },
        }

    RETURN_TYPES = (
        any,
        any,
        any,
        any,
        any,
        any,
        any,
        any,
        any,
        any,
    )
    RETURN_NAMES = (
        "string_1",
        "string_2",
        "string_3",
        "string_4",
        "string_5",
        "string_6",
        "string_7",
        "string_8",
        "string_9",
        "string_10",
    )
    FUNCTION = "split"
    CATEGORY = "BIMO AI/text"

    def split(self, text, delimiter=""):
        # Split the text string
        parts = text.split(delimiter)
        strings = [part.strip() for part in parts[:10]]
        (
            string_1,
            string_2,
            string_3,
            string_4,
            string_5,
            string_6,
            string_7,
            string_8,
            string_9,
            string_10,
        ) = strings + [""] * (10 - len(strings))

        return (
            string_1,
            string_2,
            string_3,
            string_4,
            string_5,
            string_6,
            string_7,
            string_8,
            string_9,
            string_10,
        )
