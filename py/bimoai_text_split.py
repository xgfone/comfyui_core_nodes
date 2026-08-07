class BimoAITextSplitIndex:
    """Split text by a delimiter and return the item at a zero-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "需要分割的文本。"},
                ),
                "sep": (
                    "STRING",
                    {"default": "|", "multiline": False, "tooltip": "用于分割文本的分隔符。"},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "从 0 开始读取；超出范围时返回最后一个元素。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本输出",)
    FUNCTION = "split_and_index"
    CATEGORY = "BIMOAI/文本工具"
    DESCRIPTION = "通过分隔符拆分文本，然后通过index读取元素，若超出范围时返回最后一个元素。"

    def split_and_index(self, input, sep, index):
        if sep == "" or sep not in input:
            items = [input]
        else:
            items = input.split(sep)

        selected_index = min(max(0, int(index)), len(items) - 1)
        return (items[selected_index],)
