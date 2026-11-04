"""Collect image inputs into a native ComfyUI IMAGE list."""


class ImageListMerge:
    MAX_INPUTS = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_count": ("INT", {"default": 2, "min": 1, "max": cls.MAX_INPUTS}),
            },
            "optional": {
                f"image_{index}": ("IMAGE",)
                for index in range(1, cls.MAX_INPUTS + 1)
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge"
    CATEGORY = "图像/列表"
    DESCRIPTION = "按接口顺序合并图片为列表；批量图片拆成单张，跳过未连接接口。"

    def merge(self, input_count, **kwargs):
        # ComfyUI wraps all inputs in lists when INPUT_IS_LIST is enabled.
        count = input_count[0] if isinstance(input_count, list) else input_count
        count = int(count)
        if not 1 <= count <= self.MAX_INPUTS:
            raise ValueError(f"输入数量必须在 1 到 {self.MAX_INPUTS} 之间")
        images = []
        for index in range(1, count + 1):
            batches = kwargs.get(f"image_{index}")
            if batches is None:
                continue
            if not isinstance(batches, list):
                batches = [batches]
            for batch in batches:
                if batch is not None:
                    images.extend(batch[i : i + 1] for i in range(batch.shape[0]))
        return (images,)


NODE_CLASS_MAPPINGS = {"BimoImageListMerge": ImageListMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"BimoImageListMerge": "图片列表合并"}
