# -*- coding: utf-8 -*-

from .py.bimoai_text_split import BimoAITextSplitIndex
from .py.color_ratio_node import ColorRatioCalculator
from .py.load_image_from_url import LoadImageAndMaskFromUrl
from .py.mask_sort import MaskSorter
from .py.split_string import SplitString
from .py.switch_case_node import SwitchCaseNodePro
from .py.was_text_shuffle import WASTextShuffle
from .py.zho_text_image import Text_Image_Multiline_Zho_autofit, Text_Image_Zho_autofit

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


NODE_CLASS_MAPPINGS = {
    "BimoAITextSplitIndex": BimoAITextSplitIndex,
    "ColorRatioCalculator": ColorRatioCalculator,
    "LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl,
    "MaskSorter": MaskSorter,
    "SplitString": SplitString,
    "SwitchCaseNodePro": SwitchCaseNodePro,
    "Text_Image_Multiline_Zho_autofit": Text_Image_Multiline_Zho_autofit,
    "Text_Image_Zho_autofit": Text_Image_Zho_autofit,
    "WASTextShuffle": WASTextShuffle,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BimoAITextSplitIndex": "BimoAI文本分隔元素读取",
    "ColorRatioCalculator": "Color Ratio Calculator",
    "LoadImageAndMaskFromUrl": "Load Image And Mask From Url",
    "MaskSorter": "🧩 Mask Sorter (多蒙版排序)",
    "SplitString": "Split String",
    "SwitchCaseNodePro": "Switch Case Node Pro",
    "Text_Image_Multiline_Zho_autofit": "Text Image Multiline Zho AutoFit",
    "Text_Image_Zho_autofit": "Text Image Zho AutoFit",
    "WASTextShuffle": "WAS Text Shuffle",
}
