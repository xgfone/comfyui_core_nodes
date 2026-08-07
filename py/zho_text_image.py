import logging
import math
import os
import re
import textwrap
from pathlib import Path
from typing import List, Union, cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
project_dir = Path(__file__).resolve().parent.parent
comfy_dir = project_dir.parent.parent


# ----------------------------------------------------------------------------
def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# ----------------------------------------------------------------------------
def swap_width_height(width, height):
    return height, width


# ----------------------------------------------------------------------------
base_log_level = logging.DEBUG if os.environ.get("MTB_DEBUG") else logging.INFO


class NullWriter:
    def write(self, text):
        pass


class Formatter(logging.Formatter):
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    purple = "\x1b[35;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format = "[%(name)s] | %(levelname)s -> %(message)s"

    FORMATS = {
        logging.DEBUG: purple + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def mklog(name, level=base_log_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(Formatter())
    logger.addHandler(ch)
    logger.propagate = False
    return logger


log = mklog(__package__ or __name__, base_log_level)


def log_user(arg):
    print(f"\033[34mComfy MTB Utils:\033[0m {arg}")


def get_summary(docstring):
    return docstring.strip().split("\n\n", 1)[0]


def blue_text(text):
    return f"\033[94m{text}\033[0m"


def cyan_text(text):
    return f"\033[96m{text}\033[0m"


def get_label(label):
    words = re.findall(r"(?:^|[A-Z])[a-z]*", label)
    return " ".join(words).strip()


logging.getLogger("aiohttp.access").disabled = True


# ----------------------------------------------------------------------------
def bbox_dim(bbox):
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper
    return width, height


# ----------------------------------------------------------------------------
# Auto-fit helpers
# ----------------------------------------------------------------------------


def _safe_font(font_path, font_size):
    font_size = max(1, int(font_size))
    return cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_size))


def _text_bbox(font, text, outline_size=0):
    # textbbox is closer to the real rendered result than font.getbbox,
    # especially when stroke_width is used.
    tmp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    return tmp_draw.textbbox((0, 0), text or " ", font=font, stroke_width=max(0, int(outline_size)))


def _estimate_wrap_chars(font, usable_width):
    # For wrap=0: estimate by average ASCII and CJK character widths.
    # This is more stable than width / font_size, especially for Chinese fonts.
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789一二三四五六七八九十"
    widths = []
    for char in sample:
        bbox = _text_bbox(font, char, 0)
        w = max(1, bbox[2] - bbox[0])
        widths.append(w)
    avg_char_width = max(1, sum(widths) / len(widths))
    return max(1, int(usable_width / avg_char_width))


def _wrap_paragraph(paragraph, font, wrap, usable_width):
    if wrap == 0:
        wrap_chars = _estimate_wrap_chars(font, usable_width)
    else:
        wrap_chars = max(1, int(wrap))

    lines = textwrap.wrap(
        paragraph,
        width=wrap_chars,
        expand_tabs=False,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return lines if lines else [""]


def _build_lines(text, font_path, font_size, wrap, usable_width, multiline=False):
    font = _safe_font(font_path, font_size)

    if multiline:
        paragraphs = text.split("\n")
        all_lines = []
        paragraph_line_counts = []
        for paragraph in paragraphs:
            paragraph_lines = _wrap_paragraph(paragraph, font, wrap, usable_width)
            all_lines.extend(paragraph_lines)
            paragraph_line_counts.append(len(paragraph_lines))
        return font, all_lines, paragraph_line_counts

    lines = _wrap_paragraph(text, font, wrap, usable_width)
    return font, lines, [len(lines)]


def _measure_lines(font, lines, outline_size=0):
    bboxes = []
    widths = []
    heights = []

    for line in lines:
        bbox = _text_bbox(font, line, outline_size)
        line_width = max(0, bbox[2] - bbox[0])
        line_height = max(1, bbox[3] - bbox[1])
        bboxes.append(bbox)
        widths.append(line_width)
        heights.append(line_height)

    return bboxes, widths, heights


def _total_block_size(widths, heights, paragraph_line_counts=None, linespace=0, graphspace=0):
    max_width = max(widths) if widths else 0
    total_height = sum(heights)

    if heights:
        total_height += max(0, len(heights) - 1) * max(0, int(linespace))

    if paragraph_line_counts:
        paragraph_count = len(paragraph_line_counts)
        total_height += max(0, paragraph_count - 1) * max(0, int(graphspace))

    return max_width, total_height


def _layout_text(
    text,
    font_path,
    font_size,
    wrap,
    canvas_width,
    canvas_height,
    outline_size,
    margin_x,
    margin_y,
    multiline=False,
    linespace=0,
    graphspace=0,
):
    usable_width = max(1, int(canvas_width - margin_x * 2 - outline_size * 2))
    font, lines, paragraph_line_counts = _build_lines(
        text, font_path, font_size, wrap, usable_width, multiline=multiline
    )
    bboxes, line_widths, line_heights = _measure_lines(font, lines, outline_size)
    max_width, total_height = _total_block_size(
        line_widths,
        line_heights,
        paragraph_line_counts=paragraph_line_counts,
        linespace=linespace,
        graphspace=graphspace,
    )
    return {
        "font": font,
        "lines": lines,
        "paragraph_line_counts": paragraph_line_counts,
        "bboxes": bboxes,
        "line_widths": line_widths,
        "line_heights": line_heights,
        "max_width": max_width,
        "total_height": total_height,
        "usable_width": usable_width,
        "usable_height": max(1, int(canvas_height - margin_y * 2 - outline_size * 2)),
    }


def _fit_font_to_canvas(
    text,
    font_path,
    font_size,
    wrap,
    canvas_width,
    canvas_height,
    outline_size,
    margin_x,
    margin_y,
    auto_fit,
    multiline=False,
    linespace=0,
    graphspace=0,
):
    font_size = max(1, int(font_size))

    def make_layout(size):
        return _layout_text(
            text=text,
            font_path=font_path,
            font_size=size,
            wrap=wrap,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            outline_size=outline_size,
            margin_x=margin_x,
            margin_y=margin_y,
            multiline=multiline,
            linespace=linespace,
            graphspace=graphspace,
        )

    if not auto_fit:
        return make_layout(font_size)

    low = 1
    high = font_size
    best = make_layout(1)

    while low <= high:
        mid = (low + high) // 2
        layout = make_layout(mid)
        if (
            layout["max_width"] <= layout["usable_width"]
            and layout["total_height"] <= layout["usable_height"]
        ):
            best = layout
            low = mid + 1
        else:
            high = mid - 1

    return best


def _calc_text_x(align, img_width, margin_x, outline_size, usable_width, line_width, bbox):
    # bbox[0] may be negative in some fonts. Subtract it to avoid visual clipping.
    if align == "center":
        return margin_x + outline_size + (usable_width - line_width) // 2 - bbox[0]
    if align == "right":
        return img_width - margin_x - outline_size - line_width - bbox[0]
    return margin_x + outline_size - bbox[0]


def _draw_normal_lines(
    draw,
    lines,
    bboxes,
    line_widths,
    line_heights,
    font,
    img_width,
    margin_x,
    margin_y,
    outline_size,
    usable_width,
    align,
    color,
    outline_color,
    linespace=0,
    graphspace=0,
    paragraph_line_counts=None,
):
    y_text = margin_y + outline_size
    line_index = 0
    paragraph_line_counts = paragraph_line_counts or [len(lines)]

    for paragraph_index, paragraph_count in enumerate(paragraph_line_counts):
        for _ in range(paragraph_count):
            if line_index >= len(lines):
                break

            line = lines[line_index]
            bbox = bboxes[line_index]
            line_width = line_widths[line_index]
            line_height = line_heights[line_index]

            x_text = _calc_text_x(
                align=align,
                img_width=img_width,
                margin_x=margin_x,
                outline_size=outline_size,
                usable_width=usable_width,
                line_width=line_width,
                bbox=bbox,
            )

            draw.text(
                (int(x_text), int(y_text - bbox[1])),
                text=line,
                fill=color,
                stroke_fill=outline_color,
                stroke_width=outline_size,
                font=font,
            )
            y_text += line_height + max(0, int(linespace))
            line_index += 1

        # Remove the last line space before paragraph space, keeping legacy visual behavior close.
        if paragraph_index < len(paragraph_line_counts) - 1:
            y_text += max(0, int(graphspace))


# ----------------------------------------------------------------------------
class Text_Image_Zho_autofit:
    fonts = {}

    def __init__(self):
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = []
        for extension in font_extensions:
            fonts.extend(comfy_dir.glob(f"**/{extension}"))

        if not fonts:
            log.warning(
                "> No fonts found in the comfy folder, place at least one font file somewhere in ComfyUI's hierarchy"
            )
        else:
            log.debug(f"> Found {len(fonts)} fonts")

        for font in fonts:
            log.debug(f"Adding font {font}")
            cls.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            cls.CACHE_FONTS()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")

        return {
            "required": {
                "text": ("STRING", {"default": "ZHOZHOZHO"}),
                "selected_font": ((sorted(cls.fonts.keys())),),
                # Put center first so new nodes default to horizontal centering.
                "align": (["center", "left", "right"],),
                "wrap": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "font_size": ("INT", {"default": 12, "min": 1, "max": 2500, "step": 1}),
                "auto_fit": ("BOOLEAN", {"default": False}),
                "color": ("COLOR", {"default": "red"}),
                "outline_size": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "outline_color": ("COLOR", {"default": "blue"}),
                "margin_x": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "margin_y": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8096, "step": 1}),
                "swap": ("BOOLEAN", {"default": False}),
                "arc_text": ("BOOLEAN", {"default": False}),
                "arc_radius": ("INT", {"default": 100, "min": 1, "max": 2500, "step": 1}),
                "arc_start_angle": ("INT", {"default": 180, "min": 0, "max": 360, "step": 1}),
                "arc_end_angle": ("INT", {"default": 360, "min": 0, "max": 360, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image"
    CATEGORY = "Zho模块组/text"

    def draw_text_in_arc(
        self,
        image,
        draw,
        text,
        font,
        font_path,
        font_size,
        center,
        radius,
        start_angle,
        end_angle,
        fill="black",
        stroke_fill="blue",
        stroke_width=0,
    ):
        angle_range = end_angle - start_angle
        angle_step = angle_range / (len(text) - 1) if len(text) > 1 else 1
        current_angle = start_angle

        for char in text:
            # Pillow 10 removed getsize on some font objects; textbbox is safer.
            char_bbox = _text_bbox(font, char, stroke_width)
            char_width = max(1, char_bbox[2] - char_bbox[0])
            char_height = max(1, char_bbox[3] - char_bbox[1])
            angle = math.radians(current_angle)

            super_sampling_multiplier = 10
            char_image = Image.new(
                "RGBA",
                (char_width * super_sampling_multiplier, char_height * super_sampling_multiplier),
                (0, 0, 0, 0),
            )
            char_draw = ImageDraw.Draw(char_image)
            super_sampling_font = _safe_font(font_path, font_size * super_sampling_multiplier)
            char_draw.text(
                (0, 0),
                char,
                font=super_sampling_font,
                fill=fill,
                stroke_fill=stroke_fill,
                stroke_width=stroke_width,
            )

            rotate_angle = current_angle - 90 - (current_angle - 180) * 2
            rotated_char_image = char_image.rotate(rotate_angle, expand=1, resample=Image.BICUBIC)
            new_size = (
                max(1, int(rotated_char_image.width / super_sampling_multiplier)),
                max(1, int(rotated_char_image.height / super_sampling_multiplier)),
            )
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            rotated_char_image_resized = rotated_char_image.resize(new_size, resample_filter)

            x = center[0] + radius * math.cos(angle) - rotated_char_image_resized.size[0] / 2
            y = center[1] + radius * math.sin(angle) - rotated_char_image_resized.size[1] / 2
            image.paste(rotated_char_image_resized, (int(x), int(y)), rotated_char_image_resized)
            current_angle += angle_step

    def text_to_image(
        self,
        text,
        selected_font,
        align,
        wrap,
        font_size,
        width,
        height,
        color,
        outline_size,
        outline_color,
        margin_x,
        margin_y,
        auto_fit=False,
        swap=False,
        arc_text=False,
        arc_radius=100,
        arc_start_angle=180,
        arc_end_angle=360,
    ):
        if swap:
            width, height = swap_width_height(width, height)

        font_path = self.fonts[selected_font]
        img_width = int(width)
        img_height = int(height)
        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Arc text keeps legacy behavior. auto_fit is only applied to normal rectangular layout.
        if arc_text:
            font = _safe_font(font_path, font_size)
            _, text_height = bbox_dim(font.getbbox(text))
            center_x = img_width // 2
            center_y = arc_radius + text_height
            if align == "left":
                center_x = arc_radius + text_height // 2
            elif align == "right":
                center_x = img_width - arc_radius - text_height // 2
            center = (center_x + margin_x, center_y + margin_y)
            self.draw_text_in_arc(
                img,
                draw,
                text,
                font,
                font_path,
                font_size,
                center,
                arc_radius,
                arc_start_angle,
                arc_end_angle,
                fill=color,
                stroke_fill=outline_color,
                stroke_width=outline_size,
            )
        else:
            layout = _fit_font_to_canvas(
                text=text,
                font_path=font_path,
                font_size=font_size,
                wrap=wrap,
                canvas_width=img_width,
                canvas_height=img_height,
                outline_size=outline_size,
                margin_x=margin_x,
                margin_y=margin_y,
                auto_fit=auto_fit,
                multiline=False,
                linespace=0,
                graphspace=0,
            )

            _draw_normal_lines(
                draw=draw,
                lines=layout["lines"],
                bboxes=layout["bboxes"],
                line_widths=layout["line_widths"],
                line_heights=layout["line_heights"],
                font=layout["font"],
                img_width=img_width,
                margin_x=margin_x,
                margin_y=margin_y,
                outline_size=outline_size,
                usable_width=layout["usable_width"],
                align=align,
                color=color,
                outline_color=outline_color,
                linespace=0,
                graphspace=0,
                paragraph_line_counts=layout["paragraph_line_counts"],
            )

        return (pil2tensor(img),)


# ----------------------------------------------------------------------------
class Text_Image_Multiline_Zho_autofit:
    fonts = {}

    def __init__(self):
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = []
        for extension in font_extensions:
            fonts.extend(comfy_dir.glob(f"**/{extension}"))

        if not fonts:
            log.warning(
                "> No fonts found in the comfy folder, place at least one font file somewhere in ComfyUI's hierarchy"
            )
        else:
            log.debug(f"> Found {len(fonts)} fonts")

        for font in fonts:
            log.debug(f"Adding font {font}")
            cls.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            cls.CACHE_FONTS()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")

        return {
            "required": {
                "text": ("STRING", {"default": "ZHOZHOZHO", "multiline": True}),
                "selected_font": ((sorted(cls.fonts.keys())),),
                # Put center first so new nodes default to horizontal centering.
                "align": (["center", "left", "right"],),
                "wrap": ("INT", {"default": 120, "min": 0, "max": 8096, "step": 1}),
                "graphspace": ("INT", {"default": 10, "min": 0, "max": 8096, "step": 1}),
                "linespace": ("INT", {"default": 2, "min": 0, "max": 8096, "step": 1}),
                "font_size": ("INT", {"default": 12, "min": 1, "max": 2500, "step": 1}),
                "auto_fit": ("BOOLEAN", {"default": False}),
                "color": ("COLOR", {"default": "red"}),
                "outline_size": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "outline_color": ("COLOR", {"default": "blue"}),
                "margin_x": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "margin_y": ("INT", {"default": 0, "min": 0, "max": 8096, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8096, "step": 1}),
                "swap": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image_multiline"
    CATEGORY = "Zho模块组/text"

    def text_to_image_multiline(
        self,
        text,
        selected_font,
        align,
        wrap,
        graphspace,
        linespace,
        font_size,
        width,
        height,
        color,
        outline_size,
        outline_color,
        margin_x,
        margin_y,
        auto_fit=False,
        swap=False,
    ):
        if swap:
            width, height = swap_width_height(width, height)

        font_path = self.fonts[selected_font]
        img_width = int(width)
        img_height = int(height)
        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        layout = _fit_font_to_canvas(
            text=text,
            font_path=font_path,
            font_size=font_size,
            wrap=wrap,
            canvas_width=img_width,
            canvas_height=img_height,
            outline_size=outline_size,
            margin_x=margin_x,
            margin_y=margin_y,
            auto_fit=auto_fit,
            multiline=True,
            linespace=linespace,
            graphspace=graphspace,
        )

        _draw_normal_lines(
            draw=draw,
            lines=layout["lines"],
            bboxes=layout["bboxes"],
            line_widths=layout["line_widths"],
            line_heights=layout["line_heights"],
            font=layout["font"],
            img_width=img_width,
            margin_x=margin_x,
            margin_y=margin_y,
            outline_size=outline_size,
            usable_width=layout["usable_width"],
            align=align,
            color=color,
            outline_color=outline_color,
            linespace=linespace,
            graphspace=graphspace,
            paragraph_line_counts=layout["paragraph_line_counts"],
        )

        return (pil2tensor(img),)


NODE_CLASS_MAPPINGS = {
    "Text_Image_Multiline_Zho_autofit": Text_Image_Multiline_Zho_autofit,
    "Text_Image_Zho_autofit": Text_Image_Zho_autofit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Text_Image_Multiline_Zho_autofit": "Text Image Multiline Zho AutoFit",
    "Text_Image_Zho_autofit": "Text Image Zho AutoFit",
}
