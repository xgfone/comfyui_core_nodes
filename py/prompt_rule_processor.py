"""Rule-based prompt and image URL selection for ComfyUI."""

from __future__ import annotations

import re
from typing import Mapping, Pattern

# The optional backslash makes pasted forms such as ``\<main>`` work too.
_MAIN_RE = re.compile(r"\\?<main\s*>", re.IGNORECASE)
_POSES_RE = re.compile(r"\\?<poses\s*>", re.IGNORECASE)
_IMAGES_RE = re.compile(r"\\?<images\s*>", re.IGNORECASE)
_POSE_BLOCK_RE = re.compile(r"\\?<pose\s+count\s*=\s*[\"']?(\d+)[\"']?\s*>", re.IGNORECASE)
_IMAGE_GROUP_RE = re.compile(r"\\?<image_group\s+count\s*=\s*[\"']?(\d+)[\"']?\s*>", re.IGNORECASE)
_IMAGE_RE = re.compile(r"\\?<image\s*>", re.IGNORECASE)
_COUNT_RE = re.compile(r"\\?<count\s*>", re.IGNORECASE)
_POSE_PROMPT_RE = re.compile(r"\\?<pose_prompt\s*>", re.IGNORECASE)

_DIGITS = "零一二三四五六七八九"


def number_to_chinese(number: int) -> str:
    """Convert an integer from 0 through 9999 to common Chinese numerals."""

    number = int(number)
    if not 0 <= number <= 9999:
        raise ValueError("人数必须在 0 到 9999 之间")
    if number < 10:
        return _DIGITS[number]

    units = ("", "十", "百", "千")
    digits = [int(char) for char in str(number)]
    highest_position = len(digits) - 1
    result: list[str] = []
    zero_pending = False

    for index, digit in enumerate(digits):
        position = highest_position - index
        if digit == 0:
            if result and any(remaining != 0 for remaining in digits[index + 1 :]):
                zero_pending = True
            continue
        if zero_pending:
            result.append("零")
            zero_pending = False
        # 10-19 are normally written 十、十一…… instead of 一十、一十一……
        if not (digit == 1 and position == 1 and not result):
            result.append(_DIGITS[digit])
        result.append(units[position])

    return "".join(result)


def _section_after(
    source: str,
    start_pattern: Pattern[str],
    end_pattern: Pattern[str] | None,
) -> str:
    """Return text following a section marker up to the next section marker."""

    start = start_pattern.search(source)
    if not start:
        return ""
    end = end_pattern.search(source, start.end()) if end_pattern else None
    return source[start.end() : end.start() if end else len(source)]


def _parse_count_blocks(section: str, marker: Pattern[str]) -> dict[int, str]:
    """Parse repeated ``count`` markers into an explicit count-to-text map."""

    parts = marker.split(section)
    blocks: dict[int, str] = {}
    for index in range(1, len(parts), 2):
        count = int(parts[index])
        if 1 <= count <= 9999:
            blocks[count] = parts[index + 1].strip()
    return blocks


def _select_with_fallback(items: Mapping[int, str], people_count: int) -> str:
    """Select the greatest non-empty configured count not above the request."""

    eligible = [count for count, text in items.items() if count <= people_count and text.strip()]
    return items[max(eligible)].strip() if eligible else ""


def _format_image_group(image_group: str) -> str:
    """Trim image URL entries and join them with the public <image> delimiter."""

    entries = [entry.strip() for entry in _IMAGE_RE.split(image_group) if entry.strip()]
    return "\n<image>\n".join(entries)


def process_prompt(prompt: str, people_count: int = 1) -> tuple[str, str]:
    """Process the marked source string and return prompt plus image URL text."""

    prompt = "" if prompt is None else str(prompt)
    people_count = 1 if people_count is None else int(people_count)
    if not 1 <= people_count <= 9999:
        raise ValueError("人数必须在 1 到 9999 之间")

    main_prompt = _section_after(prompt, _MAIN_RE, _POSES_RE).strip()
    pose_section = _section_after(prompt, _POSES_RE, _IMAGES_RE)
    image_section = _section_after(prompt, _IMAGES_RE, None)

    pose_options = _parse_count_blocks(pose_section, _POSE_BLOCK_RE)
    image_options = _parse_count_blocks(image_section, _IMAGE_GROUP_RE)
    selected_pose = _select_with_fallback(pose_options, people_count)
    selected_images = _select_with_fallback(image_options, people_count)

    # Replace pose first, then count globally, so <count> inside a selected pose
    # is replaced as well.
    result_prompt = _POSE_PROMPT_RE.sub(lambda _: selected_pose, main_prompt)
    result_prompt = _COUNT_RE.sub(lambda _: number_to_chinese(people_count), result_prompt)

    return result_prompt, _format_image_group(selected_images)


class PromptRuleProcessor:
    """ComfyUI node wrapper around :func:`process_prompt`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "输入包含 <main>、<poses>、<images> 等标签的提示词",
                    },
                ),
            },
            "optional": {
                "people_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "image_urls")
    FUNCTION = "process"
    CATEGORY = "文本/提示词处理"
    DESCRIPTION = "按显式人数配置选择姿势提示词和图片 URL，并替换提示词占位符。"

    def process(self, prompt: str, people_count: int = 1):
        return process_prompt(prompt, people_count)


NODE_CLASS_MAPPINGS = {
    "PromptRuleProcessor": PromptRuleProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRuleProcessor": "提示词规则处理器(适用于api调用)",
}
