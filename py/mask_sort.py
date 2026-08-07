from __future__ import annotations

import math

import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _flatten_to_BHW(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    通用维度归一：
    - 接受任意 >=2 维的 mask，只要最后两维是 (H,W)
    - 统一 reshape 成 [N, H, W]，其中 N = 其它所有前置维度相乘
    兼容示例：
      [H,W] -> [1,H,W]
      [B,H,W] -> [B,H,W]
      [B,1,H,W] -> [B,H,W]
      [K,1,B,H,W] -> [K*1*B, H, W]
    """
    arr = _to_numpy(mask) if isinstance(mask, torch.Tensor) else mask
    if arr.ndim < 2:
        raise ValueError(f"Unsupported mask ndim: {arr.ndim}, expected >=2 with trailing (H,W).")
    H, W = arr.shape[-2], arr.shape[-1]
    front = int(np.prod(arr.shape[:-2])) if arr.ndim > 2 else 1
    arr = arr.reshape(front, H, W)
    return arr


def _mask_area(arr2d: np.ndarray, thresh: float = 0.5) -> int:
    return int((arr2d > thresh).sum())


def _mask_centroid(arr2d: np.ndarray, thresh: float = 0.5) -> tuple[float, float]:
    ys, xs = np.nonzero(arr2d > thresh)  # 2D 情况只返回 (y, x)
    if xs.size == 0:
        return (np.nan, np.nan)
    return (float(xs.mean()), float(ys.mean()))


class MaskSorter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "sort_by": (
                    ["area", "left_right", "top_bottom", "center"],
                    {"default": "area"},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("sorted_masks",)
    FUNCTION = "sort_masks"
    CATEGORY = "mask/arrange"

    def sort_masks(self, masks, sort_by: str = "area", threshold: float = 0.5):
        """
        输入:
          - masks: 任意维度的 float mask（0~1），只要最后两维是 (H,W) 即可：
                    [H,W], [B,H,W], [B,1,H,W], [K,1,B,H,W] 等
        输出:
          - 排序后的 batch mask，形状 [N,H,W]
        """
        device = masks.device if isinstance(masks, torch.Tensor) else "cpu"
        dtype = masks.dtype if isinstance(masks, torch.Tensor) else torch.float32

        # 统一展平成 [N,H,W]
        arr = _flatten_to_BHW(masks)
        N, H, W = arr.shape
        cx0, cy0 = (W / 2.0, H / 2.0)

        records = []
        for i in range(N):
            m = arr[i]
            area = _mask_area(m, threshold)
            cx, cy = _mask_centroid(m, threshold)

            if math.isnan(cx) or math.isnan(cy):
                dist_center = float("inf")
                cx_eff, cy_eff = float("inf"), float("inf")
            else:
                dist_center = math.hypot(cx - cx0, cy - cy0)
                cx_eff, cy_eff = cx, cy

            records.append(
                {
                    "idx": i,
                    "mask": m,
                    "area": area,
                    "cx": cx_eff,
                    "cy": cy_eff,
                    "dist_center": dist_center,
                }
            )

        if sort_by == "area":
            records.sort(key=lambda r: r["area"], reverse=True)  # 大->小
        elif sort_by == "left_right":
            records.sort(key=lambda r: r["cx"])  # 左->右
        elif sort_by == "top_bottom":
            records.sort(key=lambda r: r["cy"])  # 上->下
        elif sort_by == "center":
            records.sort(key=lambda r: r["dist_center"])  # 近->远
        else:
            raise ValueError(f"Unsupported sort_by: {sort_by}")

        sorted_np = [r["mask"].astype(np.float32) for r in records]
        sorted_tensor = torch.from_numpy(np.stack(sorted_np, axis=0)).to(device=device, dtype=dtype)
        return (sorted_tensor,)
