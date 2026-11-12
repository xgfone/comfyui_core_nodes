# -*- coding: utf-8 -*-

import asyncio
import inspect
import logging
import time
from importlib import import_module
from pkgutil import iter_modules
from typing import Dict, List, Tuple

from . import py as _node_package

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load_node_mappings():
    """Load mappings from Python modules directly inside the ``py`` package."""
    modules = sorted(
        iter_modules(_node_package.__path__, f"{_node_package.__name__}."),
        key=lambda module_info: module_info.name,
    )

    for module_info in modules:
        # Only load py/*.py; do not recurse into child packages.
        if module_info.ispkg:
            continue

        module = import_module(module_info.name)

        class_mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
        if class_mappings is not None:
            NODE_CLASS_MAPPINGS.update(class_mappings)

        display_name_mappings = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
        if display_name_mappings is not None:
            NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)


_load_node_mappings()


LOGGER = logging.getLogger("node_time_logger")
COMFY_NODE_TIME_LOG_MIN_MS = 20  # 只打印运行时间高于阈值的节点
COMFY_NODE_TIME_LOG = 1  # 关闭整个功能
COMFY_NODE_TIME_LOG_SUMMARY = 1  # 关闭汇总（只打印逐节点阈值日志）
COMFY_NODE_TIME_LOG_TOPN = 20  # 汇总 TopN

# prompt_id -> list of (dt, node_id, display_id, class_type, cached, result_name)
_NODE_TIMES: Dict[str, List[Tuple[float, str, str, str, bool, str]]] = {}
_LOCK = asyncio.Lock()


def _enabled() -> bool:
    v = COMFY_NODE_TIME_LOG
    return v not in ("0", "false", "off", "no")


def _min_ms() -> float:
    # 逐节点日志阈值（ms），低于该值不打印逐节点行；汇总仍可包含
    try:
        return float(COMFY_NODE_TIME_LOG_MIN_MS)
    except Exception:
        return 0.0


def _summary_enabled() -> bool:
    v = COMFY_NODE_TIME_LOG_SUMMARY
    return v not in ("0", "false", "off", "no")


def _top_n() -> int:
    try:
        return int(COMFY_NODE_TIME_LOG_TOPN)
    except Exception:
        return 20


async def _clear_prompt(prompt_id: str):
    async with _LOCK:
        _NODE_TIMES[prompt_id] = []


async def _append_time(prompt_id: str, row):
    async with _LOCK:
        _NODE_TIMES.setdefault(prompt_id, []).append(row)


async def _dump_summary(prompt_id: str):
    async with _LOCK:
        rows = list(_NODE_TIMES.get(prompt_id, []))

    if not rows:
        LOGGER.info(f"[node_time] prompt={prompt_id} (no nodes recorded)")
        return

    total = sum(dt for dt, *_ in rows)
    rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
    n = min(_top_n(), len(rows_sorted))

    LOGGER.info(
        f"[node_time] prompt={prompt_id} total={total * 1000:.1f}ms nodes={len(rows_sorted)} top={n}"
    )
    for i in range(n):
        dt, node_id, display_id, class_type, cached, result_name = rows_sorted[i]
        LOGGER.info(
            f"[node_time]  #{i + 1:02d} {dt * 1000:8.1f}ms  node={node_id} display={display_id} "
            f"type={class_type} cached={1 if cached else 0} result={result_name}"
        )


def _install_patch():
    try:
        import execution  # ComfyUI 根目录 execution.py
    except Exception as e:
        LOGGER.exception(f"[node_time] failed to import execution.py: {e}")
        return

    if getattr(execution, "_node_time_logger_patched", False):
        return
    execution._node_time_logger_patched = True

    # 兜底：确保能看到日志（若你有自己的 logging 配置，这不会强行覆盖 handler）
    if LOGGER.level == logging.NOTSET:
        LOGGER.setLevel(logging.INFO)

    orig_execute = execution.execute
    sig_execute = inspect.signature(orig_execute)

    orig_execute_async = execution.PromptExecutor.execute_async
    sig_execute_async = inspect.signature(orig_execute_async)

    async def timed_execute(*args, **kwargs):
        """
        wrapper for: async def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id,
                                      execution_list, pending_subgraph_results, pending_async_nodes, ui_outputs)
        使用 *args/**kwargs 透传 + bind_partial 获取参数
        """
        if not _enabled():
            return await orig_execute(*args, **kwargs)

        t0 = time.perf_counter()
        ret = await orig_execute(*args, **kwargs)
        dt = time.perf_counter() - t0

        min_ms = _min_ms()
        need_summary = _summary_enabled()
        need_detail = dt * 1000.0 >= min_ms

        # 如果既不需要逐节点日志、也不需要汇总数据，就不做 bind_partial（省开销）
        if not need_detail and not need_summary:
            return ret

        try:
            bound = sig_execute.bind_partial(*args, **kwargs).arguments
        except TypeError:
            bound = {}

        dynprompt = bound.get("dynprompt")
        caches = bound.get("caches")
        current_item = bound.get("current_item")
        prompt_id = bound.get("prompt_id")

        # 没拿到关键字段：不影响执行，只降级输出
        if prompt_id is None or current_item is None or dynprompt is None:
            if need_detail:
                LOGGER.info(f"[node_time] dt={dt * 1000:.1f}ms (missing bindings)")
            return ret

        # 丰富信息
        try:
            node = dynprompt.get_node(current_item)
            class_type = str(node.get("class_type"))
            display_id = str(dynprompt.get_display_node_id(current_item))
        except Exception:
            class_type = "unknown"
            display_id = str(current_item)

        cached = False
        try:
            if caches is not None:
                cached = caches.outputs.get(current_item) is not None
        except Exception:
            cached = False

        result_name = getattr(ret, "name", type(ret).__name__)

        # 记录到汇总
        if need_summary:
            await _append_time(
                str(prompt_id),
                (
                    dt,
                    str(current_item),
                    str(display_id),
                    str(class_type),
                    bool(cached),
                    str(result_name),
                ),
            )

        # 逐节点打印（带阈值）
        if need_detail:
            LOGGER.info(
                f"[node_time] prompt={prompt_id} node={current_item} display={display_id} "
                f"type={class_type} dt={dt * 1000:.1f}ms cached={1 if cached else 0} result={result_name}"
            )

        return ret

    async def timed_execute_async(self, *args, **kwargs):
        """
        wrapper for: PromptExecutor.execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[])
        这里主要用于：每个工作流开始清空记录、结束打印汇总
        """
        if not _enabled() or not _summary_enabled():
            return await orig_execute_async(self, *args, **kwargs)

        # bind 得到 prompt_id
        try:
            bound = sig_execute_async.bind_partial(self, *args, **kwargs).arguments
        except TypeError:
            bound = {}

        prompt_id = bound.get("prompt_id")
        if prompt_id is None:
            # 拿不到 prompt_id，就退化成不汇总
            return await orig_execute_async(self, *args, **kwargs)

        await _clear_prompt(str(prompt_id))
        wall0 = time.perf_counter()
        try:
            return await orig_execute_async(self, *args, **kwargs)
        finally:
            wall = time.perf_counter() - wall0
            LOGGER.info(f"[node_time] prompt={prompt_id} workflow_wall={wall * 1000:.1f}ms")
            await _dump_summary(str(prompt_id))

    # 打补丁
    execution.execute = timed_execute
    execution.PromptExecutor.execute_async = timed_execute_async

    LOGGER.info(
        "[node_time] installed monkey patch (bind_partial): execution.execute + PromptExecutor.execute_async"
    )


_install_patch()
