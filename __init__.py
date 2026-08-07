# -*- coding: utf-8 -*-

from importlib import import_module
from pkgutil import iter_modules

from . import py as _node_package

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


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
