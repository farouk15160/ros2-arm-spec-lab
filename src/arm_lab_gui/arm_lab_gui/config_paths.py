"""YAML path helpers shared by the editor and headless schema tests."""
from typing import Any


def get_path(data: Any, path: str, default=None) -> Any:
    node = data
    for key in path.split('.'):
        try:
            node = node[int(key)] if isinstance(node, list) else node[key]
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    return node


def set_path(data: Any, path: str, value: Any) -> None:
    keys = path.split('.')
    node = data
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
