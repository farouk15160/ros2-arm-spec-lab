"""Small strict validation primitives for versioned configuration documents."""
import math
import re
from pathlib import Path

import yaml


def fields(value, required, optional=(), *, path):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f'{path}: expected mapping with string keys')
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f'{path}: missing {sorted(missing)}; unknown {sorted(unknown)}')
    return value


def number(value, path, *, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0):
        raise ValueError(f'{path}: expected finite {"positive " if positive else ""}number')
    return value


def vector(value, path, size=3):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'{path}: expected {size} numbers')
    return tuple(number(item, path) for item in value)


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path}: expected nonempty string')
    return value


def identifier(value, path):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', text(value, path)):
        raise ValueError(f'{path}: expected identifier [A-Za-z_][A-Za-z0-9_]*')
    return value


def choice(value, options, path):
    if not isinstance(value, str) or value not in options:
        raise ValueError(f'{path}: expected one of {options}')


def version(value, path):
    if type(value) is not int or value != 1:
        raise ValueError(f'{path}.schema_version: only version 1 is supported')


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(not isinstance(key, str) for key in keys):
            raise ValueError('YAML mapping keys must be strings')
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate YAML mapping key')
        return super().construct_mapping(node, deep=deep)


def read_yaml(path):
    """Safe YAML subset: reject aliases, duplicate keys and recursive documents."""
    path = Path(path)
    try:
        contents = path.read_text()
        if any(isinstance(token, yaml.tokens.AliasToken) for token in yaml.scan(contents)):
            raise ValueError('YAML aliases are not supported; use explicit values')
        return yaml.load(contents, Loader=_StrictLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
