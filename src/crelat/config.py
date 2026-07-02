"""Strict YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import yaml


def load_config(
    path: Union[str, Path], *, allowed: Iterable[str], required: Iterable[str] = ()
) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration {source} must be a YAML mapping")
    allowed_keys = set(allowed)
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    return payload


def normalized_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True)
