"""Shared helpers for loader setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def ensure_mapping(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    """Validate a mapping-like value and return a plain dict."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, received {type(value)!r}")
    return dict(value)


def ensure_local_cache_dir(
    local_path: Path, remote_path: str | None, *, loader_name: str
) -> None:
    """Ensure the local/cache directory exists when using a remote source."""
    if remote_path:
        local_path.mkdir(parents=True, exist_ok=True)
    elif not local_path.exists():
        raise FileNotFoundError(f"{loader_name} local path not found: {local_path}")
