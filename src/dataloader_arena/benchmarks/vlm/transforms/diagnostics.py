"""
Image pipeline diagnostic logging.

Enable with ARENA_VLM_IMAGE_DIAG=1 to trace where images are lost
in the Zephon and Grain pipelines. All counters are plain dicts
for cloudpickle safety (Zephon MTP mode).
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_env = os.environ.get("ARENA_VLM_IMAGE_DIAG", "")
IMAGE_DIAG_ENABLED: bool = _env.strip().lower() in {"1", "true", "yes"}

DIAG_SUMMARY_INTERVAL: int = int(os.environ.get("ARENA_VLM_IMAGE_DIAG_INTERVAL", "256"))


def diag_log(stage: str, path: str, **kwargs: Any) -> None:
    """Emit one structured diagnostic log line.

    Format: [IMAGE_DIAG] stage=X path=Y key1=val1 key2=val2

    Uses logger.warning() because this codebase has no logging handlers
    configured, and the last-resort handler only prints WARNING+.
    logger.info() was silently swallowed.
    """
    if not IMAGE_DIAG_ENABLED:
        return
    parts = [f"stage={stage}", f"path={path}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    logger.warning("[IMAGE_DIAG] " + " ".join(parts))


def make_diag_counters() -> dict:
    """Create a fresh counter dict. Pickle-safe (plain dict)."""
    return {
        "total": 0,
        "with_images": 0,
        "no_images": 0,
        "raw_images": 0,
        "after_filter": 0,
        "dropped_align": 0,
        "dropped_maxlen": 0,
        "dropped_parse": 0,
    }
