"""Load-and-batch benchmark entrypoint for Zephon."""

from typing import Any, Iterable, Iterator

from dataloader_arena.context import LoaderConfig
from dataloader_arena.loaders.zephon import (
    build_pipeline_with_dataset,
    create_base_loader,
)

__all__ = ["create_loader", "build_pipeline"]


def build_pipeline(config: LoaderConfig):
    """Expose the same fetch/batch graph to the checkpoint harness."""
    _, pipeline, resolved, _, _ = build_pipeline_with_dataset(config)
    return pipeline, resolved


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Thin wrapper to use the shared Zephon base loader for this benchmark."""
    return create_base_loader(config)
