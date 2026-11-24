"""Load-and-batch benchmark entrypoint for Grain."""

from typing import Any, Iterable, Iterator

from dataloader_arena.context import LoaderConfig
from dataloader_arena.loaders.grain import create_loader as create_base_loader

__all__ = ["create_loader"]


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Instantiate the Grain loader for benchmarks."""
    return create_base_loader(config)
