"""Shared LitData loader setup for streaming shards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Tuple

from pydantic import Field

from dataloader_arena.config.models import ConfigModel

from ..context import LoaderConfig
from .utils import ensure_local_cache_dir

if TYPE_CHECKING:
    from litdata import StreamingDataset

__all__ = ["LitDataConfig", "materialize_config", "build_dataset", "create_loader"]


class LitDataConfig(ConfigModel):
    """JSON-backed configuration for the LitData loader."""

    dataset_format: Literal["litdata"] | None = Field(default=None, alias="format")
    shuffle: bool = False
    seed: int | None = None
    prefetch_factor: int | None = None


def materialize_config(
    config: LoaderConfig,
) -> Tuple[LitDataConfig, dict[str, Any], dict[str, Any]]:
    loader_config = LitDataConfig.parse(config.config)
    dataset_opts: dict[str, Any] = {
        "shuffle": loader_config.shuffle,
    }
    seed_value = loader_config.seed if loader_config.seed is not None else config.seed
    if seed_value is not None:
        dataset_opts["seed"] = seed_value
    if config.remote_path:
        dataset_opts["input_dir"] = config.remote_path
        dataset_opts["cache_dir"] = str(config.local_path)
    else:
        dataset_opts["input_dir"] = str(config.local_path)

    dataloader_opts: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
    }
    if loader_config.prefetch_factor is not None and config.num_workers > 0:
        dataloader_opts["prefetch_factor"] = loader_config.prefetch_factor
    return loader_config, dataset_opts, dataloader_opts


def build_dataset(
    config: LoaderConfig,
) -> Tuple[LitDataConfig, StreamingDataset, dict[str, Any], dict[str, Any]]:
    """Prepare the LitData StreamingDataset and options."""
    ensure_local_cache_dir(config.local_path, config.remote_path, loader_name="LitData")
    loader_config, dataset_opts, dataloader_opts = materialize_config(config)

    from litdata import StreamingDataset

    dataset = StreamingDataset(**dataset_opts)
    return loader_config, dataset, dataset_opts, dataloader_opts


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Instantiate the LitData streaming dataloader for local shards."""

    from litdata import StreamingDataLoader

    loader_config, dataset, dataset_opts, dataloader_opts = build_dataset(config)

    dataloader = StreamingDataLoader(
        dataset,
        **dataloader_opts,
    )
    dataloader.__arena_resolved_config__ = {  # type: ignore[attr-defined]
        "parsed": loader_config.model_dump(),
        "dataset_options": dataset_opts,
        "dataloader_options": dataloader_opts,
        "run": {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "limit_steps": config.limit_steps,
            "verbose": config.verbose,
        },
        "paths": {
            "local": str(config.local_path),
            "remote": config.remote_path,
        },
    }
    return dataloader
