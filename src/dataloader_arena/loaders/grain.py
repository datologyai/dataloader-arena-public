"""Grain loader (IterDataset-based) with configurable sources and pipeline."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

from pydantic import Field

from dataloader_arena.config.models import ConfigModel
from dataloader_arena.context import LoaderConfig
from dataloader_arena.loaders.utils import ensure_local_cache_dir, ensure_mapping

__all__ = ["GrainConfig", "create_loader", "build_loader"]

import grain


class FactoryIterDataset(grain.IterDataset):
    """Sliceable source dataset for Grain mp_prefetch.

    Grain mp_prefetch needs to 'slice' the upstream IterDataset per worker process.
    We implement set_slice() and forward the worker_index/worker_count into the iterator
    (primarily for deterministic seeding / debug; sharding is handled by the source
    library via rank/world_size).
    """

    def __init__(
        self, factory: Callable[[], Iterable[Any]], label: str, *, base_seed: int = 0
    ):
        super().__init__()
        self._factory = factory
        self._label = label
        self._base_seed = int(base_seed)

        self._worker_index = 0
        self._worker_count = 1

    def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
        self._worker_index = int(sl.start or 0)
        self._worker_count = int(sl.step or 1)

    def __iter__(self) -> Any:
        return FactoryIterator(
            self._factory,
            worker_index=self._worker_index,
            worker_count=self._worker_count,
            base_seed=self._base_seed,
        )

    def __repr__(self) -> str:
        return f"{self._label}IterDataset"


class FactoryIterator(grain.DatasetIterator):
    """Checkpointable iterator that constructs a fresh iterable per iterator."""

    def __init__(
        self,
        factory: Callable[[], Iterable[Any]],
        *,
        worker_index: int,
        worker_count: int,
        base_seed: int = 0,
    ):
        super().__init__()
        self._factory = factory
        self._worker_index = int(worker_index)
        self._worker_count = int(worker_count)
        self._base_seed = int(base_seed)

        self._position = 0
        self._reset()

    def _reset(self) -> None:
        seed = self._base_seed + self._worker_index
        try:
            import random

            random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

        self._iterable = self._factory()
        self._iterator = iter(self._iterable)

    def __next__(self) -> Any:
        item = next(self._iterator)
        self._position += 1
        return item

    def get_state(self) -> dict[str, int]:
        return {"position": self._position}

    def set_state(self, state: Mapping[str, int]) -> None:
        target = int(state.get("position", 0))
        self._reset()
        for _ in range(target):
            next(self._iterator)
        self._position = target


# -------------------------------------------------------------------------
# CONFIG MODELS
# -------------------------------------------------------------------------


class GrainSourceConfig(ConfigModel):
    type: Literal["litdata"] = "litdata"
    options: Mapping[str, Any] = Field(default_factory=dict)


class GrainPrefetchConfig(ConfigModel):
    mode: Literal["thread", "process", "disabled"] | None = None
    buffer_size: int = Field(0, ge=0)
    enable_profiling: bool = False
    sequential_slice: bool = False


class GrainPipelineConfig(ConfigModel):
    seed: int | None = None
    shuffle_window: int | None = None
    shuffle_seed: int | None = None
    repeat: int | None = None
    cache: bool = False
    limit: int | None = None
    batch_size: int | None = None
    drop_remainder: bool = False
    prefetch: GrainPrefetchConfig | None = None
    dataset_options: Mapping[str, Any] | None = None
    disable_shm_passthrough: bool = False
    use_torch_default_collate: bool = False


_DISABLE_SHM_MIN_SIZE = sys.maxsize


def _torch_default_collate_batch(values: Sequence[Any]) -> Any:
    """Batch using PyTorch ``default_collate`` instead of Grain's numpy stacking."""
    from torch.utils.data._utils.collate import default_collate

    return default_collate(list(values))


class GrainConfig(ConfigModel):
    source: GrainSourceConfig = Field(default_factory=GrainSourceConfig)
    pipeline: GrainPipelineConfig = Field(default_factory=GrainPipelineConfig)
    vlm: Mapping[str, Any] | None = None  # For VLM benchmarks; ignored by base loader
    packing: Mapping[str, Any] | None = (
        None  # For VLM packing config; ignored by base loader
    )


# -------------------------------------------------------------------------
# UTILS & WORKER INIT
# -------------------------------------------------------------------------


@contextlib.contextmanager
def _suppress_fd(fd: int):
    """Suppress a POSIX file descriptor (e.g., 1=stdout)."""
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


@dataclass(slots=True)
class _StreamingSourceFactory:
    module: str
    class_name: str
    options: dict[str, Any]

    def __call__(self) -> Iterable[Any]:
        mod = importlib.import_module(self.module)
        cls = getattr(mod, self.class_name)
        return cls(**self.options)


@dataclass(frozen=True, slots=True)
class GrainRankWorkerInit:
    """Assign distributed ranks so LitData partitions shards across Grain workers."""

    init_file: str
    backend: str = "gloo"
    quiet_gloo: bool = True
    quiet_gloo_nonzero_only: bool = True
    quiet_absl: bool = True
    torch_cpp_log_level: str | None = "ERROR"
    init_timeout_s: int = 600

    def __call__(self, worker_index: int, worker_count: int) -> None:
        from datetime import timedelta

        os.environ["RANK"] = str(worker_index)
        os.environ["WORLD_SIZE"] = str(worker_count)
        os.environ["LOCAL_RANK"] = str(worker_index)
        os.environ["LOCAL_WORLD_SIZE"] = str(worker_count)

        if self.quiet_absl:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["ABSL_MIN_LOG_LEVEL"] = "error"

        if self.quiet_gloo:
            os.environ["GLOO_LOG_LEVEL"] = "ERROR"
        if self.torch_cpp_log_level is not None:
            os.environ.setdefault("TORCH_CPP_LOG_LEVEL", self.torch_cpp_log_level)

        if self.quiet_absl:
            try:
                from absl import logging as absl_logging

                absl_logging.set_verbosity(absl_logging.ERROR)
            except Exception:
                pass

        try:
            import torch.utils.data._utils.worker as torch_worker

            torch_worker._worker_info = None
        except Exception:
            pass

        try:
            import torch.distributed as dist
        except Exception as e:
            raise RuntimeError(
                "torch.distributed is required for rank-style Grain workers"
            ) from e

        suppress = self.quiet_gloo and (
            not self.quiet_gloo_nonzero_only or worker_index != 0
        )

        cm = _suppress_fd(1) if suppress else contextlib.nullcontext()
        if not dist.is_initialized():
            with cm:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=f"file://{self.init_file}",
                    rank=int(worker_index),
                    world_size=int(worker_count),
                    timeout=timedelta(seconds=self.init_timeout_s),
                )


def make_grain_rank_worker_init() -> GrainRankWorkerInit:
    import tempfile

    fd, path = tempfile.mkstemp(prefix="grain_rank_pg_", suffix=".tmp")
    os.close(fd)
    os.unlink(path)  # FileStore requires the file to NOT exist before init
    return GrainRankWorkerInit(init_file=path)


# -------------------------------------------------------------------------
# BUILDERS
# -------------------------------------------------------------------------


def _build_litdata_iter_dataset(
    opts: Mapping[str, Any], runtime_cfg: LoaderConfig
) -> tuple[Any, dict[str, Any]]:
    from dataloader_arena.loaders import litdata as litdata_loader

    ensure_local_cache_dir(
        runtime_cfg.local_path, runtime_cfg.remote_path, loader_name="LitData"
    )
    nested_cfg = LoaderConfig(
        loader_name="grain-litdata-source", run=runtime_cfg.run, config=opts
    )
    lit_cfg, dataset_opts, dataloader_opts = litdata_loader.materialize_config(
        nested_cfg
    )

    factory = _StreamingSourceFactory("litdata", "StreamingDataset", dataset_opts)
    ds = FactoryIterDataset(factory, "LitData")
    resolved = {
        "type": "litdata",
        "config": lit_cfg.model_dump(),
        "dataset_options": dataset_opts,
        "dataloader_options": dataloader_opts,
    }
    return ds, resolved


def _make_source_iter_dataset(
    source_cfg: GrainSourceConfig, runtime_cfg: LoaderConfig
) -> tuple[Any, dict[str, Any]]:
    src_type = source_cfg.type.lower()
    opts = ensure_mapping(source_cfg.options, label="source.options")

    if src_type == "litdata":
        return _build_litdata_iter_dataset(opts, runtime_cfg)

    raise ValueError(f"Unsupported Grain source type: {source_cfg.type}")


def _apply_prefetch(
    ds: Any,
    cfg: GrainPrefetchConfig | None,
    *,
    runtime_num_workers: int | None,
) -> tuple[Any, dict[str, Any] | None]:
    if cfg is None or cfg.mode is None or cfg.mode == "disabled":
        return ds, None

    if cfg.mode == "thread":
        ds = grain.experimental.ThreadPrefetchIterDataset(
            ds, prefetch_buffer_size=cfg.buffer_size
        )
        return ds, {"mode": "thread", "buffer_size": cfg.buffer_size}

    if cfg.mode == "process":
        worker_count = runtime_num_workers
        if worker_count is None or worker_count <= 0:
            warnings.warn(
                "Process prefetch requires num_workers > 0; skipping prefetch (set via CLI flag).",
                RuntimeWarning,
            )
            return ds, None

        rank_init = make_grain_rank_worker_init()

        mp_options = grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=cfg.buffer_size,
            enable_profiling=cfg.enable_profiling,
        )
        ds = ds.mp_prefetch(
            mp_options,
            worker_init_fn=rank_init,
            sequential_slice=cfg.sequential_slice,
        )
        return ds, {
            "mode": "process",
            "num_workers": worker_count,
            "per_worker_buffer_size": cfg.buffer_size,
            "enable_profiling": cfg.enable_profiling,
            "sequential_slice": cfg.sequential_slice,
        }

    raise ValueError(f"Unknown prefetch mode: {cfg.mode}")


def _apply_pipeline(
    ds: Any,
    cfg: GrainPipelineConfig,
    *,
    default_batch_size: int | None,
    runtime_num_workers: int | None,
) -> tuple[Any, dict[str, Any]]:
    pipeline_resolved: dict[str, Any] = {"config": cfg.model_dump()}
    batch_size = cfg.batch_size if cfg.batch_size is not None else default_batch_size
    if cfg.seed is not None:
        ds = ds.seed(cfg.seed)
    if cfg.shuffle_window:
        shuffle_seed = cfg.shuffle_seed if cfg.shuffle_seed is not None else cfg.seed
        ds = grain.experimental.WindowShuffleIterDataset(
            ds, window_size=cfg.shuffle_window, seed=shuffle_seed or 0
        )
    if cfg.repeat is not None:
        ds = grain.experimental.RepeatIterDataset(ds, num_epochs=cfg.repeat)
    if cfg.cache:
        ds = grain.experimental.CacheIterDataset(ds)
    if cfg.limit is not None:
        ds = grain.experimental.LimitIterDataset(ds, count=cfg.limit)
    batch_fn: Callable[[Sequence[Any]], Any] | None = None
    batch_fn_name = "grain_stack"
    if cfg.use_torch_default_collate:
        batch_fn = _torch_default_collate_batch
        batch_fn_name = "torch_default_collate"
    if batch_size is not None:
        batch_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "drop_remainder": cfg.drop_remainder,
        }
        if batch_fn is not None:
            batch_kwargs["batch_fn"] = batch_fn
        ds = ds.batch(**batch_kwargs)
        pipeline_resolved["batch_fn"] = batch_fn_name
    pipeline_resolved["batch_size"] = batch_size

    options_dict = (
        dict(ensure_mapping(cfg.dataset_options, label="pipeline.dataset_options"))
        if cfg.dataset_options
        else {}
    )
    if cfg.disable_shm_passthrough:
        options_dict["min_shm_size"] = _DISABLE_SHM_MIN_SIZE
    if options_dict:
        dataset_options = grain.experimental.DatasetOptions(**options_dict)
        ds = grain.experimental.WithOptionsIterDataset(ds, dataset_options)
        pipeline_resolved["dataset_options"] = options_dict

    prefetch_cfg = cfg.prefetch
    if prefetch_cfg is None and (runtime_num_workers or 0) > 0:
        prefetch_cfg = GrainPrefetchConfig(mode="process", buffer_size=1)

    ds, prefetch_info = _apply_prefetch(
        ds, prefetch_cfg, runtime_num_workers=runtime_num_workers
    )
    if prefetch_info:
        pipeline_resolved["prefetch"] = prefetch_info

    return ds, pipeline_resolved


def build_loader(
    config: LoaderConfig,
) -> Iterable[Any] | Iterator[Any]:
    loader_cfg = GrainConfig.parse(config.config)
    source_ds, source_resolved = _make_source_iter_dataset(loader_cfg.source, config)

    ds, pipeline_resolved = _apply_pipeline(
        source_ds,
        loader_cfg.pipeline,
        default_batch_size=config.batch_size,
        runtime_num_workers=config.num_workers,
    )

    ds.__arena_resolved_config__ = {  # type: ignore[attr-defined]
        "parsed": loader_cfg.model_dump(),
        "source": source_resolved,
        "pipeline": pipeline_resolved,
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
    return ds


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    return build_loader(config)
