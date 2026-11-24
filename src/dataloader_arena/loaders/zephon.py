"""Shared Zephon loader setup and configuration helpers."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping as AbcMapping
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Mapping

from pydantic import ConfigDict, Field, field_validator, model_validator

from dataloader_arena.config.models import ConfigModel

from ..context import LoaderConfig
from .utils import ensure_local_cache_dir

if TYPE_CHECKING:
    import torch
    from zephon import Pipeline
    from zephon.observability import MetricsSinkConfig
__all__ = [
    "ZephonConfig",
    "build_zephon_pipeline",
    "materialize_config",
    "build_pipeline_with_dataset",
    "create_base_loader",
]


def _zephon_collate(batch: list[Any]) -> Any:
    """Pass through SampleBatch objects so callers can inspect Zephon outputs."""
    return batch


def _jsonify(o):
    if callable(o):
        return getattr(o, "__qualname__", getattr(o, "__name__", str(o)))
    return str(o)


def _coerce_bool(value: Any) -> bool:
    """Best-effort conversion of loose truthy/falsey values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _stringify_tags(raw: Mapping[str, object] | None) -> dict[str, str]:
    """Normalize user-supplied tags to string pairs."""
    if not isinstance(raw, Mapping):
        return {}
    tags: dict[str, str] = {}
    for key, value in raw.items():
        if key is None or value is None:
            continue
        name = str(key)
        if isinstance(value, bool):
            tags[name] = "true" if value else "false"
        else:
            tags[name] = str(value)
    return tags


class _InitLogger:
    """Lightweight timer for logging initialization milestones."""

    def __init__(self, enabled: bool, loader_name: str) -> None:
        self.enabled = enabled
        self.loader_name = loader_name
        self._start = time.perf_counter()
        self._last = self._start
        if self.enabled:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[INFO] {self.loader_name} init start @ {ts}", file=sys.stderr)

    def mark(self, step: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        delta_ms = (now - self._last) * 1000.0
        total_ms = (now - self._start) * 1000.0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(
            f"[INFO] {self.loader_name} init {step} @ {ts} "
            f"(+{delta_ms:.1f}ms total={total_ms:.1f}ms)",
            file=sys.stderr,
        )
        self._last = now


class ZephonWorkSourceConfig(ConfigModel):
    """Typed work source configuration for Zephon."""

    seed: int | None = 0
    chunk_size: int | None = None
    shuffle_shards: bool = False
    shuffle_within_shard: bool = False
    shuffle_block_size: int | None = None
    exhausted_policy: str = "stop"


class ZephonObservabilityConfig(ConfigModel):
    """Typed observability configuration for Zephon."""

    enabled: bool = False
    tracking: str = "nodes"
    sink: Mapping[str, object] | str | None = None
    tags: Mapping[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        populate_by_name=True,
        frozen=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(
        cls, raw: Mapping[str, object] | str | bool | None
    ) -> Mapping[str, object] | None:
        if raw is None:
            return {}
        if isinstance(raw, bool):
            return {"enabled": raw}
        if isinstance(raw, str):
            return {"tracking": raw}
        return raw

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Mapping[str, object] | None) -> dict[str, str]:
        return _stringify_tags(value)


class ZephonPipelineConfig(ConfigModel):
    """Typed pipeline configuration for Zephon."""

    deterministic: bool = True
    prefetch_batches: int = 8
    default_stage_prefetch: int = 8
    op_queue_capacity: int = 64
    io_options: Mapping[str, object] = Field(default_factory=dict)

    # Allow passthrough extras to keep pipeline options extensible.
    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        populate_by_name=True,
        frozen=True,
    )

    def to_options(self) -> dict[str, object]:
        # model_dump includes extras when extra="allow"
        return self.model_dump()


class ZephonConfig(ConfigModel):
    """Structured configuration parsed from benchmark JSON."""

    name: str = "default"
    dataset_format: Literal["litdata"] | None = Field(default=None, alias="format")
    mixture: dict[str, float] = Field(default_factory=dict)
    work_source: ZephonWorkSourceConfig = Field(default_factory=ZephonWorkSourceConfig)
    pipeline: ZephonPipelineConfig = Field(default_factory=ZephonPipelineConfig)
    observability: ZephonObservabilityConfig = Field(
        default_factory=ZephonObservabilityConfig
    )
    experiment: str | None = None
    vlm: Mapping[str, object] | None = (
        None  # For VLM benchmarks; ignored by base loader
    )
    vlm_zephon: Mapping[str, object] | None = (
        None  # For VLM Zephon benchmarks; ignored by base loader
    )
    packing: Mapping[str, object] | None = (
        None  # For VLM packing config; ignored by base loader
    )
    prefetch_buffer_size: int = (
        0  # 0 = disabled; >0 = enable shard prefetch with this lookahead
    )
    prefetch_parallelism: int = 4
    fetch_parallelism: int | None = 4
    batch_parallelism: int = 1
    drop_last: bool = False
    stateful_dataset: bool = False
    skip_loader: bool = False
    runner: str = "threads"

    @field_validator("mixture", mode="before")
    @classmethod
    def _default_mixture(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _validate_parallelisms(self) -> "ZephonConfig":
        if self.batch_parallelism <= 0:
            raise ValueError("zephon batch_parallelism must be a positive integer.")
        if self.fetch_parallelism is not None and self.fetch_parallelism <= 0:
            raise ValueError("zephon fetch_parallelism must be a positive integer.")
        return self


def materialize_config(config: LoaderConfig) -> tuple[ZephonConfig, dict[str, Any]]:
    zephon_cfg = ZephonConfig.parse(config.config)
    resolved = {
        "parsed": zephon_cfg.model_dump(),
        "run": {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "limit_steps": config.limit_steps,
            "seed": config.seed,
            "verbose": config.verbose,
        },
        "paths": {
            "local": str(config.local_path),
            "remote": config.remote_path,
        },
    }
    return zephon_cfg, resolved


def build_zephon_pipeline(
    config: LoaderConfig,
    zephon_cfg: ZephonConfig,
    resolved: dict[str, Any],
    *,
    loader_name: str = "zephon",
    init_logger: _InitLogger | None = None,
) -> tuple["Pipeline", dict[str, Any]]:
    """Build a Zephon pipeline from configuration.

    This helper encapsulates the common pipeline building logic used by both
    fetch-and-batch and full VLM workloads.

    Args:
        config: Loader configuration
        zephon_cfg: Parsed Zephon configuration
        resolved: Resolved config dictionary (will be updated)
        loader_name: Name for observability tags (default: "zephon")
        init_logger: Optional timing logger for initialization steps

    Returns:
        Tuple of (Pipeline, updated_resolved_config)
    """
    from zephon import Pipeline
    from zephon.io.dataset import Dataset
    from zephon.observability import ExecutionTrackingMode
    from zephon.work import StaticMixtureWorkSource

    local_path = config.local_path
    remote_path = config.remote_path
    ensure_local_cache_dir(local_path, remote_path, loader_name="Zephon")
    dataset_root = remote_path if remote_path else str(local_path)

    dataset_name = zephon_cfg.name
    dataset_fmt = zephon_cfg.dataset_format
    dataset = Dataset.from_path(dataset_name, dataset_root, fmt=dataset_fmt)
    resolved["dataset"] = {
        "name": dataset_name,
        "format": dataset_fmt,
        "root": dataset_root,
    }
    if init_logger:
        init_logger.mark("Dataset.from_path")

    work_cfg = zephon_cfg.work_source
    seed = config.seed if config.seed is not None else work_cfg.seed
    if seed is None:
        seed = 0

    default_chunk_size = (
        config.batch_size
        if isinstance(config.batch_size, int) and config.batch_size > 0
        else 1
    )
    shards_meta = (
        dataset.backend.get("shards")
        if isinstance(dataset.backend, AbcMapping)
        else None
    )
    if isinstance(shards_meta, AbcMapping):
        for shard in shards_meta.values():
            chunk_meta = {}
            if isinstance(shard, AbcMapping):
                chunk_meta = shard.get("chunk") or shard.get("config") or {}
            candidate = (
                chunk_meta.get("chunk_size")
                if isinstance(chunk_meta, AbcMapping)
                else None
            )
            if isinstance(candidate, int) and candidate > 0:
                default_chunk_size = candidate
                break

    chunk_size = (
        work_cfg.chunk_size if work_cfg.chunk_size is not None else default_chunk_size
    )
    shuffle_shards = work_cfg.shuffle_shards
    shuffle_within_shard = work_cfg.shuffle_within_shard
    shuffle_block_size = work_cfg.shuffle_block_size
    exhausted_policy = work_cfg.exhausted_policy

    mixture = dict(zephon_cfg.mixture) if zephon_cfg.mixture else {dataset_name: 1.0}
    resolved["work_source_options"] = {
        "mixture": mixture,
        "chunk_size": chunk_size,
        "seed": seed,
        "shuffle_shards": shuffle_shards,
        "shuffle_within_shard": shuffle_within_shard,
        "shuffle_block_size": shuffle_block_size,
        "exhausted_policy": exhausted_policy,
    }

    work_source = StaticMixtureWorkSource(
        [dataset],
        mixture=mixture,
        chunk_size=chunk_size,
        seed=seed,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within_shard,
        shuffle_block_size=shuffle_block_size,
        exhausted_policy=exhausted_policy,
    )
    if init_logger:
        init_logger.mark("StaticMixtureWorkSource")

    if config.verbose:
        print(
            "[INFO] Zephon Derived WorkSource Settings\n"
            f"chunk_size = {chunk_size} seed = {seed} shuffle_shards = {shuffle_shards} "
            f"shuffle_within_shard = {shuffle_within_shard} shuffle_block_size = {shuffle_block_size} "
            f"exhausted_policy = {exhausted_policy}\n"
            f"mixture = {json.dumps(mixture, indent=4)}",
            file=sys.stderr,
        )

    pipeline = Pipeline(work_source)
    if init_logger:
        init_logger.mark("Pipeline(work_source)")

    if zephon_cfg.prefetch_buffer_size > 0:
        pipeline.prefetch(
            buffer_size=zephon_cfg.prefetch_buffer_size,
            parallelism=zephon_cfg.prefetch_parallelism,
        )
        resolved["prefetch"] = {
            "buffer_size": zephon_cfg.prefetch_buffer_size,
            "parallelism": zephon_cfg.prefetch_parallelism,
        }
        if init_logger:
            init_logger.mark("pipeline.prefetch")

    resolved["fetch_parallelism"] = zephon_cfg.fetch_parallelism
    if zephon_cfg.fetch_parallelism is not None:
        pipeline.fetch(zephon_cfg.fetch_parallelism)

    observability_cfg = zephon_cfg.observability
    resolved["observability"] = {
        "enabled": observability_cfg.enabled,
        "tracking": observability_cfg.tracking,
        "sink": observability_cfg.sink,
        "tags": observability_cfg.tags,
    }
    if observability_cfg.enabled:
        tracking_mode = ExecutionTrackingMode.from_value(observability_cfg.tracking)
        base_tags = {
            "loader": loader_name,
            "dataset": dataset_name,
            "remote": "true" if bool(remote_path) else "false",
            "batch_size": "none"
            if config.batch_size is None
            else str(config.batch_size),
            "num_workers": str(config.num_workers),
        }
        experiment = zephon_cfg.experiment
        if experiment is not None:
            base_tags["experiment"] = experiment
        base_tags.update(observability_cfg.tags)
        sink_config = _build_sink_config(observability_cfg.sink, base_tags=base_tags)
        pipeline.enable_observability(tracking_mode, sink=sink_config)
        resolved["observability"].update(  # type: ignore[call-arg]
            {
                "tracking_mode": tracking_mode,
                "base_tags": base_tags,
                "sink_config": sink_config,
            }
        )
        if config.verbose:
            print(
                f"[INFO] Zephon Derived Observability Settings\nbase_tags = {json.dumps(base_tags, indent=4)}",
                file=sys.stderr,
            )
        if init_logger:
            init_logger.mark("pipeline.enable_observability")

    pipeline_opts = zephon_cfg.pipeline.to_options()
    resolved["pipeline_options"] = pipeline_opts
    if remote_path:
        io_opts = dict(pipeline_opts.get("io_options", {}))
        cache_opts = dict(io_opts.get("cache", {}))
        if "enabled" in cache_opts and not cache_opts["enabled"]:
            raise RuntimeError(
                "Cache needs to be enabled when using a remote path, but was explicitly disabled."
            )
        cache_opts["enabled"] = True
        cache_opts.setdefault("root", str(local_path))
        io_opts["cache"] = cache_opts
        pipeline_opts["io_options"] = io_opts

    cache_opts = pipeline_opts.get("io_options", {}).get("cache", {})
    if cache_opts.get("enabled", False):
        cache_root = cache_opts["root"]  # guaranteed to exist if enabled
        if cache_root == remote_path:
            raise RuntimeError(
                "Cache root directory cannot be equivalent to remote path."
            )

    pipeline_opts["runner"] = zephon_cfg.runner

    from dataclasses import fields

    from zephon.options import RuntimeOptions

    unknown = set(pipeline_opts) - {field.name for field in fields(RuntimeOptions)}
    if unknown:
        raise ValueError(f"Unsupported Zephon runtime options: {sorted(unknown)}")

    if config.verbose:
        print(
            f"[INFO] Zephon Runtime Settings\n{json.dumps(pipeline_opts, default=_jsonify, indent=4)}",
            file=sys.stderr,
        )

    if pipeline_opts:
        pipeline.options(**pipeline_opts)
        if init_logger:
            init_logger.mark("pipeline.options")

    return pipeline, resolved


def build_pipeline_with_dataset(
    config: LoaderConfig,
    pipeline_modifier: Callable[["Pipeline", ZephonConfig, dict[str, Any]], None]
    | None = None,
    *,
    loader_name: str = "Zephon",
    init_logger: _InitLogger | None = None,
) -> tuple[
    "ZephonConfig",
    "Pipeline",
    dict[str, Any],
    "torch.utils.data.Dataset | None",
    dict[str, Any] | None,
]:
    """Prepare a Zephon pipeline and (optionally) a torch Dataset for loaders.

    pipeline_modifier allows callers (e.g., benchmarks) to inject additional
    pipeline steps such as tokenization without duplicating setup logic. The
    loader_name/init_logger hooks let us emit timing logs per loader.
    """
    if init_logger is None and config.verbose:
        init_logger = _InitLogger(True, loader_name)

    zephon_cfg, resolved = materialize_config(config)
    if init_logger:
        init_logger.mark("materialize_config")
    pipeline, resolved = build_zephon_pipeline(
        config,
        zephon_cfg,
        resolved,
        loader_name=loader_name,
        init_logger=init_logger,
    )

    if pipeline_modifier:
        pipeline_modifier(pipeline, zephon_cfg, resolved)
        if init_logger:
            init_logger.mark("pipeline_modifier")

    if config.batch_size is not None:
        if zephon_cfg.batch_parallelism != 1:
            raise ValueError(
                f"batch parallelism is set to {zephon_cfg.batch_parallelism}. Only 1 is supported ATM."
            )
        pipeline.batch(config.batch_size, drop_last=zephon_cfg.drop_last)
        if init_logger:
            init_logger.mark("pipeline.batch")

    torch_dataset = None
    dataloader_opts = None
    if not zephon_cfg.skip_loader:
        torch_dataset = pipeline.to_torch_dataset(stateful=zephon_cfg.stateful_dataset)
        if init_logger:
            init_logger.mark("pipeline.to_torch_dataset")
        dataloader_opts = {
            "batch_size": None,
            "num_workers": config.num_workers,
            "pin_memory": False,
            "collate_fn": _zephon_collate,
        }
        resolved["dataloader_options"] = dataloader_opts

    return zephon_cfg, pipeline, resolved, torch_dataset, dataloader_opts


def finalize_zephon_loader(
    config: LoaderConfig,
    zephon_cfg: ZephonConfig,
    pipeline: "Pipeline",
    resolved: dict[str, Any],
    torch_dataset: "torch.utils.data.Dataset | None",
    dataloader_opts: dict[str, Any] | None,
    *,
    loader_name: str = "Zephon",
    print_config: bool = True,
    print_pipeline_plan: bool = True,
    init_logger: _InitLogger | None = None,
) -> Iterable[Any] | Iterator[Any]:
    """Shared finalization: optionally print info, build DataLoader, honor skip_loader.

    init_logger continues timing logs across the finalization steps.
    """
    from torch.utils.data import DataLoader

    if init_logger is None and config.verbose:
        init_logger = _InitLogger(True, loader_name)

    if init_logger:
        init_logger.mark("finalize.start")

    if config.verbose and print_config:
        print(
            f"[INFO] {loader_name} Full LoaderConfig\n{json.dumps(asdict(config), default=_jsonify, indent=4)}",
            file=sys.stderr,
        )

    if config.verbose and print_pipeline_plan:
        print(
            f"[INFO] {loader_name} Pipeline Plan\n{pipeline.explain()}", file=sys.stderr
        )
        if init_logger:
            init_logger.mark("pipeline.explain")

    resolved["pipeline_plan"] = pipeline.explain()

    if not zephon_cfg.skip_loader:
        assert torch_dataset is not None and dataloader_opts is not None
        if config.verbose:
            printable_opts = dict(dataloader_opts)
            printable_opts["collate_fn"] = _jsonify(printable_opts["collate_fn"])
            print(
                f"[INFO] DataLoader opts for {loader_name}\n{json.dumps(printable_opts, default=_jsonify, indent=4)}",
                file=sys.stderr,
            )

        dataloader = DataLoader(torch_dataset, **dataloader_opts)
        if init_logger:
            init_logger.mark("torch.DataLoader ready")
        dataloader.__arena_resolved_config__ = resolved  # type: ignore[attr-defined]
        if init_logger:
            init_logger.mark("init complete")
        return dataloader

    if config.verbose:
        print(
            f"[INFO] Skipping torch.DL for {loader_name}!",
            file=sys.stderr,
        )
    pipeline.__arena_resolved_config__ = resolved  # type: ignore[attr-defined]
    if init_logger:
        init_logger.mark("init complete (no torch DataLoader)")
    return pipeline


def _build_sink_config(
    sink_options: Mapping[str, object] | str | None,
    *,
    base_tags: Mapping[str, str],
) -> "MetricsSinkConfig":
    """Build a MetricsSinkConfig from loose user inputs plus default tags."""

    from zephon.observability import MetricsSinkConfig, MetricsSinkMode

    if isinstance(sink_options, MetricsSinkConfig):
        # Preserve caller configuration while layering base tags.
        return sink_options.with_additional_tags(base_tags)

    if isinstance(sink_options, str):
        sink_options = {"mode": sink_options}

    kwargs: dict[str, Any] = {}
    tags = dict(base_tags)
    if isinstance(sink_options, Mapping):
        mode = sink_options.get("mode")
        if mode is not None:
            kwargs["mode"] = MetricsSinkMode.from_value(mode)
        namespace = sink_options.get("namespace")
        if namespace:
            kwargs["namespace"] = str(namespace)
        statsd_host = sink_options.get("statsd_host")
        if statsd_host:
            kwargs["statsd_host"] = str(statsd_host)
        if "statsd_port" in sink_options and sink_options["statsd_port"] is not None:
            kwargs["statsd_port"] = int(sink_options["statsd_port"])
        if "json_logs" in sink_options:
            kwargs["json_logs"] = _coerce_bool(sink_options["json_logs"])
        if (
            "max_batch_size" in sink_options
            and sink_options["max_batch_size"] is not None
        ):
            kwargs["max_batch_size"] = int(sink_options["max_batch_size"])
        if (
            "flush_interval_s" in sink_options
            and sink_options["flush_interval_s"] is not None
        ):
            kwargs["flush_interval_s"] = float(sink_options["flush_interval_s"])
        tags.update(_stringify_tags(sink_options.get("default_tags")))
        tags.update(_stringify_tags(sink_options.get("tags")))

    kwargs["default_tags"] = tags
    return MetricsSinkConfig(**kwargs)


def create_base_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Build the base Zephon loader (used by benchmarks as a shared foundation)."""

    loader_name = "Zephon"
    init_logger = _InitLogger(True, loader_name) if config.verbose else None
    zephon_cfg, pipeline, resolved, torch_dataset, dataloader_opts = (
        build_pipeline_with_dataset(
            config,
            loader_name=loader_name,
            init_logger=init_logger,
        )
    )

    return finalize_zephon_loader(
        config,
        zephon_cfg,
        pipeline,
        resolved,
        torch_dataset,
        dataloader_opts,
        loader_name=loader_name,
        print_config=True,
        print_pipeline_plan=True,
        init_logger=init_logger,
    )
