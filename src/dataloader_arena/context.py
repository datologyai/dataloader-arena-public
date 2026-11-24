"""Shared dataclasses and helpers for loader factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataloader_arena.config.models import ConfigModel


def _ensure_mapping(
    value: Mapping[str, Any] | None, *, label: str
) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ConfigModel):
        return value.model_dump()
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, received {type(value)!r}")
    return dict(value)


@dataclass(slots=True, frozen=True)
class LoaderPlan:
    """Loader definition loaded from configuration files."""

    name: str
    factory: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "config", _ensure_mapping(self.config, label="Loader config")
        )


@dataclass(slots=True, frozen=True)
class RunParameters:
    """Shared runtime parameters supplied to every loader."""

    local_path: Path
    remote_path: str | None
    batch_size: int | None
    num_workers: int
    seed: int | None = None
    limit_steps: int | None = None
    verbose: bool = False


@dataclass(slots=True, frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration constructed from CLI arguments."""

    run: RunParameters
    warmup_steps: int
    measure_steps: int
    loader_plans: Sequence[LoaderPlan] = field(default_factory=tuple)


@dataclass(slots=True)
class LoaderConfig:
    """Configuration passed to loader factory functions."""

    loader_name: str
    run: RunParameters
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config = _ensure_mapping(
            self.config, label=f"Loader '{self.loader_name}' config"
        )

    @property
    def local_path(self) -> Path:
        return self.run.local_path

    @property
    def remote_path(self) -> str | None:
        return self.run.remote_path

    @property
    def batch_size(self) -> int | None:
        return self.run.batch_size

    @property
    def num_workers(self) -> int:
        return self.run.num_workers

    @property
    def seed(self) -> int | None:
        return self.run.seed

    @property
    def limit_steps(self) -> int | None:
        return self.run.limit_steps

    @property
    def verbose(self) -> bool:
        return self.run.verbose
