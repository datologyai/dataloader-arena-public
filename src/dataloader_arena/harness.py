"""Shared configuration, loader lifecycle, reporting and runtime helpers."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import sysconfig
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar

import psutil
from pydantic import Field, field_validator

from .config.models import ConfigModel
from .context import LoaderConfig, LoaderPlan, RunParameters

ConfigFactory = Callable[[LoaderConfig], Iterable[Any] | Iterator[Any]]


def _warn_gil_disabled() -> None:
    import sysconfig

    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return  # not a free-threaded build — GIL is expected
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is not None and is_gil_enabled():
        print(
            "[WARN] GIL is enabled on a free-threaded build! "
            "Thread scaling will be severely limited. "
            "Set PYTHON_GIL=0 or run with -Xgil=0.",
            file=sys.stderr,
        )


_LOADERS_REQUIRE_BATCH: frozenset[str] = frozenset({"litdata"})


def log_childs():
    """
    Comprehensively logs all threads (Python & Native) and Subprocesses
    spawned by the current application.
    """
    current_pid = os.getpid()
    proc = psutil.Process(current_pid)

    print(f"\n{'=' * 60}")
    print(f" DIAGNOSTIC REPORT: PID {current_pid} ({proc.name()})")
    print(f" Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")

    # --- SECTION 1: THREADS ---
    # We compare Python's awareness vs the OS's reality
    python_threads = threading.enumerate()
    os_thread_count = proc.num_threads()
    hidden_threads = os_thread_count - len(python_threads)

    print(f"\n[ THREADS ] Total OS Threads: {os_thread_count}")

    if hidden_threads > 0:
        print(
            f"(!) Warning: {hidden_threads} native threads detected that are unknown to Python."
        )
        print(
            "    (These likely belong to C-extensions like numpy, scipy, or database drivers)"
        )

    thread_names = [t.name or "<unnamed>" for t in python_threads]
    thread_name_width = (
        max(len("Name"), *(len(name) for name in thread_names))
        if python_threads
        else len("Name")
    )
    thread_header_line = f"{'Name':<{thread_name_width}} | {'Python ID':<15} | {'Native ID':<15} | {'Daemon':<10} | {'Status'}"
    print(f"\n  {thread_header_line}")
    print(f"  {'-' * len(thread_header_line)}")

    # List Python Threads
    for t, name in zip(python_threads, thread_names):
        status = "Alive" if t.is_alive() else "Stopped"
        native_id_display = t.native_id if t.native_id is not None else "-"
        ident_display = t.ident if t.ident is not None else "-"
        print(
            f"  {name:<{thread_name_width}} | {ident_display:<15} | "
            f"{native_id_display:<15} | {str(t.daemon):<10} | {status}"
        )

    # Print stack traces for all non-main active threads to see what they're doing
    main_thread = threading.main_thread()
    current_frames = sys._current_frames()
    print("\n[ THREAD STACKS ] Non-main active threads:")
    stacks_emitted = False
    for t in python_threads:
        if t is main_thread or t.ident is None or not t.is_alive():
            continue
        frame = current_frames.get(t.ident)
        stack_lines = (
            traceback.format_stack(frame) if frame else ["<no stack available>\n"]
        )
        print(
            f"\n  Thread '{t.name or '<unnamed>'}' (id={t.ident}, native_id={t.native_id}) stack:"
        )
        for line in stack_lines:
            print(f"    {line.rstrip()}")
        stacks_emitted = True
    if not stacks_emitted:
        print("  (none)")

    # --- SECTION 2: SUBPROCESSES ---
    # recursive=True finds children of children (grand-children)
    children = proc.children(recursive=True)

    print(f"\n\n[ SUBPROCESSES ] Total Children: {len(children)}")

    if children:
        details: list[dict[str, str]] = []
        for child in children:
            try:
                with child.oneshot():
                    mem_info = child.memory_info()
                    mem_mb = f"{round(mem_info.rss / (1024 * 1024), 2)}"
                    cmdline = " ".join(child.cmdline())
                    display_name = cmdline if cmdline else child.name()
                    details.append(
                        {
                            "pid": str(child.pid),
                            "ppid": str(child.ppid()),
                            "status": child.status(),
                            "mem": mem_mb,
                            "name": display_name,
                        }
                    )
            except psutil.NoSuchProcess:
                details.append(
                    {
                        "pid": "???",
                        "ppid": "-",
                        "status": "Terminated",
                        "mem": "-",
                        "name": "Process died during check",
                    }
                )
            except psutil.AccessDenied:
                details.append(
                    {
                        "pid": str(child.pid),
                        "ppid": str(child.ppid()) if child.is_running() else "-",
                        "status": "Access Denied",
                        "mem": "-",
                        "name": "Access denied",
                    }
                )

        name_width = max(
            len("Name / Cmdline"), *(len(item["name"]) for item in details)
        )
        child_header_line = (
            f"{'PID':<8} | {'PPID':<8} | {'Status':<12} | {'Memory (MB)':<12} | "
            f"{'Name / Cmdline':<{name_width}}"
        )
        print(f"\n  {child_header_line}")
        print(f"  {'-' * len(child_header_line)}")

        for item in details:
            print(
                f"  {item['pid']:<8} | {item['ppid']:<8} | {item['status']:<12} | "
                f"{item['mem']:<12} | {item['name']:<{name_width}}"
            )

    print(f"{'=' * 60}\n")


class LoaderPlanModel(ConfigModel):
    """Pydantic-backed loader definition parsed from config files."""

    name: str
    factory: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("config", mode="before")
    @classmethod
    def _ensure_mapping(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Loader config must be a mapping, received {type(value)!r}"
            )
        return dict(value)


class BenchmarkFileConfig(ConfigModel):
    """Top-level benchmark file representation."""

    loaders: list[LoaderPlanModel] = Field(default_factory=list)


def _format_duration(seconds: float) -> str:
    seconds_int = max(int(seconds), 0)
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:02d}m{sec:02d}s"


def load_config(path: Path) -> list[LoaderPlan]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file {path} not found. Update --config or create the file."
        )
    raw = json.loads(path.read_text())
    parsed = BenchmarkFileConfig.model_validate(raw)
    if not parsed.loaders:
        raise ValueError(f"No loaders configured in {path}")
    return [
        LoaderPlan(
            name=item.name,
            factory=item.factory,
            config=item.config,
        )
        for item in parsed.loaders
    ]


def import_factory(path: str) -> ConfigFactory:
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"Factory path must include ':', got '{path}'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if factory is None:
        raise AttributeError(f"Factory '{attr}' not found in module '{module_name}'")
    return factory  # type: ignore[return-value]


def _shutdown_loader(loader: Any, iterator: Any = None) -> None:
    """Release iterators and their owners, including Torch worker processes."""
    seen: set[int] = set()
    for obj in (iterator, loader, getattr(loader, "dataset", None)):
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for method_name in ("close", "shutdown", "_shutdown_workers"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as error:
                    print(f"[WARN] Loader cleanup failed: {error}", file=sys.stderr)
                break


def _should_skip_loader(
    plan: LoaderPlan, run_params: RunParameters
) -> tuple[bool, str | None]:
    if run_params.batch_size is None and any(
        name in plan.factory.lower() for name in _LOADERS_REQUIRE_BATCH
    ):
        return True, "requires batch_size >= 1"
    return False, None


def _zephon_runner(plan: LoaderPlan) -> str | None:
    if "zephon" not in plan.factory.lower():
        return None
    runner = plan.config.get("runner", "threads")
    return runner.lower() if isinstance(runner, str) else None


def _is_zephon_process(plan: LoaderPlan) -> bool:
    runner = _zephon_runner(plan)
    return runner is not None and "process" in runner


def _prioritize_worker_combinations(
    combos: Sequence[tuple[int | None, int, int | None]],
) -> list[tuple[int | None, int, int | None]]:
    """
    Run num_workers>0 combinations first to avoid forking after main-process-only runs.
    """
    prioritized: list[tuple[int, int, tuple[int | None, int, int | None]]] = []
    for idx, combo in enumerate(combos):
        _, num_workers, _ = combo
        priority = 0 if num_workers and num_workers > 0 else 1
        prioritized.append((priority, idx, combo))
    prioritized.sort(key=lambda item: (item[0], item[1]))
    return [combo for _, _, combo in prioritized]


def _order_zephon_factories(
    factories: Sequence[tuple[LoaderPlan, ConfigFactory]],
) -> list[tuple[LoaderPlan, ConfigFactory]]:
    """Run Zephon thread and default runners last; otherwise keep the given order."""
    prioritized: list[tuple[int, int, tuple[LoaderPlan, ConfigFactory]]] = []
    for idx, pair in enumerate(factories):
        plan, _ = pair
        runner = _zephon_runner(plan)
        if runner is None and "zephon" in plan.factory.lower():
            priority = 1
        else:
            priority = 0 if runner is None or "process" in runner else 1
        prioritized.append((priority, idx, pair))
    prioritized.sort(key=lambda item: (item[0], item[1]))
    return [pair for _, _, pair in prioritized]


def resolve_local_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _stringify_config(config: Mapping[str, Any]) -> str:
    try:
        return json.dumps(config, sort_keys=True)
    except TypeError:
        return repr(dict(config))


def _optional(value: Any) -> Any:
    return "" if value is None else value


T = TypeVar("T")


def _normalize_multi_values(
    values: Sequence[Sequence[T]] | None, *, default: Sequence[T]
) -> list[T]:
    flattened: list[T] = []
    if values:
        for group in values:
            flattened.extend(group)
    return flattened or list(default)


def _batch_option(value: str) -> int | None:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid value for --batch-size: '{value}'"
        ) from err
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "--batch-size must be a positive integer or 'none'"
        )
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'") from err
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed


def _seed_option(value: str) -> int | None:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid value for --seed: '{value}'"
        ) from err


def _ensure_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ConfigModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{label} must be a mapping, received {type(value)!r}")


def _extract_resolved_config(
    loader: Iterable[Any] | Iterator[Any],
    factory: ConfigFactory,
    loader_cfg: LoaderConfig,
) -> Mapping[str, Any]:
    attr = getattr(loader, "__arena_resolved_config__", None)
    if attr is not None:
        if isinstance(attr, ConfigModel):
            return attr.model_dump()
        try:
            return _ensure_mapping(attr, label="Loader resolved config")
        except TypeError:
            pass

    config = loader_cfg.config
    if isinstance(config, ConfigModel):
        return config.model_dump()

    return dict(config)


def _collect_fieldnames(
    rows: Sequence[Mapping[str, Any]], base_fields: Sequence[str]
) -> list[str]:
    remaining: set[str] = set()
    for row in rows:
        remaining.update(row.keys())

    fieldnames: list[str] = []
    for base in base_fields:
        if base in remaining:
            fieldnames.append(base)
            remaining.remove(base)

    fieldnames.extend(sorted(remaining))
    return fieldnames


def _write_csv_output(
    path: Path, rows: Sequence[Mapping[str, Any]], *, base_fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_fieldnames(rows, base_fields) if rows else list(base_fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def json_value(value: Any) -> Any:
    """Represent resolved options as JSON without executable Python reprs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if callable(value):
        kind = type(value)
        return f"{getattr(value, '__module__', kind.__module__)}.{getattr(value, '__qualname__', kind.__qualname__)}"
    return str(value)


def environment_snapshot() -> dict[str, Any]:
    from .utils.environment import host_snapshot

    distributions = list(importlib.metadata.distributions())
    packages = {dist.metadata["Name"]: dist.version for dist in distributions}
    sources = {}
    for distribution in distributions:
        raw = distribution.read_text("direct_url.json")
        if raw:
            try:
                sources[distribution.metadata["Name"]] = json.loads(raw)
            except ValueError:
                sources[distribution.metadata["Name"]] = {
                    "error": "invalid direct_url.json"
                }
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": is_enabled() if is_enabled else True,
        "packages": dict(sorted(packages.items())),
        "package_sources": dict(sorted(sources.items())),
        "host": host_snapshot(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
                "PYTHON_GIL",
                "PYTHONHASHSEED",
                "OMP_PLACES",
                "OMP_PROC_BIND",
                "OMP_WAIT_POLICY",
                "GOMP_CPU_AFFINITY",
                "KMP_AFFINITY",
                "KMP_BLOCKTIME",
                "MKL_DYNAMIC",
                "RAYON_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "MALLOC_ARENA_MAX",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
                "AWS_PROFILE",
                "AWS_ENDPOINT_URL_S3",
            )
        },
    }


def _require_no_gil(environment: dict[str, Any]) -> None:
    if not environment["free_threaded_build"] or environment["gil_enabled"]:
        raise RuntimeError("Requires a free-threaded interpreter with GIL disabled")


def write_json_output(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a complete JSON document before replacing the destination."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
