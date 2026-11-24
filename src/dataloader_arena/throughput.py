"""Throughput measurement functions and CLI for data loading pipelines."""

import os
import sys

# Force single-threaded math libs before any other imports.  Without this,
# each multiprocess worker spawns its own full OpenMP/MKL thread pool, causing
# massive contention when multiple workers do CPU-heavy work (e.g. torchvision
# decode/transform).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import dataclasses
import math
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .context import BenchmarkConfig, LoaderConfig, LoaderPlan, RunParameters
from .harness import (
    ConfigFactory,
    _batch_option,
    _extract_resolved_config,
    _format_duration,
    _is_zephon_process,
    _normalize_multi_values,
    _optional,
    _order_zephon_factories,
    _positive_int,
    _prioritize_worker_combinations,
    _require_no_gil,
    _seed_option,
    _should_skip_loader,
    _shutdown_loader,
    _stringify_config,
    _warn_gil_disabled,
    _write_csv_output,
    environment_snapshot,
    import_factory,
    json_value,
    load_config,
    log_childs,
    resolve_local_path,
    write_json_output,
)
from .utils.cleanup import cleanup_between_runs, purge_directory_contents
from .utils.stable_context import SystemStabilizer, check_storage_path

_BASE_CSV_FIELDS: tuple[str, ...] = (
    "loader_name",
    "benchmark.warmup_steps",
    "benchmark.measure_steps",
    "result.setup_time_s",
    "result.steps",
    "result.processed",
    "result.total_time_s",
    "result.step_time_s",
    "result.it_per_second",
    "result.samples_per_second",
    "run.local_path",
    "run.remote_path",
    "run.batch_size",
    "run.num_workers",
    "run.seed",
    "run.limit_steps",
    "run.verbose",
    "config.plaintext",
    "config.resolved.plaintext",
)


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    steps: int
    processed: int
    total_time: float
    num_workers: int
    batch_size: int | None
    seed: int | None
    setup_time: float = 0.0
    # Packing stats (only populated when --packing-stats is enabled)
    total_usable_tokens: int | None = None
    total_capacity_tokens: int | None = None

    @property
    def step_time(self) -> float:
        return self.total_time / self.steps if self.steps else 0.0

    @property
    def it_per_second(self) -> float:
        return self.steps / self.total_time if self.total_time else float("inf")

    @property
    def samples_per_second(self) -> float | None:
        bs = self.batch_size if self.batch_size is not None else 1
        return self.it_per_second * bs

    @property
    def packing_efficiency(self) -> float | None:
        if self.total_usable_tokens is None or self.total_capacity_tokens is None:
            return None
        return (
            self.total_usable_tokens / self.total_capacity_tokens
            if self.total_capacity_tokens
            else 0.0
        )

    @property
    def usable_tokens_per_second(self) -> float | None:
        if self.total_usable_tokens is None:
            return None
        return self.total_usable_tokens / self.total_time if self.total_time else 0.0


def _consume(iterator: Iterator[Any], steps: int) -> int:
    """Advance an iterator by up to `steps`. Returns the number of items consumed."""
    if steps <= 0:
        return 0
    consumed = 0
    while consumed < steps:
        try:
            next(iterator)
        except StopIteration:
            break
        consumed += 1
    return consumed


def _measure_plain(iterator: Iterator[Any], steps: int) -> tuple[int, float]:
    """Measure iteration throughput. Returns (processed, total_time)."""
    processed = 0
    start = time.perf_counter()
    while processed < steps:
        try:
            next(iterator)
        except StopIteration:
            break
        processed += 1
    total_time = time.perf_counter() - start
    return processed, total_time


def _measure_with_packing(
    iterator: Iterator[Any],
    steps: int,
    max_length: int,
) -> tuple[int, float, int, int]:
    """Measure iteration throughput + packing stats.

    Returns (processed, total_time, total_usable_tokens, total_capacity_tokens).
    No per-iteration branching — this is a separate loop from _measure_plain.
    """
    processed = 0
    total_usable = 0
    total_capacity = 0
    start = time.perf_counter()
    while processed < steps:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        seq_lens = batch.get("sequence_lengths") if isinstance(batch, dict) else None
        if seq_lens is not None:
            total_usable += int(seq_lens.sum().item())
            total_capacity += len(seq_lens) * max_length
        processed += 1
    total_time = time.perf_counter() - start
    return processed, total_time, total_usable, total_capacity


def _resolve_max_length(resolved_config: Mapping[str, Any]) -> int | None:
    """Extract max_length from resolved config for packing stats."""
    vlm = resolved_config.get("vlm")
    if isinstance(vlm, Mapping):
        sl = vlm.get("sequence_length")
        if isinstance(sl, int) and sl > 0:
            return sl
    return None


def run_benchmark(
    factory: ConfigFactory,
    cfg: LoaderConfig,
    warmup_steps: int,
    measure_steps: int,
    *,
    packing_stats: bool = False,
) -> tuple[BenchmarkResult, Mapping[str, Any]]:
    """
    Create the loader once, warm it up, then measure on the same iterator.
    `limit_steps` (if provided) is treated as a total cap across warmup+measure.
    When `packing_stats` is True, also collects usable-token counts per batch.
    """
    # Instantiate the loader once (timed separately as setup latency)
    setup_start = time.perf_counter()
    loader = factory(cfg)
    iterator = None
    try:
        resolved_config = _extract_resolved_config(loader, factory, cfg)
        iterator = iter(loader)
        setup_time = time.perf_counter() - setup_start

        max_length: int | None = None
        if packing_stats:
            max_length = _resolve_max_length(resolved_config)
            if max_length is None:
                raise ValueError(
                    "--packing-stats requires max_length but could not find "
                    "vlm.sequence_length in the loader's resolved config."
                )

        # Compute warmup target respecting an overall cap if provided
        total_cap = cfg.limit_steps
        warmup_target = min(warmup_steps, total_cap) if total_cap else warmup_steps
        consumed_warmup = _consume(iterator, warmup_target)

        # Compute measurement target, ensuring we don't exceed the total cap
        if total_cap is not None:
            remaining = max(total_cap - consumed_warmup, 0)
            measure_target = min(measure_steps, remaining)
        else:
            measure_target = measure_steps

        total_usable: int | None = None
        total_capacity: int | None = None

        if measure_target > 0:
            if packing_stats:
                assert max_length is not None
                processed, total_time, total_usable, total_capacity = (
                    _measure_with_packing(
                        iterator,
                        measure_target,
                        max_length,
                    )
                )
            else:
                processed, total_time = _measure_plain(iterator, measure_target)
        else:
            processed = 0
            total_time = 0.0

    finally:
        _shutdown_loader(loader, iterator)

    result = BenchmarkResult(
        name="",
        steps=processed,
        processed=processed,
        total_time=total_time,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        setup_time=setup_time,
        total_usable_tokens=total_usable,
        total_capacity_tokens=total_capacity,
    )
    return result, resolved_config


def format_results(results: list[BenchmarkResult]) -> str:
    has_packing = any(r.total_usable_tokens is not None for r in results)
    header = (
        f"{'Loader':<20} "
        f"{'Batch':>7} "
        f"{'Workers':>9} "
        f"{'Seed':>7} "
        f"{'Steps':>7} "
        f"{'it/s':>10} "
        f"{'samples/s':>12} "
        f"{'step_time(s)':>15} "
        f"{'total_time(s)':>15} "
        f"{'setup(s)':>12}"
    )
    if has_packing:
        header += f" {'pack_eff':>10} {'usable_tok/s':>14}"
    lines = [header, "-" * len(header)]
    for result in results:
        batch_display = "-" if result.batch_size is None else str(result.batch_size)
        worker_display = str(result.num_workers)
        seed_display = "-" if result.seed is None else str(result.seed)
        samples = result.samples_per_second
        samples_display = "-" if samples is None else f"{samples:0.2f}"
        line = (
            f"{result.name:<20} "
            f"{batch_display:>7} "
            f"{worker_display:>9} "
            f"{seed_display:>7} "
            f"{result.steps:>7d} "
            f"{result.it_per_second:>10.2f} "
            f"{samples_display:>12} "
            f"{result.step_time:>15.5f} "
            f"{result.total_time:>15.5f} "
            f"{result.setup_time:>12.2f}"
        )
        if has_packing:
            eff = result.packing_efficiency
            utps = result.usable_tokens_per_second
            eff_display = "-" if eff is None else f"{eff:.2%}"
            utps_display = "-" if utps is None else f"{utps:.0f}"
            line += f" {eff_display:>10} {utps_display:>14}"
        lines.append(line)
    return "\n".join(lines)


def _scaled_steps(
    base_steps: int,
    batch_size: int | None,
    *,
    min_steps: int | None,
) -> int:
    """
    Adjust steps for the provided batch size and clamp to an optional minimum.

    When `base_steps` represents a sample target for batch_size=1, dividing by the
    batch size keeps the overall sample count stable. Ceil division preserves at least
    one iteration when `base_steps < batch_size`.
    """
    bs = batch_size if batch_size is not None else 1
    if bs <= 0:
        raise ValueError("batch_size must be >= 1 when provided")
    base = max(base_steps, 0)
    scaled = (base + bs - 1) // bs if base else 0
    if min_steps is not None:
        scaled = max(scaled, min_steps)
    return scaled


def _prepare_csv_row(
    result: BenchmarkResult,
    loader_cfg: LoaderConfig,
    benchmark_cfg: BenchmarkConfig,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    run = loader_cfg.run
    row: dict[str, Any] = {
        "loader_name": result.name,
        "benchmark.warmup_steps": benchmark_cfg.warmup_steps,
        "benchmark.measure_steps": benchmark_cfg.measure_steps,
        "result.setup_time_s": result.setup_time,
        "result.steps": result.steps,
        "result.processed": result.processed,
        "result.total_time_s": result.total_time,
        "result.step_time_s": result.step_time,
        "result.it_per_second": result.it_per_second,
        "result.samples_per_second": _optional(result.samples_per_second),
        "run.local_path": str(run.local_path),
        "run.remote_path": _optional(run.remote_path),
        "run.batch_size": _optional(run.batch_size),
        "run.num_workers": run.num_workers,
        "run.seed": _optional(run.seed),
        "run.limit_steps": _optional(run.limit_steps),
        "run.verbose": run.verbose,
        "config.plaintext": _stringify_config(loader_cfg.config),
        "config.resolved.plaintext": _stringify_config(resolved_config),
    }
    if result.total_usable_tokens is not None:
        row["result.total_usable_tokens"] = result.total_usable_tokens
        row["result.total_capacity_tokens"] = result.total_capacity_tokens
        row["result.packing_efficiency"] = _optional(result.packing_efficiency)
        row["result.usable_tokens_per_second"] = _optional(
            result.usable_tokens_per_second
        )
    return row


def app(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dataloader-bench",
        description="Benchmark data loading pipelines.",
    )
    parser.add_argument(
        "--local",
        dest="local",
        type=str,
        required=True,
        help="Path to the local dataset/cache directory.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=None,
        help="Optional remote dataset URI passed to loader factories (e.g. s3://bucket/path).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("examples/loaders.json")),
        help="Path to loader configuration file (JSON).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=512,
        help=(
            "Target sample count for the measurement phase (batch_size=1 equivalent). "
            "Divided by batch size to determine actual iterations."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=512,
        help=(
            "Target sample count for the warmup phase (batch_size=1 equivalent). "
            "Divided by batch size to determine actual iterations."
        ),
    )
    parser.add_argument(
        "--min-steps",
        type=_positive_int,
        default=None,
        help="Optional lower bound applied to the scaled measurement iterations.",
    )
    parser.add_argument(
        "--min-warmup",
        type=_positive_int,
        default=None,
        help="Optional lower bound applied to the scaled warmup iterations.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        type=_batch_option,
        nargs="+",
        action="append",
        default=None,
        metavar="INT|none",
        help=(
            "Global batch size passed to loader factories. May be supplied multiple times "
            "or with several values per flag (e.g. --batch-size 32 64). Use 'none' to disable batching. "
            "Defaults to 32."
        ),
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers_values",
        type=int,
        nargs="+",
        action="append",
        default=None,
        metavar="INT",
        help=(
            "Number of workers or parallelism level for each loader. May be supplied multiple "
            "times or with several values per flag. Defaults to 4."
        ),
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=_seed_option,
        nargs="+",
        action="append",
        default=None,
        metavar="SEED",
        help=(
            "Optional RNG seed forwarded to loader factories. May be supplied multiple times "
            "or with several values per flag. Use 'none' to include an unseeded run."
        ),
    )
    parser.add_argument(
        "--limit-steps",
        type=_positive_int,
        default=None,
        help="Cap warmup plus measured batches per loader.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-loader debug information.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show a progress bar on stderr (default).",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the progress indicator.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional path to write benchmark results as CSV with full configuration details.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite an existing CSV file instead of failing.",
    )
    parser.add_argument(
        "--packing-stats",
        action="store_true",
        default=False,
        help=(
            "Collect packing efficiency and usable-token throughput per measurement run. "
            "Requires the loader to emit batches with a 'sequence_lengths' tensor "
            "and the resolved config to contain vlm.sequence_length."
        ),
    )
    parser.add_argument(
        "--dont-cleanup-local",
        action="store_true",
        help="Do NOT purge the local cache directory before runs where both local and remote are set.",
    )
    parser.add_argument(
        "--only",
        dest="only_loaders",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            'Only benchmark loaders whose "name" appears in the config JSON. '
            "Repeat --only or use comma-separated names (e.g. --only a,b). "
            "Useful for one-loader-per-process reproducibility runs."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on skipped, failed, incomplete or empty measurements. Without it, the run fails only when no loader succeeds.",
    )
    parser.add_argument(
        "--require-no-gil",
        action="store_true",
        help="Require a free-threaded interpreter with the GIL disabled.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write results, resolved options, environment and failures as JSON.",
    )
    parser.add_argument(
        "--no-system-tuning",
        action="store_true",
        help="Skip host tuning and OS page-cache drops",
    )
    args = parser.parse_args(argv)
    _warn_gil_disabled()
    if args.strict and (args.steps <= 0 or args.warmup < 0):
        parser.error("--strict requires positive --steps and nonnegative --warmup")
    # Refuse accidental replacement before doing any expensive work.
    for output in (
        args.json_output,
        Path(args.csv_output) if args.csv_output else None,
    ):
        if output and output.expanduser().exists() and not args.force_overwrite:
            parser.error(
                f"Output exists: {output}; use --force-overwrite to replace it"
            )
    failures: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    local_path = resolve_local_path(args.local)
    check_storage_path(local_path)
    config_path = Path(args.config)
    specs = load_config(config_path)
    if args.only_loaders:
        allowed: set[str] = set()
        for group in args.only_loaders:
            for part in group.split(","):
                name = part.strip()
                if name:
                    allowed.add(name)
        config_names = {p.name for p in specs}
        unknown = sorted(allowed - config_names)
        if unknown:
            parser.error(
                f"--only: loader name(s) not in {config_path}: {unknown}. "
                f"Known names: {sorted(config_names)}"
            )
        specs = [p for p in specs if p.name in allowed]
        if not specs:
            parser.error("--only: no loaders left to run")

    batch_sizes = _normalize_multi_values(args.batch_sizes, default=(32,))
    worker_counts = _normalize_multi_values(args.num_workers_values, default=(4,))
    seed_values = _normalize_multi_values(args.seeds, default=(None,))
    packing_stats = bool(args.packing_stats)
    progress_enabled = bool(args.progress)

    loader_plans = tuple(specs)
    resolved_factories: list[tuple[LoaderPlan, ConfigFactory]] = []
    for plan in loader_plans:
        try:
            factory = import_factory(plan.factory)
        except (ImportError, AttributeError, ValueError) as err:
            failures.append(
                {"loader_name": plan.name, "error": str(err), "phase": "import"}
            )
            print(f"[WARN] Skipping loader '{plan.name}': {err}", file=sys.stderr)
            continue
        resolved_factories.append((plan, factory))
    zephon_process_factories: list[tuple[LoaderPlan, ConfigFactory]] = []
    regular_factories: list[tuple[LoaderPlan, ConfigFactory]] = []
    for pair in resolved_factories:
        plan = pair[0]
        if _is_zephon_process(plan):
            zephon_process_factories.append(pair)
        else:
            regular_factories.append(pair)
    regular_factories = _order_zephon_factories(regular_factories)
    # Start process runners before threaded workloads initialize runtime state.
    ordered_factories: list[tuple[LoaderPlan, ConfigFactory]] = [
        *zephon_process_factories,
        *regular_factories,
    ]
    active_loader_plans: tuple[LoaderPlan, ...] = tuple(
        plan for plan, _ in ordered_factories
    )
    results: list[BenchmarkResult] = []
    csv_rows: list[dict[str, Any]] = []
    if ordered_factories:
        with nullcontext() if args.no_system_tuning else SystemStabilizer():
            combinations = list(product(batch_sizes, worker_counts, seed_values))
            combinations = _prioritize_worker_combinations(combinations)
            total_combinations = len(combinations)
            total_runs = total_combinations * len(ordered_factories)
            group_totals = {
                "zephon-process": len(zephon_process_factories) * total_combinations,
                "regular": len(regular_factories) * total_combinations,
            }
            overall_completed = 0
            group_completed: dict[str, int] = {name: 0 for name in group_totals}
            overall_start = time.perf_counter()
            group_start: dict[str, float] = {}

            def _emit(message: str) -> None:
                print(message, file=sys.stderr)

            def _advance(group: str, status: str) -> None:
                nonlocal overall_completed
                now = time.perf_counter()
                if not progress_enabled or total_runs <= 0:
                    return
                overall_completed = min(overall_completed + 1, total_runs)
                group_completed[group] = min(
                    group_completed.get(group, 0) + 1, group_totals.get(group, 0)
                )
                if group not in group_start:
                    group_start[group] = now

                overall_elapsed = now - overall_start
                overall_remaining = max(total_runs - overall_completed, 0)
                overall_eta = (
                    (overall_elapsed / overall_completed * overall_remaining)
                    if overall_completed
                    else None
                )

                group_total = group_totals.get(group, 0)
                group_done = group_completed.get(group, 0)
                g_elapsed = now - group_start.get(group, overall_start)
                g_remaining = max(group_total - group_done, 0)
                group_eta = (
                    (g_elapsed / group_done * g_remaining) if group_done else None
                )

                overall_eta_display = (
                    "--:--" if overall_eta is None else _format_duration(overall_eta)
                )
                group_eta_display = (
                    "--:--" if group_eta is None else _format_duration(group_eta)
                )
                status_clean = status.replace("\n", " ").strip()
                print(
                    f"[PROGRESS] overall {overall_completed}/{total_runs} "
                    f"elapsed {_format_duration(overall_elapsed)} "
                    f"eta {overall_eta_display} | "
                    f"{group} {group_done}/{group_total} eta {group_eta_display} | "
                    f"{status_clean}",
                    file=sys.stderr,
                )

            def _run_factory_group(
                factories: Sequence[tuple[LoaderPlan, ConfigFactory]], group_name: str
            ) -> None:
                if not factories:
                    return
                if group_name not in group_start:
                    group_start[group_name] = time.perf_counter()
                for combo_index, (batch_size, num_workers, seed) in enumerate(
                    combinations, start=1
                ):
                    batch_display = "none" if batch_size is None else str(batch_size)
                    seed_display = "none" if seed is None else str(seed)
                    combo_label = (
                        f"combo {combo_index}/{total_combinations} "
                        f"(batch={batch_display}, workers={num_workers}, seed={seed_display})"
                    )
                    warmup_steps = _scaled_steps(
                        args.warmup,
                        batch_size,
                        min_steps=args.min_warmup,
                    )
                    measure_steps = _scaled_steps(
                        args.steps,
                        batch_size,
                        min_steps=args.min_steps,
                    )
                    if args.limit_steps is not None:
                        warmup_steps = min(warmup_steps, args.limit_steps)
                        measure_steps = min(
                            measure_steps, max(args.limit_steps - warmup_steps, 0)
                        )
                    run_params = RunParameters(
                        local_path=local_path,
                        remote_path=args.remote,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        seed=seed,
                        limit_steps=args.limit_steps,
                        verbose=args.verbose,
                    )
                    benchmark_cfg = BenchmarkConfig(
                        run=run_params,
                        warmup_steps=warmup_steps,
                        measure_steps=measure_steps,
                        loader_plans=active_loader_plans,
                    )
                    if benchmark_cfg.run.verbose:
                        _emit(
                            f"[INFO] Running combination {combo_index}/{total_combinations}: "
                            f"batch_size={batch_display}, num_workers={num_workers}, seed={seed_display}, "
                            f"warmup_steps={warmup_steps}, measure_steps={measure_steps}"
                        )
                    for plan, factory in factories:
                        status_prefix = f"{plan.name}: {combo_label}"
                        should_skip, reason = _should_skip_loader(plan, run_params)
                        if should_skip:
                            failures.append(
                                {
                                    "loader_name": plan.name,
                                    "error": reason or "Skipped combination",
                                    "batch_size": batch_size,
                                    "num_workers": num_workers,
                                    "seed": seed,
                                }
                            )
                            if run_params.verbose:
                                detail = (
                                    " for batch_size=none"
                                    if run_params.batch_size is None
                                    else ""
                                )
                                reason_suffix = f": {reason}" if reason else ""
                                _emit(
                                    f"[INFO] Skipping loader '{plan.name}'{detail}{reason_suffix}"
                                )
                            _advance(group_name, f"{status_prefix} skipped")
                            continue
                        loader_cfg = LoaderConfig(
                            loader_name=plan.name,
                            run=benchmark_cfg.run,
                            config=plan.config,
                        )

                        _emit("[INFO] Running pre-run cleanup")
                        cleanup_between_runs(tune_system=not args.no_system_tuning)

                        if args.local and args.remote and not args.dont_cleanup_local:
                            _emit(
                                f"[cleanup] purging local cache contents: {local_path}"
                            )
                            purge_directory_contents(local_path)

                        try:
                            if benchmark_cfg.run.verbose:
                                _emit(
                                    f"[INFO] Warming up + measuring loader '{plan.name}'"
                                )
                                log_childs()

                            if args.require_no_gil:
                                _require_no_gil(environment_snapshot())
                            result, resolved_config = run_benchmark(
                                factory=factory,
                                cfg=loader_cfg,
                                warmup_steps=benchmark_cfg.warmup_steps,
                                measure_steps=benchmark_cfg.measure_steps,
                                packing_stats=packing_stats,
                            )
                            if result.steps <= 0:
                                raise RuntimeError("No batches were measured")
                            if (
                                not math.isfinite(result.total_time)
                                or result.total_time <= 0
                            ):
                                raise RuntimeError("Nonpositive measured duration")
                            if args.strict:
                                if result.steps != measure_steps:
                                    raise RuntimeError(
                                        f"Incomplete measurement: {result.steps}/{measure_steps} batches"
                                    )
                                if packing_stats and not result.total_usable_tokens:
                                    raise RuntimeError(
                                        "Packing measurement did not produce usable tokens"
                                    )
                            environment = (
                                environment_snapshot()
                                if args.json_output or args.require_no_gil
                                else None
                            )
                            if args.require_no_gil:
                                _require_no_gil(environment)
                        except NotImplementedError as err:
                            failures.append(
                                {
                                    "loader_name": plan.name,
                                    "error": str(err),
                                    "batch_size": batch_size,
                                    "num_workers": num_workers,
                                    "seed": seed,
                                }
                            )
                            _emit(f"[WARN] Loader '{plan.name}' not implemented: {err}")
                            _advance(group_name, f"{status_prefix} not implemented")
                            continue
                        except Exception as err:
                            failures.append(
                                {
                                    "loader_name": plan.name,
                                    "error": str(err),
                                    "batch_size": batch_size,
                                    "num_workers": num_workers,
                                    "seed": seed,
                                }
                            )
                            tb = traceback.format_exc()

                            _emit(f"[ERROR] Loader '{plan.name}' failed: {err}\n\n{tb}")
                            _advance(group_name, f"{status_prefix} failed: {err}")
                            continue
                        result.name = plan.name
                        results.append(result)
                        row = _prepare_csv_row(
                            result, loader_cfg, benchmark_cfg, resolved_config
                        )
                        csv_rows.append(row)
                        if args.json_output:
                            observations.append(
                                {
                                    "result": json_value(dataclasses.asdict(result)),
                                    "resolved_config": json_value(resolved_config),
                                    "environment": environment,
                                    "cache_policy": "unchanged"
                                    if args.no_system_tuning
                                    else "best-effort page-cache drop",
                                    "csv_row": row,
                                }
                            )
                        _advance(
                            group_name,
                            f"{status_prefix} finished in {result.total_time:0.2f}s "
                            f"({result.steps} steps, setup {result.setup_time:0.2f}s)",
                        )
                        if benchmark_cfg.run.verbose:
                            log_childs()

            _run_factory_group(zephon_process_factories, "zephon-process")
            _run_factory_group(regular_factories, "regular")

    if results:
        print(format_results(results))
    else:
        print("No loaders produced results. Check warnings above.", file=sys.stderr)

    if args.csv_output:
        _write_csv_output(
            Path(args.csv_output).expanduser(), csv_rows, base_fields=_BASE_CSV_FIELDS
        )
    if not results:
        failures.append({"error": "No loaders produced results"})
    if args.json_output:
        write_json_output(
            args.json_output,
            {
                "schema_version": 1,
                "metric": "throughput",
                "results": observations,
                "failures": failures,
            },
        )
    return int(not results or (bool(failures) and (args.strict or args.require_no_gil)))


if __name__ == "__main__":
    raise SystemExit(app())
