"""CLI entry point for time-to-first-batch (TTFB) benchmarking."""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import dataclasses
import statistics
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from .context import LoaderConfig, LoaderPlan, RunParameters
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

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TTFBTrial:
    """Timing breakdown for a single TTFB measurement.

    Phase timings (factory_time, iter_time, first_next_time) are wall-clock
    durations of each individual call.  Composite timings capture spans that
    matter when work overlaps phase boundaries:

    * ``iter_to_batch`` = iter_time + first_next_time — wall-clock from
      ``iter(loader)`` to the first batch.  Captures the case where ``iter()``
      spawns prefetch workers that run concurrently before ``next()`` is called.
    * ``total_ttfb`` = factory_time + iter_to_batch — full end-to-end.
    """

    factory_time: float
    iter_time: float
    first_next_time: float
    iter_to_batch: float  # iter() → first batch (captures prefetch overlap)
    total_ttfb: float  # factory() → first batch (full e2e)


@dataclass(slots=True)
class TTFBResult:
    """Aggregated TTFB result across multiple repetitions."""

    name: str
    batch_size: int | None
    num_workers: int
    seed: int | None
    repetitions: int

    # e2e
    ttfb_median: float
    ttfb_mean: float
    ttfb_std: float
    ttfb_min: float
    ttfb_max: float

    # iter → first batch
    iter_to_batch_median: float

    # per-phase
    factory_time_median: float
    iter_time_median: float
    first_next_time_median: float

    trials: list[TTFBTrial] = field(default_factory=list)


def _aggregate_trials(trials: list[TTFBTrial]) -> dict[str, float]:
    """Compute aggregate statistics over a list of trials."""
    ttfbs = [t.total_ttfb for t in trials]
    iter_to_batches = [t.iter_to_batch for t in trials]
    factory_times = [t.factory_time for t in trials]
    iter_times = [t.iter_time for t in trials]
    first_next_times = [t.first_next_time for t in trials]
    n = len(ttfbs)
    return {
        "ttfb_median": statistics.median(ttfbs),
        "ttfb_mean": statistics.mean(ttfbs),
        "ttfb_std": statistics.stdev(ttfbs) if n >= 2 else 0.0,
        "ttfb_min": min(ttfbs),
        "ttfb_max": max(ttfbs),
        "iter_to_batch_median": statistics.median(iter_to_batches),
        "factory_time_median": statistics.median(factory_times),
        "iter_time_median": statistics.median(iter_times),
        "first_next_time_median": statistics.median(first_next_times),
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_ttfb(
    factory: ConfigFactory,
    cfg: LoaderConfig,
) -> tuple[TTFBTrial, Mapping[str, Any]]:
    """Time loader construction through its first batch, excluding bookkeeping."""
    loader = iterator = None
    try:
        t0 = time.perf_counter()
        loader = factory(cfg)
        t_factory = time.perf_counter()
        iterator = iter(loader)
        t_iter = time.perf_counter()
        try:
            first_batch = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"Loader '{cfg.loader_name}' produced no batches"
            ) from error
        t_first_batch = time.perf_counter()
        trial = TTFBTrial(
            factory_time=t_factory - t0,
            iter_time=t_iter - t_factory,
            first_next_time=t_first_batch - t_iter,
            iter_to_batch=t_first_batch - t_factory,
            total_ttfb=t_first_batch - t0,
        )
        resolved_config = _extract_resolved_config(loader, factory, cfg)
        del first_batch
        return trial, resolved_config
    finally:
        _shutdown_loader(loader, iterator)


def run_ttfb(
    factory: ConfigFactory,
    cfg: LoaderConfig,
    repetitions: int,
    *,
    cleanup_local: bool = False,
    local_path: Path | None = None,
    verbose: bool = False,
    system_tuning: bool = True,
    cache_drops: list[bool | None] | None = None,
) -> tuple[TTFBResult, Mapping[str, Any]]:
    """Run *repetitions* TTFB trials with cleanup between each.

    ``cache_drops`` receives each page-cache drop's outcome.
    """
    trials: list[TTFBTrial] = []
    resolved_config: Mapping[str, Any] = {}

    for rep in range(repetitions):
        if rep > 0:
            dropped = cleanup_between_runs(tune_system=system_tuning)
            if cache_drops is not None:
                cache_drops.append(dropped)
            if cleanup_local and local_path is not None:
                purge_directory_contents(local_path)

        trial, resolved_config = measure_ttfb(factory, cfg)
        trials.append(trial)

        if verbose:
            print(
                f"  [rep {rep + 1}/{repetitions}] "
                f"ttfb={trial.total_ttfb:.3f}s  "
                f"iter→batch={trial.iter_to_batch:.3f}s  "
                f"(factory={trial.factory_time:.3f}s  "
                f"iter={trial.iter_time:.3f}s  "
                f"next={trial.first_next_time:.3f}s)",
                file=sys.stderr,
            )

    agg = _aggregate_trials(trials)
    result = TTFBResult(
        name="",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        repetitions=repetitions,
        trials=trials,
        **agg,
    )
    return result, resolved_config


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------


def format_ttfb_results(results: list[TTFBResult], repetitions: int) -> str:
    header = (
        f"{'Loader':<20} "
        f"{'Batch':>7} "
        f"{'Workers':>9} "
        f"{'Seed':>7} "
        f"{'TTFB(med)':>11} "
        f"{'TTFB(std)':>11} "
        f"{'factory(med)':>14} "
        f"{'iter→batch(med)':>17} "
        f"{'iter(med)':>12} "
        f"{'next(med)':>12}"
    )
    lines = [
        f"TTFB Results ({repetitions} repetitions)",
        header,
        "-" * len(header),
    ]
    for r in results:
        batch_display = "-" if r.batch_size is None else str(r.batch_size)
        seed_display = "-" if r.seed is None else str(r.seed)
        lines.append(
            f"{r.name:<20} "
            f"{batch_display:>7} "
            f"{str(r.num_workers):>9} "
            f"{seed_display:>7} "
            f"{r.ttfb_median:>10.3f}s "
            f"{r.ttfb_std:>10.3f}s "
            f"{r.factory_time_median:>13.3f}s "
            f"{r.iter_to_batch_median:>16.3f}s "
            f"{r.iter_time_median:>11.3f}s "
            f"{r.first_next_time_median:>11.3f}s"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_BASE_TTFB_CSV_FIELDS: tuple[str, ...] = (
    "loader_name",
    "benchmark.repetitions",
    "result.ttfb_median_s",
    "result.ttfb_mean_s",
    "result.ttfb_std_s",
    "result.ttfb_min_s",
    "result.ttfb_max_s",
    "result.iter_to_batch_median_s",
    "result.factory_time_median_s",
    "result.iter_time_median_s",
    "result.first_next_time_median_s",
    "run.local_path",
    "run.remote_path",
    "run.batch_size",
    "run.num_workers",
    "run.seed",
    "run.verbose",
    "config.plaintext",
    "config.resolved.plaintext",
)

_BASE_TTFB_TRIAL_CSV_FIELDS: tuple[str, ...] = (
    "loader_name",
    "trial",
    "result.total_ttfb_s",
    "result.iter_to_batch_s",
    "result.factory_time_s",
    "result.iter_time_s",
    "result.first_next_time_s",
    "run.local_path",
    "run.remote_path",
    "run.batch_size",
    "run.num_workers",
    "run.seed",
    "config.plaintext",
    "config.resolved.plaintext",
)


def _prepare_ttfb_csv_row(
    result: TTFBResult,
    loader_cfg: LoaderConfig,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    run = loader_cfg.run
    return {
        "loader_name": result.name,
        "benchmark.repetitions": result.repetitions,
        "result.ttfb_median_s": result.ttfb_median,
        "result.ttfb_mean_s": result.ttfb_mean,
        "result.ttfb_std_s": result.ttfb_std,
        "result.ttfb_min_s": result.ttfb_min,
        "result.ttfb_max_s": result.ttfb_max,
        "result.iter_to_batch_median_s": result.iter_to_batch_median,
        "result.factory_time_median_s": result.factory_time_median,
        "result.iter_time_median_s": result.iter_time_median,
        "result.first_next_time_median_s": result.first_next_time_median,
        "run.local_path": str(run.local_path),
        "run.remote_path": _optional(run.remote_path),
        "run.batch_size": _optional(run.batch_size),
        "run.num_workers": run.num_workers,
        "run.seed": _optional(run.seed),
        "run.verbose": run.verbose,
        "config.plaintext": _stringify_config(loader_cfg.config),
        "config.resolved.plaintext": _stringify_config(resolved_config),
    }


def _prepare_ttfb_trial_rows(
    result: TTFBResult,
    loader_cfg: LoaderConfig,
    resolved_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    run = loader_cfg.run
    base = {
        "loader_name": result.name,
        "run.local_path": str(run.local_path),
        "run.remote_path": _optional(run.remote_path),
        "run.batch_size": _optional(run.batch_size),
        "run.num_workers": run.num_workers,
        "run.seed": _optional(run.seed),
        "config.plaintext": _stringify_config(loader_cfg.config),
        "config.resolved.plaintext": _stringify_config(resolved_config),
    }
    rows: list[dict[str, Any]] = []
    for idx, trial in enumerate(result.trials, start=1):
        row = {
            **base,
            "trial": idx,
            "result.total_ttfb_s": trial.total_ttfb,
            "result.iter_to_batch_s": trial.iter_to_batch,
            "result.factory_time_s": trial.factory_time,
            "result.iter_time_s": trial.iter_time,
            "result.first_next_time_s": trial.first_next_time,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def ttfb_app(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dataloader-ttfb",
        description="Measure time-to-first-batch for dataloaders.",
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
        help="Optional remote dataset URI (e.g. s3://bucket/path).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("examples/loaders.json")),
        help="Path to loader configuration file (JSON).",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=5,
        help="Number of TTFB trials per loader/combination (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        type=_batch_option,
        nargs="+",
        action="append",
        default=None,
        metavar="INT|none",
        help="Global batch size. May be supplied multiple times. Defaults to 32.",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers_values",
        type=int,
        nargs="+",
        action="append",
        default=None,
        metavar="INT",
        help="Number of workers. May be supplied multiple times. Defaults to 4.",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=_seed_option,
        nargs="+",
        action="append",
        default=None,
        metavar="SEED",
        help="Optional RNG seed. May be supplied multiple times.",
    )
    parser.add_argument(
        "--only",
        dest="only_loaders",
        action="append",
        default=None,
        metavar="NAME",
        help="Only benchmark loaders whose name appears in the config JSON.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Path to write aggregated TTFB results as CSV.",
    )
    parser.add_argument(
        "--per-trial-csv",
        type=str,
        default=None,
        help="Path to write per-trial (individual repetition) CSV.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of failing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial debug information.",
    )
    parser.add_argument(
        "--dont-cleanup-local",
        action="store_true",
        help="Do NOT purge the local cache directory between trials.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any loader is skipped or fails. Without it, the run fails only when no loader succeeds.",
    )
    parser.add_argument(
        "--require-no-gil",
        action="store_true",
        help="Require a free-threaded Python with GIL disabled",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write results, trials, provenance and failures as JSON",
    )
    parser.add_argument(
        "--no-system-tuning",
        action="store_true",
        help="Skip host tuning and OS page-cache drops",
    )
    args = parser.parse_args(argv)
    _warn_gil_disabled()
    failures: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for output in (args.csv_output, args.per_trial_csv, args.json_output):
        if output and Path(output).expanduser().exists() and not args.force_overwrite:
            parser.error(
                f"Output exists: {output}; use --force-overwrite to replace it"
            )

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
    repetitions = args.repetitions
    cleanup_local = bool(args.local and args.remote and not args.dont_cleanup_local)

    # Resolve and order factories.
    loader_plans = tuple(specs)
    resolved_factories: list[tuple[LoaderPlan, ConfigFactory]] = []
    for plan in loader_plans:
        try:
            factory = import_factory(plan.factory)
        except (ImportError, AttributeError, ValueError) as err:
            failures.append(
                {"loader_name": plan.name, "phase": "import", "error": str(err)}
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
    ordered_factories: list[tuple[LoaderPlan, ConfigFactory]] = [
        *zephon_process_factories,
        *regular_factories,
    ]

    results: list[TTFBResult] = []
    csv_rows: list[dict[str, Any]] = []
    trial_csv_rows: list[dict[str, Any]] = []

    if ordered_factories:
        with nullcontext() if args.no_system_tuning else SystemStabilizer():
            combinations = list(product(batch_sizes, worker_counts, seed_values))
            combinations = _prioritize_worker_combinations(combinations)
            total_runs = len(combinations) * len(ordered_factories)
            completed = 0
            overall_start = time.perf_counter()

            for combo_index, (batch_size, num_workers, seed) in enumerate(
                combinations, start=1
            ):
                batch_display = "none" if batch_size is None else str(batch_size)
                seed_display = "none" if seed is None else str(seed)

                run_params = RunParameters(
                    local_path=local_path,
                    remote_path=args.remote,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    seed=seed,
                    verbose=args.verbose,
                )

                for plan, factory in ordered_factories:
                    should_skip, reason = _should_skip_loader(plan, run_params)
                    if should_skip:
                        failures.append(
                            {
                                "loader_name": plan.name,
                                "phase": "skip",
                                "error": reason,
                            }
                        )
                        if run_params.verbose:
                            print(
                                f"[INFO] Skipping loader '{plan.name}': {reason}",
                                file=sys.stderr,
                            )
                        completed += 1
                        continue

                    loader_cfg = LoaderConfig(
                        loader_name=plan.name,
                        run=run_params,
                        config=plan.config,
                    )

                    print(
                        f"[TTFB] {plan.name}  "
                        f"batch={batch_display} workers={num_workers} seed={seed_display}  "
                        f"({repetitions} reps)",
                        file=sys.stderr,
                    )

                    # Initial cleanup before first trial
                    cache_drops = [
                        cleanup_between_runs(tune_system=not args.no_system_tuning)
                    ]
                    if cleanup_local:
                        purge_directory_contents(local_path)

                    try:
                        if run_params.verbose:
                            log_childs()

                        if args.require_no_gil:
                            _require_no_gil(environment_snapshot())
                        result, resolved_config = run_ttfb(
                            factory=factory,
                            cfg=loader_cfg,
                            repetitions=repetitions,
                            cleanup_local=cleanup_local,
                            local_path=local_path if cleanup_local else None,
                            verbose=args.verbose,
                            system_tuning=not args.no_system_tuning,
                            cache_drops=cache_drops,
                        )
                        environment = environment_snapshot()
                        if args.require_no_gil:
                            _require_no_gil(environment)
                    except Exception as err:
                        failures.append(
                            {
                                "loader_name": plan.name,
                                "error": str(err),
                                "num_workers": num_workers,
                                "batch_size": batch_size,
                                "seed": seed,
                            }
                        )
                        tb = traceback.format_exc()
                        print(
                            f"[ERROR] Loader '{plan.name}' failed: {err}\n\n{tb}",
                            file=sys.stderr,
                        )
                        completed += 1
                        continue

                    result.name = plan.name
                    results.append(result)
                    if False in cache_drops:
                        print(
                            f"[WARN] {plan.name}: page-cache drop failed; "
                            "some trials ran with a warm cache.",
                            file=sys.stderr,
                        )
                    observations.append(
                        {
                            "result": json_value(dataclasses.asdict(result)),
                            "resolved_config": json_value(resolved_config),
                            "environment": environment,
                            "cache_policy": "unchanged"
                            if args.no_system_tuning
                            else "best-effort page-cache drop",
                            # One entry per trial; False means that trial ran warm.
                            "page_cache_dropped": cache_drops,
                        }
                    )
                    csv_rows.append(
                        _prepare_ttfb_csv_row(result, loader_cfg, resolved_config)
                    )
                    if args.per_trial_csv:
                        trial_csv_rows.extend(
                            _prepare_ttfb_trial_rows(
                                result, loader_cfg, resolved_config
                            )
                        )

                    completed += 1
                    elapsed = time.perf_counter() - overall_start
                    remaining = max(total_runs - completed, 0)
                    eta = (elapsed / completed * remaining) if completed else None
                    eta_display = "--:--" if eta is None else _format_duration(eta)
                    print(
                        f"[PROGRESS] {completed}/{total_runs}  "
                        f"elapsed {_format_duration(elapsed)}  eta {eta_display}  "
                        f"| {plan.name} median={result.ttfb_median:.3f}s",
                        file=sys.stderr,
                    )

    # Print terminal results
    if results:
        print(format_ttfb_results(results, repetitions))
    else:
        print("No loaders produced results. Check warnings above.", file=sys.stderr)

    if args.csv_output:
        _write_csv_output(
            Path(args.csv_output).expanduser(),
            csv_rows,
            base_fields=_BASE_TTFB_CSV_FIELDS,
        )
    if args.per_trial_csv:
        _write_csv_output(
            Path(args.per_trial_csv).expanduser(),
            trial_csv_rows,
            base_fields=_BASE_TTFB_TRIAL_CSV_FIELDS,
        )
    if not results:
        failures.append({"error": "No loaders produced results"})
    if args.json_output:
        write_json_output(
            args.json_output,
            {
                "schema_version": 1,
                "metric": "ttfb",
                "results": observations,
                "failures": failures,
            },
        )
    return int(not results or (bool(failures) and (args.strict or args.require_no_gil)))


if __name__ == "__main__":
    raise SystemExit(ttfb_app())
