"""Checkpoint measurement functions and CLI for checkpointable data loaders."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy
import hashlib
import importlib.metadata
import json
import pickle
import platform
import time
import uuid
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .context import LoaderConfig, RunParameters
from .harness import (
    _extract_resolved_config,
    _positive_int,
    _shutdown_loader,
    environment_snapshot,
    import_factory,
    load_config,
)
from .utils.checkpointing import (
    RssSampler,
    cache_summary,
    fingerprint,
    prepare_cache,
    read_bytes,
    summarize_checkpoint,
    write_bytes,
    write_json,
)


def batches_for_steps(
    steps: int, global_batch: int, micro_batch: int, ranks: int
) -> int:
    if min(steps, global_batch, micro_batch, ranks) < 1:
        raise ValueError("progress and batch sizes must be positive")
    batches, remainder = divmod(steps * global_batch, micro_batch * ranks)
    if remainder:
        raise ValueError("optimizer steps must map to whole batches per DP rank")
    return batches


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "save", "restore"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="loader configuration file (same format as throughput/TTFB)",
    )
    parser.add_argument(
        "--only", help="select one loader by name from the shared loader config"
    )
    parser.add_argument(
        "--local",
        type=Path,
        required=True,
        help="dataset directory, or cache base with --remote",
    )
    parser.add_argument("--remote", help="optional dataset URI")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=4,
        help="microbatch size per DP rank",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="source and shared VLM packing seed (overrides config)",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "--batches",
        type=_positive_int,
        help="pulls per rank before saving (default: 100)",
    )
    progress.add_argument(
        "--optimizer-steps",
        type=_positive_int,
        help="simulated progress; no optimizer runs",
    )
    parser.add_argument("--global-batch-size", type=_positive_int)
    parser.add_argument(
        "--checkpoint-uri",
        help="local path or s3://bucket/key; required for save/restore",
    )
    parser.add_argument(
        "--aggregate-dir",
        help="shared local directory or S3 prefix; required with multiple ranks",
    )
    parser.add_argument(
        "--json-output",
        required=True,
        dest="report_uri",
        help="output JSON, local path or S3 URI",
    )
    parser.add_argument("--cache-policy", choices=("cold", "reuse"), default="cold")
    parser.add_argument(
        "--cache-scope",
        choices=("rank", "host"),
        default="rank",
        help="host shares one cache across local ranks",
    )
    parser.add_argument(
        "--cache-namespace", help="default: a fresh namespace per invocation"
    )
    parser.add_argument(
        "--verify-batches",
        type=_positive_int,
        default=4,
        help="consecutive uninterrupted/resumed batch hashes to compare",
    )
    parser.add_argument("--rss-poll-ms", type=float, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "fresh" and not args.checkpoint_uri:
        parser.error("--checkpoint-uri is required for save and restore")
    if args.optimizer_steps is not None and args.global_batch_size is None:
        parser.error("--optimizer-steps requires --global-batch-size")
    if args.rss_poll_ms <= 0:
        parser.error("--rss-poll-ms must be positive")
    return args


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def gather(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def prepare_remote_cache(args, local: Path, namespace: str, rank: int) -> Path:
    """Initialize a cold cache once per rank or host before any peer opens it."""
    host_shared = args.cache_scope == "host"
    leader = not host_shared or int(os.environ.get("LOCAL_RANK", "0")) == 0
    cache = None
    error = None
    if leader:
        try:
            cache = prepare_cache(
                local,
                namespace,
                None if host_shared else rank,
                cold=args.cache_policy == "cold",
            )
        except Exception as exception:
            error = f"rank {rank}: {exception}"
    errors = [message for message in gather(error) if message is not None]
    if errors:
        raise RuntimeError("cache initialization failed: " + "; ".join(errors))
    if cache is None:
        cache = prepare_cache(local, namespace, None, cold=False)
    return cache


def identity(
    factory: str, options: dict, *, dataset: str, batch_size: int, seed: int, ranks: int
) -> str:
    """Require the same workload on restore, allowing new cache/aggregation paths."""
    stable = copy.deepcopy(options)
    for key in ("run_id", "aggregate_dir", "global_rank", "dp_group_id"):
        stable.get("pipeline", {}).pop(key, None)
    content = {
        "factory": factory,
        "options": stable,
        "dataset": dataset,
        "batch_size": batch_size,
        "seed": seed,
        "ranks": ranks,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _timed(timings: dict, name: str, function, *args):
    start = time.perf_counter()
    result = function(*args)
    timings[name] = time.perf_counter() - start
    return result


def _versions() -> dict[str, Any]:
    packages = {}
    for name in (
        "zephon",
        "litdata",
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "dataloader-arena",
        "boto3",
        "botocore",
        "obstore",
    ):
        try:
            distribution = importlib.metadata.distribution(name)
            packages[name] = {
                "version": distribution.version,
                "direct_url": json.loads(
                    distribution.read_text("direct_url.json") or "null"
                ),
            }
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import zephon

    source_hashes = {}
    for name, root in {
        "arena": Path(__file__).resolve().parent,
        "zephon": Path(zephon.__file__).resolve().parent,
    }.items():
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes())
        source_hashes[name] = digest.hexdigest()
    return {
        "python": platform.python_version(),
        "packages": packages,
        "source_sha256": source_hashes,
    }


def _optional_metrics(pipeline, name: str) -> dict:
    method = getattr(pipeline, name, None)
    if not callable(method):
        return {
            "status": "unavailable",
            "reason": f"installed Zephon has no public {name}()",
        }
    value = method()
    if value is None:
        return {
            "status": "unavailable",
            "reason": f"installed Zephon does not provide {name}() data",
        }
    return {"status": "available", "snapshot": value}


def _input_index_sha256(local: Path, remote: str | None) -> str | None:
    if remote:
        if not remote.startswith("s3://"):
            raise ValueError(
                "checkpoint dataset URIs currently support S3 or local paths"
            )
        payload = read_bytes(remote.rstrip("/") + "/index.json")
    elif (local / "index.json").is_file():
        payload = (local / "index.json").read_bytes()
    else:
        return None  # Custom factories may not use LitData.
    return hashlib.sha256(payload).hexdigest()


def measure_checkpoint(
    pipeline,
    args,
    *,
    rank: int,
    ranks: int,
    batches: int,
    workload_id: str,
    mapping_strategy: str = "contiguous",
    source_index: str | None = None,
    versions: dict | None = None,
):
    """Keep all hashing, summaries and manifest IO outside latency/RSS windows."""
    timings: dict[str, float] = {}
    record: dict[str, Any] = {"rank": rank, "timings_s": timings}
    state = payload = manifest = None
    iterator = None
    try:
        if args.mode == "restore":
            manifest = json.loads(read_bytes(args.checkpoint_uri + ".manifest.json"))
            if manifest["workload_id"] != workload_id:
                raise ValueError(
                    "restore configuration differs from the saved workload"
                )
            if manifest["dataset_index_sha256"] != source_index:
                raise ValueError("dataset index changed since save")
            if manifest["versions"] != versions:
                raise ValueError(
                    "restore requires the same source/dependency versions as save"
                )
            if len(manifest["continuation_hashes"][rank]) != args.verify_batches:
                raise ValueError("--verify-batches differs from the save run")
            barrier()
            with RssSampler(args.rss_poll_ms / 1000) as rss:
                start = time.perf_counter()
                payload = _timed(
                    timings, "checkpoint_read", read_bytes, args.checkpoint_uri
                )
                state = _timed(timings, "checkpoint_deserialize", pickle.loads, payload)
                _timed(timings, "restore_stash", pipeline.restore, state)
                _timed(timings, "restore_sync_barrier", barrier)
                iterator = _timed(timings, "iterator_build", iter, pipeline)
                first = _timed(timings, "first_batch_pull", next, iterator)
                timings["restore_to_first_batch"] = time.perf_counter() - start
                record["replay_metrics_at_first_batch"] = _optional_metrics(
                    pipeline, "replay_metrics"
                )
                first_batch_rss = rss.snapshot()
            record["memory"] = {
                "window": "checkpoint read through first restored batch",
                "poll_interval_ms": args.rss_poll_ms,
                "rss_baseline_bytes": first_batch_rss.baseline_bytes,
                "rss_peak_bytes": first_batch_rss.peak_bytes,
                "rss_incremental_bytes": first_batch_rss.incremental_bytes,
            }
            record["memory_through_replay_completion"] = {
                "status": "unavailable",
                "reason": "requires a public replay-completion signal",
            }
            # Reading/validating this manifest is deliberately outside restore TTFB.
            if hashlib.sha256(payload).hexdigest() != manifest["checkpoint_sha256"]:
                raise ValueError("checkpoint integrity check failed")
            actual = [fingerprint(first)] + [
                fingerprint(next(iterator)) for _ in range(args.verify_batches - 1)
            ]
            expected = manifest["continuation_hashes"][rank]
            record["continuation"] = {
                "expected": expected,
                "actual": actual,
                "matches": actual == expected,
            }
            record["replay_metrics_after_validation"] = _optional_metrics(
                pipeline, "replay_metrics"
            )
        else:
            barrier()
            with (
                RssSampler(args.rss_poll_ms / 1000)
                if args.mode == "fresh"
                else nullcontext()
            ) as fresh_rss:
                iterator = _timed(timings, "iterator_build", iter, pipeline)
                _timed(timings, "first_batch_pull", next, iterator)
                timings["fresh_to_first_batch"] = (
                    timings["iterator_build"] + timings["first_batch_pull"]
                )
            if fresh_rss is not None:
                record["memory"] = {
                    "window": "iterator startup through first fresh batch",
                    "poll_interval_ms": args.rss_poll_ms,
                    **fresh_rss.result(),
                }
            if args.mode == "save":
                consume_start = time.perf_counter()
                for _ in range(batches - 1):
                    next(iterator)
                timings["consume_after_first"] = time.perf_counter() - consume_start
                record["batches_before_save_per_rank"] = batches
                barrier()
                with RssSampler(args.rss_poll_ms / 1000) as rss:
                    start = time.perf_counter()
                    state = _timed(timings, "pipeline_checkpoint", pipeline.checkpoint)
                    payload = _timed(
                        timings,
                        "checkpoint_serialize",
                        pickle.dumps,
                        state,
                        pickle.HIGHEST_PROTOCOL,
                    )
                    if rank == 0:
                        _timed(
                            timings,
                            "checkpoint_write",
                            write_bytes,
                            args.checkpoint_uri,
                            payload,
                        )
                    _timed(timings, "save_sync_barrier", barrier)
                    timings["save_end_to_end"] = time.perf_counter() - start
                record["memory"] = {
                    "window": "checkpoint capture through serialization, final write and synchronization",
                    "poll_interval_ms": args.rss_poll_ms,
                    **rss.result(),
                }
                record["checkpoint_metrics"] = _optional_metrics(
                    pipeline, "checkpoint_metrics"
                )
                # Oracle: continue the SAME live iterator after capturing its checkpoint.
                record["continuation_hashes"] = [
                    fingerprint(next(iterator)) for _ in range(args.verify_batches)
                ]
    finally:
        _shutdown_loader(pipeline, iterator)
    if state is not None:
        record["checkpoint_bytes"] = len(payload)
        record["checkpoint_source_state"] = summarize_checkpoint(
            state, ranks, mapping_strategy
        )
    return record, payload, manifest


def run_checkpoint(args) -> dict | None:
    torch.set_num_threads(1)
    for key, value in {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(key, value)
    ranks = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if ranks > 1 and not args.aggregate_dir:
        raise ValueError("multiple ranks require --aggregate-dir on shared storage")
    if ranks > 1:
        dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    try:
        return _run(args, rank, ranks)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(args, rank: int, ranks: int) -> dict | None:
    invocation = gather(uuid.uuid4().hex)[0]
    plans = load_config(args.config)
    selected = [plan for plan in plans if args.only is None or plan.name == args.only]
    if len(selected) != 1:
        raise ValueError("select exactly one checkpointable loader with --only NAME")
    plan = selected[0]
    options = copy.deepcopy(dict(plan.config))
    runtime = options.setdefault("pipeline", {})
    # This harness simulates pure data parallelism: one distinct partition per rank.
    if (
        runtime.get("world_size", ranks) != ranks
        or runtime.get("dp_degree", ranks) != ranks
    ):
        raise ValueError(
            "world_size and dp_degree must match torchrun's number of ranks"
        )
    runtime.update(
        world_size=ranks,
        global_rank=rank,
        dp_degree=ranks,
        dp_group_id=rank,
        run_id=invocation,
        mtp_auto_checkpoint=False,
    )
    if args.aggregate_dir:
        runtime["aggregate_dir"] = args.aggregate_dir.rstrip("/") + "/" + invocation
    else:
        runtime["aggregate_dir"] = None
    if plan.factory == "dataloader_arena.benchmarks.vlm.zephon:create_loader":
        vlm_options = options.setdefault("vlm_zephon", {})
        if (
            vlm_options.get("batch_metadata", "preserve_contributors")
            != "preserve_contributors"
        ):
            raise ValueError(
                "checkpoint VLM requires batch_metadata=preserve_contributors"
            )
        vlm_options["batch_metadata"] = "preserve_contributors"
        options.setdefault("packing", {})["seed"] = args.seed
    cache = None
    local = args.local.resolve()
    if args.remote:
        cache_options = runtime.get("io_options", {}).get("cache", {})
        if "root" in cache_options or "enabled" in cache_options:
            raise ValueError(
                "cache root/enabled are managed by --local/--cache-namespace; "
                "pipeline.io_options.cache may set limits and retry options"
            )
        cache = prepare_remote_cache(
            args,
            local,
            args.cache_namespace or invocation,
            rank,
        )
        local = cache
    batches = (
        batches_for_steps(
            args.optimizer_steps, args.global_batch_size, args.batch_size, ranks
        )
        if args.optimizer_steps is not None
        else args.batches or 100
    )
    workload_id = identity(
        plan.factory,
        options,
        dataset=args.remote or str(local),
        batch_size=args.batch_size,
        seed=args.seed,
        ranks=ranks,
    )
    source_index = _input_index_sha256(local, args.remote)
    provenance = {
        "environment": environment_snapshot(),
        "versions": _versions(),
        "dataset_index_sha256": source_index,
        "workload_id": workload_id,
    }
    config = LoaderConfig(
        plan.name,
        RunParameters(
            local, args.remote, args.batch_size, 0, args.seed, verbose=args.verbose
        ),
        options,
    )
    start = time.perf_counter()
    factory = import_factory(plan.factory)
    pipeline = factory(config)
    if not callable(getattr(pipeline, "checkpoint", None)) or not callable(
        getattr(pipeline, "restore", None)
    ):
        _shutdown_loader(pipeline)
        raise TypeError(
            f"Loader {plan.name!r} must expose checkpoint() and restore(); "
            "for Zephon fetch-and-batch, set skip_loader=true"
        )
    resolved = dict(_extract_resolved_config(pipeline, factory, config))
    construction_s = time.perf_counter() - start
    record, payload, manifest = measure_checkpoint(
        pipeline,
        args,
        rank=rank,
        ranks=ranks,
        batches=batches,
        workload_id=workload_id,
        mapping_strategy=runtime.get("mapping_strategy") or "contiguous",
        source_index=source_index,
        versions=provenance["versions"],
    )
    if _input_index_sha256(local, args.remote) != source_index:
        raise RuntimeError("dataset index changed during measurement")
    if _versions() != provenance["versions"]:
        raise RuntimeError(
            "installed sources or dependencies changed during measurement"
        )
    record.update(construction_s=construction_s, provenance=provenance)
    if cache is not None:
        record["cache"] = {
            "policy": args.cache_policy,
            "scope": args.cache_scope,
            "path": str(cache),
            **(
                cache_summary(cache)
                if args.cache_scope == "rank"
                or int(os.environ.get("LOCAL_RANK", "0")) == 0
                else {"summary_owner_local_rank": 0}
            ),
        }
    records = gather(record)
    if rank != 0:
        return None
    if args.mode == "save":
        manifest = {
            "schema_version": 1,
            "workload_id": workload_id,
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "dataset_index_sha256": source_index,
            "continuation_hashes": [r["continuation_hashes"] for r in records],
            "resolved_config": resolved,
            "versions": provenance["versions"],
        }
        write_json(args.checkpoint_uri + ".manifest.json", manifest)
    keys = set().union(*(r["timings_s"] for r in records))
    report = {
        "schema_version": 1,
        "loader_name": plan.name,
        "factory": plan.factory,
        "mode": args.mode,
        "invocation": invocation,
        "world_size": ranks,
        "resolved_config": resolved,
        "ranks": records,
        "max_across_ranks_s": {
            key: max(r["timings_s"].get(key, 0) for r in records)
            for key in sorted(keys)
        },
        "measurement_scope": "CPU input pipeline; no model or optimizer; construction excluded from TTFB",
        "not_measured": [
            "checkpoint aggregation phase timings",
            "replay-filter/source replay counters",
            "full replay completion latency",
            "RSS through full replay completion",
            "exact aggregation codec bytes",
        ],
    }
    write_json(args.report_uri, report)
    if args.mode == "restore" and not all(
        r["continuation"]["matches"] for r in records
    ):
        raise RuntimeError(f"resumed continuation differs; see {args.report_uri}")
    print(f"{args.mode}: wrote {args.report_uri}")
    return report


def checkpoint_app(argv: list[str] | None = None) -> int:
    run_checkpoint(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(checkpoint_app())
