"""The storage-aware preparation path behind the single public data CLI."""

from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path

from .s3 import digest, json_bytes, location


def settings(args):
    import prepare_data as prep

    spec = json.loads(args.recipe.read_text()) if args.recipe else {}
    if spec.get("mode", args.mode) != args.mode:
        raise ValueError("Recipe dataset mode differs from the CLI")
    spec.update(
        mode=args.mode,
        samples=args.samples,
        seed=args.seed,
        chunk_bytes=int(args.chunk_mib * prep.MIB),
        workers=args.workers,
        revision=args.revision
        or (prep.VLM_REVISION if args.mode == "vlm" else prep.TEXT_REVISION),
        max_download_bytes=int(args.max_download_mib * prep.MIB),
        max_source_rows=args.max_source_rows
        or max(1024, args.samples * (4 if args.mode == "vlm" else 32)),
        image_representation=args.images,
        compression=args.compression,
    )
    if args.cache_dir is not None:
        spec["cache_dir"] = str(args.cache_dir.expanduser().resolve())
    spec.setdefault("spool_bytes", 256 * prep.MIB)
    spec.setdefault("scratch_bytes", 128 * 1024**3)
    spec.setdefault("max_spool_bytes", max(8 * 1024**3, args.samples * 8 * prep.MIB))
    spec.setdefault("max_output_bytes", max(8 * 1024**3, args.samples * 16 * prep.MIB))
    for option, key in (
        ("scratch_gib", "scratch_bytes"),
        ("max_spool_gib", "max_spool_bytes"),
        ("max_output_gib", "max_output_bytes"),
    ):
        if getattr(args, option) is not None:
            spec[key] = int(getattr(args, option) * 1024**3)
    spec.setdefault("minimum_records", args.samples)
    spec.setdefault("maximum_records", args.samples)
    spec.setdefault("minimum_shards", 1)
    spec.setdefault("maximum_shards", args.samples)
    if args.shard_samples_counts is not None:
        spec["shard_samples"] = args.shard_samples_counts
    if args.mode == "text":
        spec.update(
            sequence_length=args.sequence_length,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision
            or (
                prep.TOKENIZER_REVISION
                if args.tokenizer == prep.TOKENIZER_REPO
                else "main"
            ),
            batch_size=args.batch_size,
        )
    validate_settings(spec)
    return spec


def validate_settings(spec):
    for key in (
        "samples",
        "chunk_bytes",
        "spool_bytes",
        "scratch_bytes",
        "workers",
        "max_download_bytes",
        "max_spool_bytes",
        "max_output_bytes",
        "max_source_rows",
        "minimum_records",
        "maximum_records",
        "minimum_shards",
        "maximum_shards",
    ):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise ValueError(f"Recipe requires a positive integer {key}")
    if len(spec.get("revision", "")) != 40 or any(
        c not in "0123456789abcdef" for c in spec["revision"]
    ):
        raise ValueError("Use an immutable 40-character source revision")
    if not spec["minimum_records"] <= spec["samples"] <= spec["maximum_records"]:
        raise ValueError("Requested records are outside the recipe bounds")
    if not spec["minimum_shards"] <= spec["maximum_shards"] <= spec["samples"]:
        raise ValueError("Invalid physical shard bounds")
    if (
        not 1024 <= spec["chunk_bytes"] <= 1023 * 1024**2
        or not 1024 <= spec["spool_bytes"] <= 1024**3
        or spec["workers"] > 32
    ):
        raise ValueError(
            "Chunks must be 1 KiB–1023 MiB and spools 1 KiB–1 GiB, with at most 32 workers"
        )
    if spec["samples"] >= 2**32:
        raise ValueError("Permutation supports fewer than 2**32 records")


def describe(args, spec):
    scratch = (
        args.scratch
        or Path(".cache/arena-data") / digest(str(args.output).encode())[:16]
    )
    temporary = args.shuffle_storage or str(scratch / "shuffle")
    journal = args.journal or scratch / "journal"
    cache = Path(spec["cache_dir"])
    if not str(args.output).startswith("s3://"):
        output = Path(args.output).expanduser().resolve()
        for value in (scratch, journal, cache, temporary):
            if str(value).startswith("s3://"):
                continue
            other = Path(value).expanduser().resolve()
            if output.is_relative_to(other) or other.is_relative_to(output):
                raise ValueError(
                    "Local output must not overlap scratch, journal, cache or shuffle storage"
                )
    if not str(temporary).startswith("s3://"):
        temp = Path(temporary).expanduser().resolve()
        for value in (journal, cache):
            other = Path(value).expanduser().resolve()
            if temp.is_relative_to(other) or other.is_relative_to(temp):
                raise ValueError(
                    "Local shuffle storage must not overlap journal or source cache"
                )
    if str(args.output).startswith("s3://"):
        location(str(args.output))
    if str(temporary).startswith("s3://"):
        location(str(temporary))
    if spec["mode"] == "text":
        size = spec["samples"] * (4 + 4 * spec["sequence_length"])
        estimate = [size // 2 if spec["compression"] == "zstd" else size, size]
        spool = size
    else:
        per_record = (
            (700000, 1000000)
            if spec["image_representation"] == "decoded-rgb"
            else (350000, 600000)
        )
        estimate = [spec["samples"] * n for n in per_record]
        spool = spec["samples"] * 480000
    local_required = (spool if not str(temporary).startswith("s3://") else 0) + (
        32 * 1024**3 if spec["mode"] == "vlm" else 0
    )
    return dict(
        estimated_local_required_bytes=local_required,
        fits_scratch_estimate=local_required <= spec["scratch_bytes"],
        destination=str(args.output),
        mode=spec["mode"],
        records=spec["samples"],
        image_representation=spec["image_representation"],
        compression=spec["compression"],
        shuffle_storage=temporary,
        scratch=str(scratch),
        journal=str(journal),
        estimated_final_bytes=estimate,
        estimated_temporary_bytes=spool,
        estimated_s3_peak_bytes=[
            (n if str(args.output).startswith("s3://") else 0)
            + (spool if str(temporary).startswith("s3://") else 0)
            for n in estimate
        ],
        source_byte_cap=spec["max_download_bytes"],
        output_byte_cap=spec["max_output_bytes"],
        temporary_byte_cap=spec["max_spool_bytes"],
        scratch_byte_cap=spec["scratch_bytes"],
        physical_shards=[spec["minimum_shards"], spec["maximum_shards"]],
        caveat="Storage estimates are uncalibrated; exact size depends on source payloads and compression. Record shortages fail; samples are never repeated.",
    )


def size(value):
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{value} bytes"


def show_plan(plan):
    print(f"{plan['mode'].upper()} dataset: {plan['records']:,} records")
    print(f"Output: {plan['destination']}")
    print(f"Format: {plan['image_representation']}, {plan['compression']}")
    print(f"Temporary shuffle storage: {plan['shuffle_storage']}")
    low, high = plan["estimated_final_bytes"]
    print(f"Estimated final size: {size(low)}–{size(high)}")
    print(f"Estimated temporary records: {size(plan['estimated_temporary_bytes'])}")
    if plan["estimated_s3_peak_bytes"][1]:
        low, high = plan["estimated_s3_peak_bytes"]
        print(f"Estimated S3 peak: {size(low)}–{size(high)}")
    print(
        f"Caps: source {size(plan['source_byte_cap'])}; output {size(plan['output_byte_cap'])}; temporary {size(plan['temporary_byte_cap'])}"
    )
    print(
        f"Estimated local temporary records plus metadata: {size(plan['estimated_local_required_bytes'])}"
    )
    print(f"Local scratch cap: {plan['scratch_byte_cap'] / 1024**3:g} GiB")
    if not plan["fits_scratch_estimate"]:
        print(
            "Plan exceeds local scratch estimate; choose S3 shuffle storage or raise the scratch cap."
        )
    print(
        f"Physical shard guard: {plan['physical_shards'][0]:,}–{plan['physical_shards'][1]:,}"
    )
    if high >= 10**12:
        print("WARNING: this preparation creates multiple terabytes of data.")
    print(plan["caveat"], flush=True)


def confirm(plan, yes):
    show_plan(plan)
    if not plan["fits_scratch_estimate"]:
        raise ValueError(
            "Estimated temporary records plus annotation cache exceed the scratch cap; choose S3 shuffle storage or raise --scratch-gib"
        )
    if yes:
        return
    if (
        not sys.stdin.isatty()
        or input("Create this dataset and temporary shuffle records? Type yes: ")
        .strip()
        .lower()
        != "yes"
    ):
        raise ValueError(
            "Preparation requires explicit confirmation; use --yes for unattended execution"
        )


def run(args):
    from .storage import open_store

    if (args.existing or args.all_objects) and not args.check_only:
        raise ValueError("--existing/--all-objects apply only to --check-only")
    spec = settings(args)
    plan = describe(args, spec)
    if args.plan:
        show_plan(plan)
        result = plan
    elif args.check_only:
        from .validate import check

        result = check(
            open_store(args.output, create=False),
            spec,
            existing=args.existing,
            all_objects=args.all_objects,
        )
        print(json.dumps(result, indent=2))
    else:
        if args.abandon:
            if not (Path(plan["journal"]) / "journal.sqlite").is_file():
                raise ValueError("--abandon requires the existing journal")
            if not args.yes and (
                not sys.stdin.isatty()
                or input(
                    "Stop the writer first. Type yes to abandon this unfinished preparation: "
                )
                .strip()
                .lower()
                != "yes"
            ):
                raise ValueError("Abandon requires explicit confirmation")
        else:
            confirm(plan, args.yes)
        from .job import Job
        from .source import ingest
        from .validate import check

        with closing(
            Job(
                open_store(args.output),
                plan["scratch"],
                spec,
                journal=plan["journal"],
                abandon=args.abandon,
                spool_store=open_store(plan["shuffle_storage"]),
            )
        ) as job:
            if args.abandon:
                result = job.abandon()
            else:
                if not job.get("published"):
                    ingest(job)
                    job.convert()
                    job.publish()
                result = check(job.store, spec)
                if not args.keep_spool and not job.get("spool_cleaned"):
                    job.cleanup()
        print(json.dumps(result, indent=2))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(json_bytes(result))
    return result
