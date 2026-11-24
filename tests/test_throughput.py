import argparse
import csv
import json
import os
from typing import Iterable, Mapping

import numpy as np
import pytest
import torch
from litdata import optimize

from dataloader_arena import harness, throughput
from dataloader_arena.benchmarks.load_and_batch.litdata import (
    create_loader as litdata_loader,
)
from dataloader_arena.benchmarks.load_and_batch.zephon import (
    create_loader as zephon_loader,
)
from dataloader_arena.context import (
    BenchmarkConfig,
    LoaderConfig,
    LoaderPlan,
    RunParameters,
)


def test_load_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        '{"loaders": [{"name": "example", "factory": "math:sqrt", "config": {"foo": 1}}]}'
    )

    specs = harness.load_config(config)

    assert len(specs) == 1
    assert specs[0].name == "example"
    assert specs[0].factory == "math:sqrt"
    assert specs[0].config == {"foo": 1}


def test_format_results():
    result = throughput.BenchmarkResult(
        name="demo",
        steps=10,
        processed=10,
        total_time=1.0,
        num_workers=4,
        batch_size=32,
        seed=None,
    )
    table = throughput.format_results([result])
    assert "demo" in table
    assert "10" in table


def test_normalize_multi_values_flattens_and_defaults():
    values = harness._normalize_multi_values([[16], [32, 64]], default=(8,))
    assert values == [16, 32, 64]
    assert harness._normalize_multi_values(None, default=(8,)) == [8]


def test_normalize_multi_values_preserves_none_entries():
    values = harness._normalize_multi_values([[None], [1]], default=(None,))
    assert values == [None, 1]


def test_seed_option_parsing_and_errors():
    assert harness._seed_option("none") is None
    assert harness._seed_option("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        harness._seed_option("invalid")


def test_batch_option_parsing_and_errors():
    assert harness._batch_option("none") is None
    assert harness._batch_option("4") == 4
    with pytest.raises(argparse.ArgumentTypeError):
        harness._batch_option("0")
    with pytest.raises(argparse.ArgumentTypeError):
        harness._batch_option("oops")


def test_scaled_steps_respects_batch_and_minimum():
    assert throughput._scaled_steps(512, 4, min_steps=None) == 128
    assert throughput._scaled_steps(32, None, min_steps=None) == 32
    assert throughput._scaled_steps(10, 3, min_steps=5) == 5
    assert throughput._scaled_steps(0, 8, min_steps=3) == 3


def test_loader_stubs_error_for_missing_path(tmp_path):
    run = RunParameters(
        local_path=tmp_path / "missing",
        remote_path=None,
        batch_size=4,
        num_workers=2,
        seed=123,
        limit_steps=None,
    )
    cfg = LoaderConfig(loader_name="litdata", run=run, config={})

    for factory in (litdata_loader, zephon_loader):
        try:
            factory(cfg)
            raised = False
        except FileNotFoundError:
            raised = True
        assert raised, f"{factory.__name__} did not validate missing path"


_SYNTHETIC_SAMPLES = [{"value": idx, "text": f"sample-{idx}"} for idx in range(8)]


def _emit_sample(sample: Mapping[str, object]) -> Mapping[str, object]:
    return sample


def _to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return value


def _extract_values(batch) -> list[int]:
    if isinstance(batch, Mapping):
        values = batch["value"]
        if isinstance(values, torch.Tensor):
            return [int(v) for v in values.tolist()]
        if isinstance(values, np.ndarray):
            return [int(v) for v in values.tolist()]
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            return [int(_to_python(v)) for v in values]
        return [int(_to_python(values))]

    records = getattr(batch, "records", None)
    if records is not None:
        return [int(_to_python(rec.payload["value"])) for rec in records]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def test_optimize_generates_dataset_loadable_by_all_loaders(tmp_path):
    os.environ.setdefault("LITDATA_DISABLE_PROGRESS", "1")

    dataset_dir = tmp_path / "litdata-shards"
    dataset_dir.mkdir()

    optimize(
        _emit_sample,
        inputs=_SYNTHETIC_SAMPLES,
        output_dir=str(dataset_dir),
        chunk_size=4,
        num_workers=1,
        verbose=False,
    )

    expected_values = [sample["value"] for sample in _SYNTHETIC_SAMPLES]
    batch_size = 2

    loader_specs = {
        "litdata": (
            litdata_loader,
            {"shuffle": False},
        ),
        "zephon": (
            zephon_loader,
            {"drop_last": False},
        ),
    }

    for name, (factory, extra) in loader_specs.items():
        run = RunParameters(
            local_path=dataset_dir,
            remote_path=None,
            batch_size=batch_size,
            num_workers=0,
            seed=123,
            limit_steps=None,
        )
        cfg = LoaderConfig(loader_name=name, run=run, config=extra)
        loader = factory(cfg)
        values = []
        for batch in loader:
            values.extend(_extract_values(batch))
            if len(values) >= len(expected_values):
                break

        assert values[: len(expected_values)] == expected_values, (
            f"{name} loader failed to read dataset"
        )


def test_prepare_csv_row_includes_config_columns(tmp_path):
    run = RunParameters(
        local_path=tmp_path,
        remote_path="s3://remote-dataset",
        batch_size=8,
        num_workers=4,
        seed=77,
        limit_steps=10,
        verbose=True,
    )
    loader_cfg = LoaderConfig(
        loader_name="demo",
        run=run,
        config={"alpha": 123, "nested": {"beta": True}},
    )
    bench_cfg = BenchmarkConfig(
        run=run,
        warmup_steps=3,
        measure_steps=5,
        loader_plans=(),
    )
    result = throughput.BenchmarkResult(
        name="demo",
        steps=5,
        processed=5,
        total_time=2.0,
        num_workers=run.num_workers,
        batch_size=run.batch_size,
        seed=run.seed,
    )

    resolved = {"alpha": 456, "nested": {"beta": False}}

    row = throughput._prepare_csv_row(result, loader_cfg, bench_cfg, resolved)

    assert row["loader_name"] == "demo"
    assert row["benchmark.warmup_steps"] == 3
    raw_config = json.loads(row["config.plaintext"])
    resolved_config = json.loads(row["config.resolved.plaintext"])
    assert raw_config["alpha"] == 123
    assert raw_config["nested"]["beta"] is True
    assert resolved_config["alpha"] == 456
    assert resolved_config["nested"]["beta"] is False
    assert row["run.remote_path"] == "s3://remote-dataset"


def test_order_zephon_factories_prioritizes_process_runner():
    plan_threads = LoaderPlan(
        name="zephon-threads",
        factory="dataloader_arena.benchmarks.vlm.zephon:create_loader",
        config={"runner": "threads"},
    )
    plan_process = LoaderPlan(
        name="zephon-process",
        factory="dataloader_arena.benchmarks.vlm.zephon:create_loader",
        config={"runner": "process"},
    )
    plan_other = LoaderPlan(
        name="other-loader",
        factory="dataloader_arena.benchmarks.load_and_batch.litdata:create_loader",
        config={},
    )

    def _factory(cfg):
        return cfg

    ordered = harness._order_zephon_factories(
        [
            (plan_threads, _factory),
            (plan_other, _factory),
            (plan_process, _factory),
        ]
    )

    assert [plan.name for plan, _ in ordered] == [
        "other-loader",
        "zephon-process",
        "zephon-threads",
    ]


def test_is_zephon_process_uses_top_level_runner():
    plan_direct = LoaderPlan(
        name="zephon-process-direct-runner",
        factory="dataloader_arena.benchmarks.vlm.zephon:create_loader",
        config={"runner": "process"},
    )
    plan_threads = LoaderPlan(
        name="zephon-threads",
        factory="dataloader_arena.benchmarks.vlm.zephon:create_loader",
        config={"runner": "threads"},
    )
    plan_other = LoaderPlan(
        name="litdata",
        factory="dataloader_arena.benchmarks.load_and_batch.litdata:create_loader",
        config={},
    )

    assert harness._is_zephon_process(plan_direct) is True
    assert harness._is_zephon_process(plan_threads) is False
    assert harness._is_zephon_process(plan_other) is False


def test_prioritize_worker_combinations_runs_workers_first():
    combos = [
        (1, 0, None),
        (1, 1, 42),
        (2, 2, None),
        (1, 0, 123),
    ]

    ordered = harness._prioritize_worker_combinations(combos)

    assert ordered == [
        (1, 1, 42),
        (2, 2, None),
        (1, 0, None),
        (1, 0, 123),
    ]


def test_prepare_csv_row_handles_none_batch_size(tmp_path):
    run = RunParameters(
        local_path=tmp_path,
        remote_path=None,
        batch_size=None,
        num_workers=2,
        seed=None,
        limit_steps=None,
        verbose=False,
    )
    loader_cfg = LoaderConfig(loader_name="demo", run=run, config={})
    bench_cfg = BenchmarkConfig(
        run=run, warmup_steps=0, measure_steps=0, loader_plans=()
    )
    result = throughput.BenchmarkResult(
        name="demo",
        steps=0,
        processed=0,
        total_time=0.0,
        num_workers=run.num_workers,
        batch_size=run.batch_size,
        seed=run.seed,
    )

    row = throughput._prepare_csv_row(result, loader_cfg, bench_cfg, {})

    assert row["run.batch_size"] == ""
    assert row["run.num_workers"] == 2


def test_write_csv_output_creates_file_with_header(tmp_path):
    run = RunParameters(
        local_path=tmp_path,
        remote_path=None,
        batch_size=8,
        num_workers=4,
        seed=None,
        limit_steps=None,
        verbose=False,
    )
    loader_cfg = LoaderConfig(
        loader_name="writer",
        run=run,
        config={"foo": "bar"},
    )
    bench_cfg = BenchmarkConfig(
        run=run,
        warmup_steps=1,
        measure_steps=2,
        loader_plans=(),
    )
    result = throughput.BenchmarkResult(
        name="writer",
        steps=2,
        processed=2,
        total_time=1.0,
        num_workers=run.num_workers,
        batch_size=run.batch_size,
        seed=run.seed,
    )
    resolved = {"foo": "bar"}

    row = throughput._prepare_csv_row(result, loader_cfg, bench_cfg, resolved)

    output = tmp_path / "results.csv"
    harness._write_csv_output(output, [row], base_fields=throughput._BASE_CSV_FIELDS)

    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "loader_name" in reader.fieldnames
    assert "config.plaintext" in reader.fieldnames
    assert "config.resolved.plaintext" in reader.fieldnames
    assert rows[0]["loader_name"] == "writer"
    assert json.loads(rows[0]["config.plaintext"])["foo"] == "bar"
    assert json.loads(rows[0]["config.resolved.plaintext"])["foo"] == "bar"
