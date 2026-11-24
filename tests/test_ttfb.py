"""Timing boundaries, teardown and error reporting for first-batch latency."""

import csv
import json
from pathlib import Path

import pytest

from dataloader_arena import throughput, ttfb
from dataloader_arena.context import LoaderConfig, RunParameters


def config(tmp_path):
    return LoaderConfig("probe", RunParameters(tmp_path, None, 1, 0))


class Probe:
    def __init__(self, failure=None):
        self.failure = failure
        self.closed = False

    def __iter__(self):
        if self.failure == "iter":
            raise ValueError("iterator failed")
        return self

    def __next__(self):
        if self.failure == "empty":
            raise StopIteration
        if self.failure == "next":
            raise ValueError("read failed")
        return 1

    def close(self):
        self.closed = True


def test_ttfb_phases_exclude_metadata_and_teardown(tmp_path, monkeypatch):
    ticks = iter([0.0, 2.0, 5.0, 12.0])
    monkeypatch.setattr(ttfb.time, "perf_counter", lambda: next(ticks))
    loader = Probe()
    monkeypatch.setattr(ttfb, "_extract_resolved_config", lambda *_: {"resolved": True})
    result, resolved = ttfb.measure_ttfb(lambda _: loader, config(tmp_path))
    assert (result.factory_time, result.iter_time, result.first_next_time) == (2, 3, 7)
    assert result.total_ttfb == result.factory_time + result.iter_to_batch == 12
    assert resolved == {"resolved": True}
    assert loader.closed


@pytest.mark.parametrize(
    "metric,failure",
    [
        ("ttfb", "iter"),
        ("ttfb", "empty"),
        ("ttfb", "next"),
        ("throughput", "iter"),
        ("throughput", "next"),
    ],
)
def test_failed_measurement_closes_loader(tmp_path, failure, metric):
    loader = Probe(failure)
    with pytest.raises((ValueError, RuntimeError)):
        if metric == "ttfb":
            ttfb.measure_ttfb(lambda _: loader, config(tmp_path))
        else:
            throughput.run_benchmark(lambda _: loader, config(tmp_path), 0, 1)
    assert loader.closed


def test_ttfb_cli_failed_import_returns_nonzero_with_provenance(tmp_path):
    configuration = tmp_path / "loaders.json"
    configuration.write_text(
        json.dumps(
            {
                "loaders": [
                    {
                        "name": "unavailable",
                        "factory": "missing_loader_package:factory",
                        "config": {},
                    }
                ]
            }
        )
    )
    output = tmp_path / "out.json"
    code = ttfb.ttfb_app(
        [
            "--config",
            str(configuration),
            "--local",
            str(tmp_path),
            "--json-output",
            str(output),
            "--no-system-tuning",
            "--csv-output",
            str(tmp_path / "summary.csv"),
            "--per-trial-csv",
            str(tmp_path / "trials.csv"),
        ]
    )
    payload = json.loads(output.read_text())
    assert code == 1
    assert payload["results"] == []
    assert payload["failures"][0]["phase"] == "import"
    for filename, expected in [
        ("summary.csv", ttfb._BASE_TTFB_CSV_FIELDS),
        ("trials.csv", ttfb._BASE_TTFB_TRIAL_CSV_FIELDS),
    ]:
        with (tmp_path / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(expected)
            assert list(reader) == []


def test_minimal_example_is_shared_by_both_metrics():
    from dataloader_arena.harness import load_config

    plans = load_config(Path(__file__).parents[1] / "examples/loaders.json")
    assert {plan.name for plan in plans} == {"zephon", "litdata", "grain"}
