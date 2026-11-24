"""Exercise the existing CLI and timing iterator, with host tuning stubbed out."""

import csv
import json
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from dataloader_arena import throughput


@pytest.fixture
def invoke(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "loaders": [
                    {"name": "fixture", "factory": "fixture:factory", "config": {}}
                ]
            }
        )
    )
    output = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    argv = [
        "--config",
        str(config),
        "--local",
        str(tmp_path),
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--seed",
        "42",
        "--warmup",
        "8",
        "--steps",
        "12",
        "--no-progress",
        "--json-output",
        str(output),
        "--csv-output",
        str(csv_path),
    ]

    def run(factory, extra=("--strict",), environment=None, import_error=None):
        with (
            patch.object(throughput, "check_storage_path"),
            patch.object(throughput, "SystemStabilizer", side_effect=nullcontext),
            patch.object(throughput, "cleanup_between_runs"),
            patch.object(
                throughput,
                "import_factory",
                return_value=factory,
                side_effect=import_error,
            ),
            patch.object(
                throughput,
                "environment_snapshot",
                side_effect=environment
                or (lambda: {"free_threaded_build": False, "gil_enabled": True}),
            ),
        ):
            code = throughput.app(argv + list(extra))
        return (
            code,
            json.loads(output.read_text()) if output.exists() else None,
            csv_path,
        )

    return run


def test_cli_one_iterator_and_matching_json_csv(invoke):
    calls = []

    def factory(cfg):
        calls.append(cfg)
        return iter(range(5))

    code, payload, csv_path = invoke(factory)
    assert code == 0
    assert len(calls) == 1
    assert payload["failures"] == []
    observation = payload["results"][0]
    assert observation["result"]["steps"] == 3
    assert observation["csv_row"]["benchmark.warmup_steps"] == 2
    with csv_path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {k: str(v) for k, v in observation["csv_row"].items()}
        ]


@pytest.mark.parametrize(
    "failure", ["incomplete", "construction", "import", "unsupported"]
)
def test_strict_failures_are_nonzero_and_have_no_observations(invoke, failure):
    def factory(cfg):
        if failure == "construction":
            raise ValueError("construction failed")
        if failure == "unsupported":
            raise NotImplementedError("unsupported")
        return iter(range(3))

    code, payload, _ = invoke(
        factory, import_error=ImportError("missing") if failure == "import" else None
    )
    assert code == 1
    assert payload["results"] == []
    assert payload["failures"]


def test_non_strict_cli_allows_partial_results(invoke):
    code, payload, _ = invoke(lambda cfg: iter(range(3)), extra=())
    assert code == 0
    assert payload["results"][0]["result"]["steps"] == 1


@pytest.mark.parametrize("before,after", [(False, True), (True, False)])
def test_require_no_gil_checks_before_and_after_loading(invoke, before, after):
    states = iter(
        [
            {"free_threaded_build": before, "gil_enabled": not before},
            {"free_threaded_build": after, "gil_enabled": not after},
        ]
    )
    code, payload, _ = invoke(
        lambda cfg: iter(range(5)),
        extra=("--strict", "--require-no-gil"),
        environment=lambda: next(states),
    )
    assert code == 1
    assert payload["results"] == []
    assert "GIL disabled" in payload["failures"][0]["error"]


def test_json_refuses_overwrite_before_work(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("original")
    with (
        patch.object(throughput, "import_factory") as factory,
        pytest.raises(SystemExit),
    ):
        throughput.app(["--local", str(tmp_path), "--json-output", str(output)])
    factory.assert_not_called()
    assert output.read_text() == "original"


def test_strict_rejects_nonpositive_duration(invoke):
    result = throughput.BenchmarkResult("fixture", 3, 3, float("nan"), 0, 4, 42)
    with patch.object(throughput, "run_benchmark", return_value=(result, {})):
        code, payload, _ = invoke(lambda cfg: iter(range(5)))
    assert code == 1
    assert payload["results"] == []


def test_strict_skip_is_failure(invoke):
    with patch.object(
        throughput,
        "_should_skip_loader",
        return_value=(True, "unsupported combination"),
    ):
        code, payload, _ = invoke(lambda cfg: iter(range(5)))
    assert code == 1
    assert payload["results"] == []
    assert "unsupported combination" in payload["failures"][0]["error"]


def test_json_supports_multiple_normal_cli_results(invoke, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "loaders": [
                    {"name": name, "factory": "fixture:factory", "config": {}}
                    for name in ("first", "second")
                ]
            }
        )
    )
    code, payload, csv_path = invoke(lambda cfg: iter(range(5)))
    assert code == 0
    assert [entry["result"]["name"] for entry in payload["results"]] == [
        "first",
        "second",
    ]
    with csv_path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_strict_packing_requires_usable_tokens(invoke):
    result = throughput.BenchmarkResult("fixture", 3, 3, 1.0, 0, 4, 42)
    with patch.object(throughput, "run_benchmark", return_value=(result, {})):
        code, payload, _ = invoke(
            lambda cfg: iter(range(5)), extra=("--strict", "--packing-stats")
        )
    assert code == 1
    assert "usable tokens" in payload["failures"][0]["error"]


def test_strict_limit_uses_effective_measurement_target(invoke):
    code, payload, _ = invoke(
        lambda cfg: iter(range(20)), extra=("--strict", "--limit-steps", "4")
    )
    assert code == 0
    observation = payload["results"][0]
    assert observation["result"]["steps"] == 2
    assert observation["csv_row"]["benchmark.measure_steps"] == 2


def test_empty_loader_fails_without_strict(invoke):
    code, payload, _ = invoke(lambda cfg: iter(()), extra=())
    assert code == 1
    assert payload["results"] == []
    assert "No batches" in payload["failures"][0]["error"]


def test_csv_refuses_overwrite_before_loading(tmp_path):
    output = tmp_path / "result.csv"
    output.write_text("original")
    with (
        patch.object(throughput, "import_factory") as factory,
        pytest.raises(SystemExit),
    ):
        throughput.app(["--local", str(tmp_path), "--csv-output", str(output)])
    factory.assert_not_called()
    assert output.read_text() == "original"
