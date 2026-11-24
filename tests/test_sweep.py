import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "arena_sweep", Path(__file__).resolve().parents[1] / "sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


@pytest.mark.parametrize("no_sync", [False, True])
def test_prepared_mode_never_syncs_and_preserves_default_sync(tmp_path, no_sync):
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin/python").touch()
    cfg = sweep.SweepConfig(
        ["3.14t"],
        [],
        tmp_path / "benchmark.json",
        uv_project_environment_template=str(env),
        force_gil_off=True,
    )
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        Path(command[command.index("--csv-output") + 1]).write_text(
            "loader_name\nfixture\n"
        )

        class Process:
            stdout = io.StringIO("ok\n")
            stderr = io.StringIO("")

            def wait(self):
                return 0

        return Process()

    with (
        patch.object(sweep, "run") as sync,
        patch("subprocess.Popen", side_effect=popen),
    ):
        result = sweep.run_bench(
            "3.14t",
            sweep.RunSpec("fixture", local="data"),
            cfg,
            tmp_path / "results",
            False,
            False,
            False,
            no_sync=no_sync,
            dont_cleanup_local=True,
        )
    assert result["ok"]
    assert sync.call_count == (0 if no_sync else 1)
    command, kwargs = commands[0]
    assert command[:4] == ["uv", "run", "--python", "3.14t"]
    assert "--no-sync" in command
    assert "--dont-cleanup-local" in command
    assert "--json-output" in command
    assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(env)
    assert kwargs["env"]["PYTHON_GIL"] == "0"
    assert "PYTHONPATH" not in kwargs["env"]
    assert "proc_env" not in json.loads(Path(result["meta"]).read_text())


def test_no_sync_requires_prepared_environment(tmp_path):
    cfg = sweep.SweepConfig(
        ["3.13"],
        [],
        tmp_path / "benchmark.json",
        uv_project_environment_template=str(tmp_path / "missing"),
    )
    with (
        patch.object(sweep, "run") as sync,
        pytest.raises(ValueError, match="existing environment"),
    ):
        sweep.run_bench(
            "3.13",
            sweep.RunSpec("fixture"),
            cfg,
            tmp_path / "results",
            False,
            False,
            False,
            no_sync=True,
        )
    sync.assert_not_called()


def test_no_sync_skips_interpreter_preinstall(tmp_path):
    cfg = sweep.SweepConfig(
        ["3.13"], [sweep.RunSpec("fixture")], tmp_path / "benchmark.json", locked=False
    )
    with (
        patch.object(
            sys,
            "argv",
            [
                "sweep.py",
                "--config",
                "unused",
                "--results-dir",
                str(tmp_path),
                "--no-sync",
            ],
        ),
        patch.object(sweep, "ensure_uv_available"),
        patch.object(sweep, "load_config", return_value={}),
        patch.object(sweep, "validate_and_coerce", return_value=cfg),
        patch.object(sweep, "preinstall_pythons") as preinstall,
        patch.object(sweep, "run_bench", return_value={"ok": True}) as run,
    ):
        assert sweep.main() == 0
    preinstall.assert_not_called()
    assert run.call_args.kwargs["no_sync"]
    assert run.call_args.kwargs["unlocked"]


@pytest.mark.parametrize("output_arg", ["--json-output=shared.json", "--per-trial-csv"])
def test_sweep_owns_all_output_paths(tmp_path, output_arg):
    bench_config = tmp_path / "loaders.json"
    bench_config.write_text("{}")
    with pytest.raises(SystemExit, match="bench_args must NOT include"):
        sweep.validate_and_coerce(
            {
                "python_versions": ["3.13"],
                "runs": [{"name": "example", "local": "data"}],
                "bench_config": str(bench_config),
                "bench_args": [output_arg],
            }
        )


def test_header_only_csv_does_not_count_as_success(tmp_path):
    cfg = sweep.SweepConfig(
        ["3.13"], [], tmp_path / "benchmark.json", bench_command="dataloader-ttfb"
    )

    def popen(command, **kwargs):
        Path(command[command.index("--csv-output") + 1]).write_text("loader_name\n")
        assert "--per-trial-csv" in command
        assert "--json-output" in command

        class Process:
            stdout = io.StringIO("")
            stderr = io.StringIO("")

            def wait(self):
                return 0

        return Process()

    with patch.object(sweep, "run"), patch("subprocess.Popen", side_effect=popen):
        result = sweep.run_bench(
            "3.13",
            sweep.RunSpec("empty"),
            cfg,
            tmp_path / "results",
            False,
            False,
            False,
        )
    assert not result["ok"]
