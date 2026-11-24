#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import shutil
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Any, Optional

# ---------------------------
# Dataclasses (typed config)
# ---------------------------


@dataclass
class RunSpec:
    """One (local, remote) scenario to benchmark."""

    name: str
    local: Optional[str] = None  # path or None
    remote: Optional[str] = None  # URI or None


@dataclass
class SweepConfig:
    """Top-level sweep configuration loaded from JSON."""

    python_versions: list[str]
    runs: list[RunSpec]
    bench_config: Path
    bench_args: list[str] = field(default_factory=list)
    locked: bool = True
    preinstall: bool = True
    uv_project_environment_template: str = ".venv-{pyver}"
    force_gil_off: bool = False
    bench_command: str = "dataloader-bench"
    loaders_default: list[str] = field(default_factory=list)
    loaders_by_python: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------
# Helpers
# ---------------------------

FORBIDDEN_BENCH_ARGS = {
    "--csv-output",
    "--json-output",
    "--per-trial-csv",
    # We manage CSV outputs.
    "--local",
    "--remote",  # we manage dataset endpoints
    "--config",  # we supply bench_config
}


def shjoin(argv: list[str]) -> str:
    return shlex.join(argv)


def sanitize(s: str) -> str:
    s = re.sub(r"[^\w.+-]+", "-", s.strip())
    return re.sub(r"-{2,}", "-", s).strip("-") or "x"


def now_stamp() -> str:
    # Include milliseconds to avoid collisions when multiple runs start within the same second.
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def ensure_uv_available() -> None:
    try:
        run(["uv", "--version"], check=True, capture_output=True, text=True)
    except Exception:
        sys.exit("`uv` not found in PATH. Install uv: https://docs.astral.sh/uv/")


def _purge_directory_contents(dir_path: Path) -> None:
    """
    Remove all files/subdirs inside dir_path, but keep dir_path itself.
    Includes basic safety guards to avoid catastrophic deletions.
    """
    p = dir_path.expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return

    # Safety rails: refuse to clean obviously-dangerous targets
    dangerous = {Path("/").resolve(), Path.home().resolve()}
    if p in dangerous:
        print(f"[warn] refusing to clean critical path: {p}", file=sys.stderr)
        return
    # Also require the path to be at least a few levels deep like /a/b/c
    if len(p.parts) < 3:
        print(f"[warn] refusing to clean shallow path: {p}", file=sys.stderr)
        return

    for child in p.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except Exception as e:
            print(f"[warn] failed to remove {child}: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sequential sweep runner for dataloader-bench via uv"
    )
    ap.add_argument("--config", required=True, help="Path to sweep.config.json")
    ap.add_argument(
        "--results-dir", required=True, help="Directory for per-run CSVs and logs"
    )
    ap.add_argument("--merge-csv", default=None, help="Optional path for merged CSV")
    ap.add_argument(
        "--dont-cleanup-local",
        action="store_true",
        help="Do NOT purge the local cache directory after runs where both local and remote are set",
    )
    ap.add_argument(
        "--show-live",
        action="store_true",
        help="Stream child stdout/stderr live while also writing to log files",
    )

    # Reproducibility knobs (defaults: locked semantics)
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--frozen",
        action="store_true",
        help="Use uv in frozen mode (use lockfile as-is; don't check freshness)",
    )
    g.add_argument(
        "--unlocked",
        action="store_true",
        help="Allow uv to relock (not recommended for reproducible benches)",
    )
    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="Use an existing environment without installing Python or syncing dependencies",
    )
    return ap.parse_args()


def load_config(json_path: Path) -> dict[str, Any]:
    try:
        return json.loads(json_path.read_text())
    except Exception as e:
        sys.exit(f"Failed to read JSON config {json_path}: {e}")


def _tee_to_file_and_optional_sink(stream, log_path: Path, sink=None) -> None:
    """
    Read a text stream line-by-line, write to log file, and optionally
    also write to a sink (e.g., sys.stdout/sys.stderr).
    """
    assert stream is not None
    with log_path.open("w", encoding="utf-8", newline="") as f:
        for line in iter(stream.readline, ""):
            f.write(line)
            f.flush()
            if sink is not None:
                sink.write(line)
                sink.flush()
    stream.close()


def validate_and_coerce(raw: dict[str, Any]) -> SweepConfig:
    # Disallow CLI-owned keys in JSON
    for forbidden in ("results_dir", "merge_csv"):
        if forbidden in raw:
            sys.exit(f"'{forbidden}' must be provided via CLI, not in JSON")

    # Required keys
    for key in ("python_versions", "runs", "bench_config", "bench_args"):
        if key not in raw:
            sys.exit(f"Config missing required key: {key}")

    # Python versions
    pythons = raw["python_versions"]
    if not isinstance(pythons, list) or not pythons:
        sys.exit("python_versions must be a non-empty list of strings")
    for v in pythons:
        if not isinstance(v, str) or not re.match(r"^\d+\.\d+(?:t)?$", v):
            sys.exit(f"Invalid python version spec: {v!r} (e.g. '3.12', '3.13t')")

    # Runs
    runs_raw = raw["runs"]
    if not isinstance(runs_raw, list) or not runs_raw:
        sys.exit("runs must be a non-empty list of {name, local?, remote?}")
    runs: list[RunSpec] = []
    for r in runs_raw:
        if "name" not in r or not isinstance(r["name"], str) or not r["name"].strip():
            sys.exit("each run needs a non-empty 'name'")
        if "local" in r and r["local"] is not None and not isinstance(r["local"], str):
            sys.exit(f"run {r['name']}: 'local' must be a string or null")
        if (
            "remote" in r
            and r["remote"] is not None
            and not isinstance(r["remote"], str)
        ):
            sys.exit(f"run {r['name']}: 'remote' must be a string or null")
        runs.append(
            RunSpec(
                name=r["name"].strip(), local=r.get("local"), remote=r.get("remote")
            )
        )

    # Bench config file
    bench_cfg = Path(raw["bench_config"]).expanduser()
    if not bench_cfg.exists():
        sys.exit(f"bench_config not found: {bench_cfg}")

    # Bench args
    bench_args = raw["bench_args"]
    if not isinstance(bench_args, list) or not all(
        isinstance(x, str) for x in bench_args
    ):
        sys.exit("bench_args must be a list of strings")
    overlaps = sorted(
        {a for a in bench_args if a.split("=", 1)[0] in FORBIDDEN_BENCH_ARGS}
    )
    if overlaps:
        sys.exit(
            f"bench_args must NOT include {sorted(FORBIDDEN_BENCH_ARGS)}; found {overlaps}"
        )

    # Optional defaults from JSON
    locked = bool(raw.get("locked", True))
    preinstall = bool(raw.get("preinstall", True))
    force_gil_off = bool(raw.get("force_gil_off", False))
    tpl = str(raw.get("uv_project_environment_template", ".venv-{pyver}"))

    loaders_default = raw.get("loaders_default", [])
    if not isinstance(loaders_default, list) or not all(
        isinstance(x, str) for x in loaders_default
    ):
        sys.exit("loaders_default must be a list of strings")

    loaders_by_python = raw.get("loaders_by_python", {})
    if not isinstance(loaders_by_python, dict) or not all(
        isinstance(k, str)
        and isinstance(v, list)
        and all(isinstance(x, str) for x in v)
        for k, v in loaders_by_python.items()
    ):
        sys.exit("loaders_by_python must be a mapping of python_version -> list[str]")

    bench_command = str(raw.get("bench_command", "dataloader-bench"))

    return SweepConfig(
        python_versions=pythons,
        runs=runs,
        bench_config=bench_cfg,
        bench_args=bench_args,
        locked=locked,
        preinstall=preinstall,
        force_gil_off=force_gil_off,
        uv_project_environment_template=tpl,
        loaders_by_python=loaders_by_python,
        loaders_default=loaders_default,
        bench_command=bench_command,
    )


def extras_for_python(cfg: SweepConfig, pyver: str) -> list[str]:
    requested = cfg.loaders_by_python.get(pyver, cfg.loaders_default)
    return list(dict.fromkeys(requested))


def preinstall_pythons(versions: list[str]) -> None:
    if not versions:
        return
    cmd = ["uv", "python", "install", *versions]
    print(f"[setup] {shjoin(cmd)}")
    try:
        run(cmd, check=True)
    except CalledProcessError as e:
        print(f"[warn] preinstall failed (continuing): {e}", file=sys.stderr)


def build_env(
    cfg: SweepConfig, pyver: str, frozen: bool, unlocked: bool
) -> dict[str, str]:
    env: dict[str, str] = {}
    if frozen:
        env["UV_FROZEN"] = "1"
    elif not unlocked:
        # default: locked semantics for reproducibility
        env["UV_LOCKED"] = "1"
    env["UV_PROJECT_ENVIRONMENT"] = cfg.uv_project_environment_template.format(
        pyver=sanitize(pyver)
    )
    return env


def run_bench(
    pyver: str,
    rs: RunSpec,
    cfg: SweepConfig,
    results_dir: Path,
    frozen: bool,
    unlocked: bool,
    show_live: bool,
    no_sync: bool = False,
    dont_cleanup_local: bool = False,
) -> dict[str, Any]:
    # Layout
    tag_py = sanitize(pyver)
    tag_run = sanitize(rs.name)
    csv_dir = results_dir / "csv"
    log_dir = results_dir / "logs" / f"{tag_run}__{tag_py}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = now_stamp()
    csv_path = csv_dir / f"{tag_run}__{tag_py}__{stamp}.csv"
    json_path = csv_path.with_suffix(".json")
    trial_path = csv_path.with_name(csv_path.stem + "-trials.csv")
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    meta_path = log_dir / "meta.json"
    selected_extras = extras_for_python(cfg, pyver)
    pre_sync = not no_sync

    env = build_env(cfg, pyver, frozen=frozen, unlocked=unlocked)
    gil_off = cfg.force_gil_off and pyver.endswith("t")
    proc_env = os.environ.copy()
    # cleanup whatever venv we were using from the parent -- we want a clean venv for the uv child!
    for k in (
        "VIRTUAL_ENV",  # generic venv activation
        "CONDA_PREFIX",  # conda/mamba activation
        "PYENV_VERSION",  # pyenv shims
        "PYTHONHOME",  # can force wrong stdlib
        "PYTHONPATH",  # can inject wrong packages
        "PDM_VENV",  # pdm activated venv
        "POETRY_ACTIVE",  # poetry activation flag
        "UV_LOCKED",
        "UV_FROZEN",
    ):
        proc_env.pop(k, None)

    proc_env.update(env)
    if gil_off:
        proc_env["PYTHON_GIL"] = "0"
    if show_live:
        proc_env.setdefault("PYTHONUNBUFFERED", "1")

    if no_sync:
        target = Path(env["UV_PROJECT_ENVIRONMENT"]).expanduser()
        if (
            not (target / "bin/python").is_file()
            and not (target / "Scripts/python.exe").is_file()
        ):
            raise ValueError(f"--no-sync requires an existing environment: {target}")
        proc_env["UV_PYTHON_DOWNLOADS"] = "never"

    if pre_sync:
        sync_cmd = ["uv", "sync", "--python", pyver]
        if frozen:
            sync_cmd.append("--frozen")
        elif not unlocked:
            sync_cmd.append("--locked")
        for x in selected_extras:
            sync_cmd += ["--extra", x]
        print(f"[sync] {shjoin(sync_cmd)}")
        try:
            run(sync_cmd, check=True, env=proc_env)
        except CalledProcessError as e:
            print(
                f"[fail] sync failed for py={pyver} extras={selected_extras}: {e}",
                file=sys.stderr,
            )
            # Don't abort the whole sweep: mark this run as failed and continue
            return {
                "ok": False,
                "rc": e.returncode,
                "csv": None,
                "stdout": str(stdout_log),
                "stderr": str(stderr_log),
                "meta": str(meta_path),
                "py": pyver,
                "name": rs.name,
                "local": rs.local,
                "remote": rs.remote,
                "started": stamp,
            }

    cmd: list[str] = ["uv", "run", "--python", pyver]
    if pre_sync or no_sync:
        cmd.append("--no-sync")  # use the environment prepared by us or the caller
    cmd += [cfg.bench_command, "--config", str(cfg.bench_config)]
    if rs.local:
        cmd += ["--local", rs.local]
    if rs.remote:
        cmd += ["--remote", rs.remote]
    cmd += [
        *cfg.bench_args,
        "--csv-output",
        str(csv_path),
        "--json-output",
        str(json_path),
    ]
    if cfg.bench_command == "dataloader-ttfb":
        cmd += ["--per-trial-csv", str(trial_path)]
    if dont_cleanup_local:
        cmd.append("--dont-cleanup-local")

    env_snapshot = {
        k: proc_env[k]
        for k in ("UV_PROJECT_ENVIRONMENT", "UV_LOCKED", "UV_FROZEN")
        if k in proc_env
    }
    if gil_off and "PYTHON_GIL" in proc_env:
        env_snapshot["PYTHON_GIL"] = proc_env["PYTHON_GIL"]

    # Persist reproducibility settings, never the entire inherited environment.
    meta = {
        "python": pyver,
        "run_name": rs.name,
        "local": rs.local,
        "remote": rs.remote,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "command": cmd,
        "env": env_snapshot,
        "timestamp": stamp,
        "extras": selected_extras,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n[run] {rs.name} | py={pyver}")
    print(f"[cmd] {shjoin(cmd)}")
    print(f"[env] {meta['env']}")

    from subprocess import PIPE, Popen

    proc = Popen(
        cmd,
        env=proc_env,
        text=True,
        stdout=PIPE,
        stderr=PIPE,
        bufsize=1,  # line-buffered in text mode
    )

    # Always write to log files; mirror to console only when --show-live
    t_out = threading.Thread(
        target=_tee_to_file_and_optional_sink,
        args=(proc.stdout, stdout_log, sys.stdout if show_live else None),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_tee_to_file_and_optional_sink,
        args=(proc.stderr, stderr_log, sys.stderr if show_live else None),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    rc = proc.wait()
    t_out.join()
    t_err.join()

    has_rows = False
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            has_rows = next(csv.DictReader(handle), None) is not None
    ok = rc == 0 and has_rows
    if not ok:
        print(f"[fail] rc={rc}  see {stderr_log}", file=sys.stderr)
    return {
        "ok": ok,
        "rc": rc,
        "csv": str(csv_path) if csv_path.exists() else None,
        "stdout": str(stdout_log),
        "stderr": str(stderr_log),
        "meta": str(meta_path),
        "py": pyver,
        "name": rs.name,
        "local": rs.local,
        "remote": rs.remote,
        "started": stamp,
    }


def preview_merged_csv(merged_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(merged_path)

    # Detect TTFB vs throughput CSV by column presence
    is_ttfb = "result.ttfb_median_s" in df.columns

    if is_ttfb:
        required = [
            "sweep_python",
            "sweep_run_name",
            "loader_name",
            "run.batch_size",
            "run.num_workers",
            "run.seed",
            "result.ttfb_median_s",
            "result.ttfb_std_s",
        ]
        cols_map = {
            "sweep_python": "python version",
            "sweep_run_name": "run_name",
            "loader_name": "loader",
            "run.batch_size": "batch",
            "run.num_workers": "workers",
            "run.seed": "seed",
            "result.ttfb_median_s": "ttfb_med(s)",
            "result.ttfb_std_s": "ttfb_std(s)",
        }
        numeric_cols = ["ttfb_med(s)", "ttfb_std(s)"]
    else:
        required = [
            "sweep_python",
            "sweep_run_name",
            "loader_name",
            "run.batch_size",
            "run.num_workers",
            "run.seed",
            "result.it_per_second",
            "result.samples_per_second",
        ]
        cols_map = {
            "sweep_python": "python version",
            "sweep_run_name": "run_name",
            "loader_name": "loader",
            "run.batch_size": "batch",
            "run.num_workers": "workers",
            "run.seed": "seed",
            "result.it_per_second": "it/s",
            "result.samples_per_second": "sample/s",
        }
        numeric_cols = ["it/s", "sample/s"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            f"[preview] Missing expected columns in merged CSV: {missing}",
            file=sys.stderr,
        )
        return

    view = df[list(cols_map.keys())].copy()
    view.rename(columns=cols_map, inplace=True)

    for c in numeric_cols:
        view[c] = pd.to_numeric(view[c], errors="coerce").round(3 if is_ttfb else 2)

    sort_keys = ["python version", "run_name", "loader", "batch", "workers", "seed"]
    view.sort_values(sort_keys, kind="stable", inplace=True, ignore_index=True)

    with pd.option_context(
        "display.max_columns",
        len(view.columns),
        "display.width",
        None,
        "display.max_colwidth",
        40,
    ):
        print(view.to_string(index=False))


def merge_csvs(
    csv_paths: list[Path], merged_out: Path, rows_meta: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Merge with extra sweep_* columns.
    Fail if *source* headers (pre-augmentation) don't all match.
    """
    import pandas as pd

    if not csv_paths:
        return {"ok": False, "reason": "no CSVs to merge"}
    # First, check header equality
    src_headers = []
    for p in csv_paths:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                src_headers.append(tuple(next(reader)))
            except StopIteration:
                return {"ok": False, "reason": f"empty CSV: {p}", "offender": str(p)}
    if len({h for h in src_headers}) != 1:
        return {
            "ok": False,
            "reason": "header mismatch",
            "offender": "see individual CSVs",
        }

    frames = []
    for p, meta in zip(csv_paths, rows_meta):
        df = pd.read_csv(p)
        df.insert(0, "sweep_python", meta["py"])
        df.insert(1, "sweep_run_name", meta["name"])
        df.insert(2, "sweep_local", meta["local"])
        df.insert(3, "sweep_remote", meta["remote"])
        df.insert(4, "sweep_csv_path", str(p))
        df.insert(5, "sweep_started_at", meta["started"])
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_out, index=False)
    return {"ok": True, "merged_path": str(merged_out), "rows": int(merged.shape[0])}


# ---------------------------
# Main (sequential)
# ---------------------------


def main() -> int:
    args = parse_args()
    ensure_uv_available()

    cfg_dict = load_config(Path(args.config).expanduser())
    cfg = validate_and_coerce(cfg_dict)

    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine locking mode
    frozen = bool(args.frozen)
    unlocked = bool(args.unlocked or (not cfg.locked and not frozen))
    if frozen and unlocked:
        sys.exit("Cannot combine --frozen and --unlocked")

    # Preinstall interpreters for determinism & speed
    if cfg.preinstall and not args.no_sync:
        # Add a tiny dedup in case of duplicates
        versions = sorted(set(cfg.python_versions))
        preinstall_pythons(versions)

    all_results: list[dict[str, Any]] = []
    for pyver in cfg.python_versions:
        for rs in cfg.runs:
            res = run_bench(
                pyver,
                rs,
                cfg,
                results_dir,
                frozen=frozen,
                unlocked=unlocked,
                show_live=args.show_live,
                no_sync=args.no_sync,
                dont_cleanup_local=args.dont_cleanup_local,
            )
            all_results.append(res)
            if rs.local and rs.remote and not args.dont_cleanup_local:
                lp = Path(rs.local)
                print(f"[cleanup] purging local cache contents: {lp}")
                _purge_directory_contents(lp)

    # Merge
    if args.merge_csv:
        csvs = [Path(r["csv"]) for r in all_results if r.get("csv")]
        metas = [r for r in all_results if r.get("csv")]
        print(f"\n[merge] {len(csvs)} CSVs -> {args.merge_csv}")
        merged = merge_csvs(csvs, Path(args.merge_csv).expanduser(), metas)
        if merged["ok"]:
            print(f"[merge] OK rows={merged['rows']} -> {merged['merged_path']}")
            preview_merged_csv(Path(args.merge_csv).expanduser())

        else:
            print(
                f"[merge] FAILED: {merged.get('reason')} offender={merged.get('offender')}",
                file=sys.stderr,
            )
            return 2

    # Exit non-zero if any failed
    failed = [r for r in all_results if not r["ok"]]
    if failed:
        print(
            f"\n[summary] {len(all_results)} runs, {len(failed)} failed",
            file=sys.stderr,
        )
        for r in failed:
            print(
                f"  - {r['name']} | py={r['py']} | rc={r['rc']} | logs: {r['stderr']}",
                file=sys.stderr,
            )
        return 1

    print(f"\n[summary] {len(all_results)} runs, all succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
