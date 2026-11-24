"""Generated provisioning scripts, without cloud writes."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_bootstrap_declares_only_assigned_variables():
    # `declare -p` on an unset name fails and stops the `set -e` bootstrap.
    text = (Path(__file__).parents[1] / "scripts/bootstrap_benchmark.sh").read_text()
    for line in re.findall(r"^\s*declare -p ([A-Z_ ]+)", text, re.M):
        for name in line.split():
            assert re.search(rf"^{name}=", text, re.M), name


def test_launcher_requires_explicit_security_group(monkeypatch):
    launcher = script("launch_benchmark_instance")
    monkeypatch.setattr(
        launcher, "create_instance", lambda _: pytest.fail("unexpected cloud call")
    )
    result = CliRunner().invoke(launcher.main, ["example"])
    assert result.exit_code == 2
    assert "--security-group" in result.output


@pytest.mark.parametrize("mode", [[], ["--bare-bones"], ["--disable-grub"]])
def test_launcher_renders_portable_bootstrap(monkeypatch, mode):
    launcher = script("launch_benchmark_instance")
    captured = []

    def create(config):
        captured.append(config)
        return "i-example", ""

    monkeypatch.setattr(launcher, "create_instance", create)
    result = CliRunner().invoke(
        launcher.main,
        [
            "example",
            "--security-group",
            "example-group",
            "--iam-instance-profile",
            "example-ssm-role",
            "--region",
            "us-west-2",
            "--repo-url",
            "https://github.com/example/arena.git",
            *mode,
        ],
    )
    assert result.exit_code == 0, result.output
    config = captured[0]
    assert config.key_pair_name is None
    assert config.iam_instance_profile == "example-ssm-role"
    assert config.subnet_tag is None
    assert "export REPO_URL=https://github.com/example/arena.git" in config.user_data
    assert "prepare_data.py" in config.user_data
    subprocess.run(["bash", "-n"], input=config.user_data, text=True, check=True)


def test_launcher_requires_a_connection_method(monkeypatch):
    launcher = script("launch_benchmark_instance")
    monkeypatch.setattr(
        launcher, "create_instance", lambda _: pytest.fail("unexpected cloud call")
    )
    result = CliRunner().invoke(
        launcher.main, ["example", "--security-group", "example-group"]
    )
    assert result.exit_code == 2
    assert "Session Manager" in result.output


def test_keyless_connection_uses_native_ssm(monkeypatch):
    launcher = script("launch_benchmark_instance")
    messages = []
    monkeypatch.setattr(
        launcher.LOGGER, "info", lambda msg, *args: messages.append(msg % args)
    )
    launcher.output_launch_details("example", "i-example", "", None, "example-ssm-role")
    assert any(
        "ssm start-session --target i-example" in message for message in messages
    )
    assert not any("SSH configuration" in message for message in messages)


def test_custom_ami_pattern_skips_public_ssm_lookup(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    launcher = script("launch_benchmark_instance")
    config = SimpleNamespace(
        subnet_tag=None,
        security_group_name="example-group",
        ami_id=None,
        ami_pattern="custom-*",
        ami_owner="example-owner",
    )
    resolve = Mock(return_value="ami-example")
    monkeypatch.setattr(launcher, "resolve_security_group_id", lambda _: "sg-example")
    monkeypatch.setattr(launcher, "resolve_latest_ami", resolve)
    monkeypatch.setattr(
        launcher, "resolve_ami_via_ssm", lambda _: pytest.fail("custom pattern ignored")
    )

    class Resolved(Exception):
        pass

    def stop_before_cloud():
        raise Resolved

    monkeypatch.setattr(launcher, "_ec2_resource", stop_before_cloud)
    with pytest.raises(Resolved):
        launcher.create_instance(config)
    resolve.assert_called_once_with(["custom-*"], "example-owner")
