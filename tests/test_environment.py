"""Environment archives remain useful on CPU VMs and machines without EC2 tools."""

import io
import json
import subprocess
from unittest.mock import Mock

from dataloader_arena.utils import environment


def test_imdsv2_archives_instance_identity_without_reading_credentials(monkeypatch):
    identity = {
        "instanceType": "r8id.48xlarge",
        "instanceId": "i-example",
        "region": "us-east-1",
        "availabilityZone": "us-east-1d",
        "accountId": "not-needed-for-hardware",
        "unexpected": "omit",
    }
    opener = Mock()
    opener.open.side_effect = [
        io.BytesIO(b"test-token"),
        io.BytesIO(json.dumps(identity).encode()),
    ]
    monkeypatch.setattr(environment, "build_opener", lambda *args: opener)
    result = environment._ec2_identity("Amazon EC2")
    assert result["available"]
    assert result["identity"]["instanceType"] == "r8id.48xlarge"
    assert "accountId" not in result["identity"]
    assert "unexpected" not in result["identity"]
    token_request, document_request = [
        call.args[0] for call in opener.open.call_args_list
    ]
    assert token_request.method == "PUT"
    assert token_request.full_url.endswith("/api/token")
    assert document_request.full_url.endswith("/dynamic/instance-identity/document")
    assert (
        dict((k.lower(), v) for k, v in document_request.header_items())[
            "x-aws-ec2-metadata-token"
        ]
        == "test-token"
    )
    assert all(call.kwargs["timeout"] == 0.5 for call in opener.open.call_args_list)


def test_non_ec2_machine_never_queries_metadata_service(monkeypatch):
    monkeypatch.setattr(
        environment,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unexpected metadata query")
        ),
    )
    assert environment._ec2_identity("Apple Inc.")["available"] is False


def test_metadata_failure_is_archived_without_failing_benchmark(monkeypatch):
    opener = Mock()
    opener.open.side_effect = TimeoutError("metadata service unavailable")
    monkeypatch.setattr(environment, "build_opener", lambda *args: opener)
    assert environment._ec2_identity("Amazon EC2") == {
        "available": False,
        "reason": "TimeoutError: metadata service unavailable",
    }


def test_missing_tools_and_files_remain_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    assert environment._command(["lscpu", "--json"])["available"] is False
    assert environment._read_file(str(tmp_path / "absent"))["available"] is False


def test_hardware_command_timeout_is_nonfatal(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/lscpu")
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        Mock(side_effect=subprocess.TimeoutExpired("lscpu", 3)),
    )
    result = environment._command(["lscpu", "--json"])
    assert result["available"] is False
    assert "timed out" in result["reason"]


def test_metadata_does_not_import_torch_before_measurement(monkeypatch):
    monkeypatch.delitem(environment.sys.modules, "torch", raising=False)
    assert environment._torch_runtime() == {
        "available": False,
        "reason": "torch not imported at capture time",
    }
    assert "torch" not in environment.sys.modules


def test_metadata_does_not_initialize_cuda(monkeypatch):
    from types import SimpleNamespace

    fake_cuda = Mock()
    fake_cuda.is_initialized.return_value = False
    fake_cuda.is_available.side_effect = AssertionError("unexpected CUDA probe")
    fake_cuda.device_count.side_effect = AssertionError(
        "unexpected CUDA initialization"
    )
    torch = SimpleNamespace(
        __version__="test",
        cuda=fake_cuda,
        version=SimpleNamespace(cuda="12.8"),
        __config__=SimpleNamespace(
            show=lambda: "build", parallel_info=lambda: "threads"
        ),
        get_num_threads=lambda: 1,
        get_num_interop_threads=lambda: 1,
    )
    monkeypatch.setitem(environment.sys.modules, "torch", torch)
    result = environment._torch_runtime()
    assert result["available"] and result["cuda_initialized"] is False
    assert result["cuda_available"] is None and result["cuda_device_count"] is None
