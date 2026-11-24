"""Tests for NVMe storage path detection."""

import io
from pathlib import Path
from unittest.mock import patch

from dataloader_arena.utils.stable_context import (
    _block_device_for_path,
    _detect_aws,
    check_storage_path,
)

# ---------------------------------------------------------------------------
# _detect_aws
# ---------------------------------------------------------------------------


def _make_open(path_contents: dict[str, str]):
    """Return a mock open() that reads from path_contents, raises OSError otherwise."""

    def _open(path, *args, **kwargs):
        if path in path_contents:
            return io.StringIO(path_contents[path])
        raise OSError(f"no mock for {path}")

    return _open


def test_detect_aws_via_product_name():
    fake = {"/sys/class/dmi/id/product_name": "Amazon EC2 c5.xlarge\n"}
    with patch("builtins.open", side_effect=_make_open(fake)):
        assert _detect_aws() is True


def test_detect_not_aws():
    fake = {"/sys/class/dmi/id/product_name": "Standard PC\n"}
    with patch("builtins.open", side_effect=_make_open(fake)):
        assert _detect_aws() is False


def test_detect_not_aws_files_missing():
    with patch("builtins.open", side_effect=OSError("not found")):
        assert _detect_aws() is False


# ---------------------------------------------------------------------------
# _block_device_for_path
# ---------------------------------------------------------------------------

PROC_MOUNTS = """\
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
/dev/nvme0n1p1 / ext4 rw,relatime 0 0
/dev/nvme1n1 /nvme_store ext4 rw,relatime 0 0
/dev/xvda1 /ebs_data ext4 rw,relatime 0 0
"""


def test_block_device_nvme_root():
    fake = {"/proc/mounts": PROC_MOUNTS}
    with patch("builtins.open", side_effect=_make_open(fake)):
        dev = _block_device_for_path(Path("/some/path/under/root"))
    assert dev == "nvme0n1p1"


def test_block_device_nvme_mount():
    fake = {"/proc/mounts": PROC_MOUNTS}
    with patch("builtins.open", side_effect=_make_open(fake)):
        dev = _block_device_for_path(Path("/nvme_store/dataset"))
    assert dev == "nvme1n1"


def test_block_device_ebs():
    fake = {"/proc/mounts": PROC_MOUNTS}
    with patch("builtins.open", side_effect=_make_open(fake)):
        dev = _block_device_for_path(Path("/ebs_data/train"))
    assert dev == "xvda1"


def test_block_device_unreadable_proc_mounts():
    with patch("builtins.open", side_effect=OSError("permission denied")):
        assert _block_device_for_path(Path("/data/train")) is None


# ---------------------------------------------------------------------------
# check_storage_path (integration-style, captures stdout)
# ---------------------------------------------------------------------------


def test_check_storage_path_not_aws(capsys):
    with patch("dataloader_arena.utils.stable_context._detect_aws", return_value=False):
        check_storage_path(Path("/data/train"))
    out = capsys.readouterr().out
    assert "[SKIP]" in out


def test_check_storage_path_nvme(capsys):
    with (
        patch("dataloader_arena.utils.stable_context._detect_aws", return_value=True),
        patch(
            "dataloader_arena.utils.stable_context._block_device_for_path",
            return_value="nvme0n1",
        ),
    ):
        check_storage_path(Path("/mnt/nvme/dataset"))
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert "nvme0n1" in out


def test_check_storage_path_ebs_warns(capsys):
    with (
        patch("dataloader_arena.utils.stable_context._detect_aws", return_value=True),
        patch(
            "dataloader_arena.utils.stable_context._block_device_for_path",
            return_value="xvda1",
        ),
    ):
        check_storage_path(Path("/data/train"))
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "xvda1" in out
    assert "NVMe" in out


def test_check_storage_path_unknown_device(capsys):
    with (
        patch("dataloader_arena.utils.stable_context._detect_aws", return_value=True),
        patch(
            "dataloader_arena.utils.stable_context._block_device_for_path",
            return_value=None,
        ),
    ):
        check_storage_path(Path("/data/train"))
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "could not determine" in out
