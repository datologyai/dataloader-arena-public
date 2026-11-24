import subprocess
from unittest.mock import Mock, patch

from dataloader_arena.utils.stable_context import SystemStabilizer


def test_latency_holder_replaces_shell_and_is_reaped():
    process = Mock()
    process.communicate.side_effect = subprocess.TimeoutExpired("sudo", 0.1)
    stabilizer = SystemStabilizer()
    with patch(
        "dataloader_arena.utils.stable_context.subprocess.Popen", return_value=process
    ) as popen:
        stabilizer._enable_cstate_lock()
    command = popen.call_args.args[0]
    assert command[:4] == ["sudo", "-n", "sh", "-c"]
    assert command[4].endswith("exec sleep 31536000")
    assert "exec 3> /dev/cpu_dma_latency" in command[4]
    with patch("sys.platform", "linux"):
        stabilizer.exit()
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=1)
