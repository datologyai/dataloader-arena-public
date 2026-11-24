"""Best-effort host metadata collected before benchmark measurement windows."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

import psutil

# Configuration that can affect CPU, IO, NUMA or process-runner performance.
_SYSTEM_FILES = (
    "/etc/os-release",
    "/proc/cmdline",
    "/proc/meminfo",
    "/proc/mdstat",
    "/proc/self/cgroup",
    "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/cpu.weight",
    "/sys/fs/cgroup/cpuset.cpus.effective",
    "/sys/fs/cgroup/cpuset.mems.effective",
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory.high",
    "/sys/devices/system/cpu/smt/active",
    "/sys/devices/system/cpu/smt/control",
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
    "/sys/devices/system/cpu/intel_pstate/no_turbo",
    "/sys/module/intel_idle/parameters/max_cstate",
    "/sys/module/processor/parameters/max_cstate",
    "/sys/kernel/mm/transparent_hugepage/enabled",
    "/proc/sys/kernel/randomize_va_space",
    "/proc/sys/kernel/nmi_watchdog",
    "/proc/sys/kernel/numa_balancing",
    "/proc/sys/kernel/perf_event_paranoid",
    "/proc/sys/vm/swappiness",
    "/proc/sys/net/core/rmem_max",
    "/proc/sys/net/core/wmem_max",
    "/proc/sys/fs/pipe-max-size",
    "/proc/sys/fs/pipe-user-pages-soft",
)
_DMI_FILES = (
    "sys_vendor",
    "product_name",
    "product_version",
    "bios_vendor",
    "bios_version",
    "bios_date",
)


def _read_file(path: str) -> dict[str, Any]:
    try:
        return {"available": True, "text": Path(path).read_text().strip()}
    except OSError as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _command(arguments: list[str]) -> dict[str, Any]:
    if shutil.which(arguments[0]) is None:
        return {"available": False, "reason": "command not installed"}
    try:
        completed = subprocess.run(
            arguments, capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": str(error)}
    result = {
        "available": completed.returncode == 0,
        "command": arguments,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0:
        try:
            result["json"] = json.loads(completed.stdout)
            result.pop("stdout")
        except ValueError:
            pass
    return result


def _ec2_identity(vendor: str) -> dict[str, Any]:
    """Read only the instance identity document through IMDSv2; never IAM credentials."""
    if vendor.lower() != "amazon ec2":
        return {"available": False, "reason": "host is not identified as Amazon EC2"}
    base = "http://169.254.169.254/latest/"
    # Instance metadata must not be sent through an HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    try:
        request = Request(
            base + "api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with opener.open(request, timeout=0.5) as response:
            token = response.read().decode()
        request = Request(
            base + "dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with opener.open(request, timeout=0.5) as response:
            document = json.load(response)
        fields = (
            "instanceId",
            "instanceType",
            "architecture",
            "imageId",
            "region",
            "availabilityZone",
            "pendingTime",
            "privateIp",
            "version",
        )
        return {
            "available": True,
            "source": "IMDSv2 instance identity document",
            "identity": {key: document.get(key) for key in fields},
        }
    except (OSError, ValueError) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _probe(function):
    try:
        return function()
    except (OSError, RuntimeError, NotImplementedError, psutil.Error) as error:
        return {"available": False, "reason": str(error)}


def _torch_runtime() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {"available": False, "reason": "torch not imported at capture time"}
    try:
        cuda_initialized = torch.cuda.is_initialized()
        cpu_build = torch.version.cuda is None
        return {
            "available": True,
            "version": torch.__version__,
            "build_config": torch.__config__.show(),
            "parallel_info": torch.__config__.parallel_info(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_build_version": torch.version.cuda,
            "cuda_initialized": cuda_initialized,
            "cuda_available": False
            if cpu_build
            else (True if cuda_initialized else None),
            "cuda_device_count": 0
            if cpu_build
            else (torch.cuda.device_count() if cuda_initialized else None),
        }
    except (ImportError, RuntimeError) as error:
        return {"available": False, "reason": str(error)}


def host_snapshot() -> dict[str, Any]:
    """Archive the host shape and tuning; missing probes stay visible in the report."""
    dmi = {name: _read_file(f"/sys/class/dmi/id/{name}") for name in _DMI_FILES}
    vendor = dmi["sys_vendor"].get("text", "")
    commands = {
        "cpu": ["lscpu", "--json"],
        "cpu_topology": ["lscpu", "--extended=CPU,CORE,SOCKET,NODE,ONLINE"],
        "numa": ["numactl", "--hardware"],
        "block_devices": [
            "lsblk",
            "--json",
            "--bytes",
            "--output",
            "NAME,TYPE,SIZE,MODEL,ROTA,FSTYPE,MOUNTPOINTS,RA,SCHED,TRAN",
        ],
        "mounts": ["findmnt", "--json", "--output", "SOURCE,TARGET,FSTYPE,OPTIONS"],
        "nvme": ["nvme", "list", "--output-format=json"],
        "gpus": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
    }
    if sys.platform == "darwin":
        commands["mac_hardware"] = [
            "sysctl",
            "hw.model",
            "hw.memsize",
            "hw.physicalcpu",
            "hw.logicalcpu",
            "machdep.cpu.brand_string",
        ]
    storage = []
    try:
        for partition in psutil.disk_partitions(all=False):
            entry = partition._asdict()
            try:
                entry["usage"] = psutil.disk_usage(partition.mountpoint)._asdict()
            except OSError as error:
                entry["usage_error"] = str(error)
            storage.append(entry)
    except (OSError, psutil.Error) as error:
        storage.append({"unavailable": str(error)})
    cpu_frequency = None
    try:
        frequency = psutil.cpu_freq()
        cpu_frequency = frequency._asdict() if frequency else None
    except (OSError, NotImplementedError):
        pass
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "uname": platform.uname()._asdict(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "physical_cpu_count": _probe(lambda: psutil.cpu_count(logical=False)),
        "cpu_frequency_mhz": cpu_frequency,
        "memory": _probe(lambda: psutil.virtual_memory()._asdict()),
        "swap": _probe(lambda: psutil.swap_memory()._asdict()),
        "filesystems": storage,
        "dmi": dmi,
        "system_files": {name: _read_file(name) for name in _SYSTEM_FILES},
        "commands": {name: _command(command) for name, command in commands.items()},
        "torch": _torch_runtime(),
        "ec2": _ec2_identity(vendor),
        "process": {"pid": os.getpid(), "cwd": str(Path.cwd())},
    }
