import functools
import glob
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path


def smart_write(path, value, description):
    """
    Tries to write 'value' to 'path'.
    PRIORITY 1: 'sudo tee' (Assumes user has sudo access)
    PRIORITY 2: Direct write (Assumes user is root or file is user-writable)
    """
    if not os.path.exists(path):
        print(f"[SKIP] {description}: File not found")
        return False

    # Attempt 1: Sudo Write
    try:
        command = ["sudo", "-n", "tee", path]
        process = subprocess.run(
            command,
            input=str(value),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.returncode == 0:
            print(f"[OK] {description} set to {value} (Sudo)")
            return True
        # If return code is non-zero, we fall through to Attempt 2
    except (FileNotFoundError, PermissionError, OSError):
        pass  # Sudo might not be installed or accessible

    # Attempt 2: Direct Write (Fallback)
    try:
        with open(path, "w") as f:
            f.write(str(value))
        print(f"[OK] {description} set to {value} (Direct)")
        return True
    except PermissionError:
        print(f"[FAIL] {description}: Permission denied (Sudo failed, Direct failed)")
        return False
    except Exception as e:
        print(f"[FAIL] {description}: Error {e}")
        return False


def _read_value(path):
    """Reads a value from a system file."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (PermissionError, IOError, FileNotFoundError):
        return None


def _batch_write(writes, description):
    """
    Batch-write multiple (path, value) pairs using a single sudo subprocess.
    Falls back to direct writes if sudo fails.
    Returns the number of successful writes.
    """
    if not writes:
        return 0

    # Build a shell script that writes all values in one go.
    parts = []
    for path, value in writes:
        parts.append(f"printf '%s' {shlex.quote(value)} > {shlex.quote(path)}")
    script = "set -e\n" + "\n".join(parts)

    # Attempt 1: Sudo batch write
    try:
        result = subprocess.run(
            ["sudo", "-n", "sh", "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print(f"[OK] {description}: {len(writes)} entries (Sudo Batch)")
            return len(writes)
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # Attempt 2: Direct writes as fallback
    count = 0
    for path, value in writes:
        try:
            with open(path, "w") as f:
                f.write(str(value))
            count += 1
        except (PermissionError, OSError):
            pass

    if count > 0:
        print(f"[OK] {description}: {count}/{len(writes)} entries (Direct Fallback)")
    else:
        print(f"[FAIL] {description}: Permission denied")
    return count


def check_kernel_cmdline():
    """Checks /proc/cmdline for C-State limits."""
    print("--- Checking Kernel Boot Parameters ---")
    try:
        cmdline = _read_value("/proc/cmdline") or ""
        required = ["intel_idle.max_cstate=1", "processor.max_cstate=1"]
        missing = [flag for flag in required if flag not in cmdline]

        if not missing:
            print("[OK] C-States are restricted in kernel boot parameters.")
        else:
            print("[WARN] CPU C-State restrictions are MISSING in boot parameters.")
            print(f"       Seen: {cmdline}")
            print(f"       Missing: {', '.join(missing)}")
            print(
                "       Fix: Add flags to GRUB_CMDLINE_LINUX_DEFAULT and run 'sudo update-grub' + reboot."
            )
            print("       Locations to check:")
            print("         1. /etc/default/grub (Standard Linux)")
            print(
                "         2. /etc/default/grub.d/50-cloudimg-settings.cfg (AWS/Cloud specific override!)"
            )
    except Exception:
        pass
    print("---------------------------------------")


def _detect_aws() -> bool:
    """Returns True if running on AWS EC2."""
    try:
        with open("/sys/class/dmi/id/product_name") as f:
            return "amazon" in f.read().strip().lower()
    except OSError:
        return False


def _block_device_for_path(path: Path) -> str | None:
    """Returns the base block device name (e.g. 'nvme0n1') hosting the given path."""
    real = str(path.resolve())
    try:
        with open("/proc/mounts") as f:
            lines = f.readlines()
    except OSError:
        return None
    best_mount, best_dev = "", ""
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        dev, mount = parts[0], parts[1]
        # Normalise mount point to ensure prefix matching works at directory boundaries
        if (real + "/").startswith(mount.rstrip("/") + "/") and len(mount) > len(
            best_mount
        ):
            best_mount, best_dev = mount, dev
    if not best_dev:
        return None
    # Resolve symlinks (e.g. /dev/disk/by-label/... → /dev/nvme0n1p1)
    return os.path.basename(os.path.realpath(best_dev))


def check_storage_path(path: Path) -> None:
    """Warns if the input path is not on a locally mounted NVMe device (AWS only)."""
    print("--- Checking Storage Path ---")
    if not _detect_aws():
        print("[SKIP] Storage check: not running on AWS EC2")
        print("-----------------------------")
        return
    dev = _block_device_for_path(path)
    if dev is None:
        print(f"[WARN] Storage check: could not determine block device for {path}")
        print("-----------------------------")
        return
    if dev.startswith("nvme"):
        print(f"[OK]   Input path is on local NVMe ({dev}): {path}")
    else:
        print(f"[WARN] Input path '{path}' is on '{dev}' — NOT a local NVMe device.")
        print(
            "       On AWS, instance store NVMe delivers far higher IO throughput than EBS."
        )
        print(
            "       Mount your dataset on a local NVMe drive (e.g. /mnt/nvme) for accurate results."
        )
    print("-----------------------------")


class SystemStabilizer:
    def __init__(self):
        self.original_state = {}
        self.aslr_path = "/proc/sys/kernel/randomize_va_space"
        self.turbo_path_pstate = "/sys/devices/system/cpu/intel_pstate/no_turbo"
        self.turbo_path_cpufreq = "/sys/devices/system/cpu/cpufreq/boost"
        self.governor_pattern = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        self.cstate_process = None
        # Pattern for C-States: cpu0/cpuidle/state1/disable, state2/disable...
        self.cstate_disable_pattern = (
            "/sys/devices/system/cpu/cpu*/cpuidle/state*/disable"
        )

    def _enable_cstate_lock(self):
        """
        Spawns a background 'sudo sh' process to hold /dev/cpu_dma_latency open.
        This forces the CPU to stay in C0 (active) state.
        Does not require Python on the root user side.
        """
        print("--- Engaging C-State Latency Lock ---")

        # 1. Open FD 3 to the latency file
        # 2. Print 4 null bytes (int32 0) to FD 3
        # 3. Sleep for a year to keep FD 3 open
        shell_cmd = (
            "exec 3> /dev/cpu_dma_latency; "
            "printf '\\0\\0\\0\\0' >&3; "
            # Replace the privileged shell so sudo forwards termination to the holder.
            "exec sleep 31536000"
        )

        try:
            # Spawn the blocker process with sudo
            self.cstate_process = subprocess.Popen(
                ["sudo", "-n", "sh", "-c", shell_cmd],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check if it failed immediately (e.g. permission error)
            try:
                # We wait 0.1s to see if it crashes
                _, stderr = self.cstate_process.communicate(timeout=0.1)
                print(
                    f"[FAIL] C-State Lock: Process exited early. Error: {stderr.strip()}"
                )
                self.cstate_process = None
            except subprocess.TimeoutExpired:
                # Timeout means the process is still running (sleeping), which is SUCCESS
                print("[OK] C-State Latency Lock engaged (Background Shell Process)")

        except Exception as e:
            print(f"[FAIL] C-State Lock: Could not spawn process. {e}")

    def _disable_deep_cstates_manual(self):
        """
        Manually disables idle states in sysfs (state1, state2, etc).
        Skips state0 (POLL/Active) to avoid hanging the CPU.
        """
        print("--- Manually Disabling Deep C-States (Sysfs) ---")
        self.original_state["cpuidle"] = {}

        # Glob all disable files
        disable_files = glob.glob(self.cstate_disable_pattern)

        # Collect reads (fast, no subprocess) and pending writes
        pending_writes = []
        for path in disable_files:
            # SAFETY CHECK: Never disable state0.
            # Path looks like .../cpu0/cpuidle/state0/disable
            if "/state0/" in path:
                continue

            orig = _read_value(path)
            if orig is not None:
                # Store original value so we can restore it later
                self.original_state["cpuidle"][path] = orig

                # If currently enabled (0), queue a write to disable it
                if orig == "0":
                    pending_writes.append((path, "1"))

        # Batch write all C-state disables in a single subprocess
        count = _batch_write(pending_writes, "C-State Disable")

        if count > 0:
            print(f"[OK] Manually disabled {count} deep sleep states.")
        else:
            print("[INFO] No deep C-states found or they were already disabled.")

    def enter(self):
        if sys.platform != "linux":
            return
        print("\n=== Stabilizing System for Benchmarking ===")
        check_kernel_cmdline()

        self._enable_cstate_lock()
        self._disable_deep_cstates_manual()

        # 1. Disable ASLR
        curr_aslr = _read_value(self.aslr_path)
        if curr_aslr:
            self.original_state["aslr"] = curr_aslr
            smart_write(self.aslr_path, "0", "ASLR")
        else:
            print("[SKIP] ASLR: File not found")

        # 2. Disable Turbo Boost
        # Method A: Intel P-State (1=Disable)
        curr_pstate = _read_value(self.turbo_path_pstate)
        if curr_pstate is not None:
            self.original_state["pstate_turbo"] = curr_pstate
            smart_write(self.turbo_path_pstate, "1", "Intel P-State Turbo (1=Off)")
        else:
            print("[SKIP] Intel P-State Turbo: Not supported")

        # Method B: CPUFreq Boost (0=Disable)
        curr_boost = _read_value(self.turbo_path_cpufreq)
        if curr_boost is not None:
            self.original_state["cpufreq_boost"] = curr_boost
            smart_write(self.turbo_path_cpufreq, "0", "CPUFreq Boost (0=Off)")
        else:
            print("[SKIP] CPUFreq Boost: Not supported")

        # 3. Set Governors
        cpu_govs = glob.glob(self.governor_pattern)
        if not cpu_govs:
            print("[SKIP] CPU Governors: No control files found (AWS Managed?)")
        else:
            self.original_state["governors"] = {}
            pending_writes = []
            for gov_file in cpu_govs:
                orig = _read_value(gov_file)
                if orig:
                    self.original_state["governors"][gov_file] = orig
                    pending_writes.append((gov_file, "performance"))
            success = _batch_write(pending_writes, "CPU Governors")
            if success == 0:
                print("[WARN] Could not set any CPU governors.")

        print("=== Setup Complete ===\n")

    def exit(self):
        if sys.platform != "linux":
            return
        print("\n=== Restoring System State ===")

        # 1. Release C-State Lock
        if self.cstate_process:
            print("Releasing C-State Lock...", end=" ")
            self.cstate_process.terminate()
            try:
                self.cstate_process.wait(timeout=1)
                print("Done.")
            except subprocess.TimeoutExpired:
                self.cstate_process.kill()
                print("Killed.")

        if "cpuidle" in self.original_state and self.original_state["cpuidle"]:
            writes = list(self.original_state["cpuidle"].items())
            print(f"Restoring {len(writes)} C-state flags...")
            _batch_write(writes, "C-State Restore")

        if "aslr" in self.original_state:
            smart_write(self.aslr_path, self.original_state["aslr"], "Restoring ASLR")
        if "pstate_turbo" in self.original_state:
            smart_write(
                self.turbo_path_pstate,
                self.original_state["pstate_turbo"],
                "Restoring P-State",
            )
        if "cpufreq_boost" in self.original_state:
            smart_write(
                self.turbo_path_cpufreq,
                self.original_state["cpufreq_boost"],
                "Restoring Boost",
            )
        if "governors" in self.original_state:
            writes = list(self.original_state["governors"].items())
            _batch_write(writes, "Governor Restore")
        print("=== Restoration Complete ===")

    def __enter__(self):
        # SIGTERM would skip __exit__ and leave the host tuned; turn it into SystemExit.
        self._previous_sigterm = None
        if threading.current_thread() is threading.main_thread():
            self._previous_sigterm = signal.signal(
                signal.SIGTERM, functools.partial(_raise_system_exit, os.getpid())
            )
        try:
            self.enter()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.exit()
        finally:
            if self._previous_sigterm is not None:
                signal.signal(signal.SIGTERM, self._previous_sigterm)


def _raise_system_exit(owner_pid, signum, _frame):
    # Forked loader workers inherit this handler; they keep the default SIGTERM.
    if os.getpid() != owner_pid:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    raise SystemExit(128 + signum)
