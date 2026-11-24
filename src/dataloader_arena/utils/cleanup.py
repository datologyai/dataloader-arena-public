import ctypes
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path


def purge_directory_contents(dir_path: Path) -> None:
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


def trim_malloc():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        pass  # non-glibc, ignore


def cleanup_between_runs(*, tune_system: bool = True) -> bool | None:
    """Free memory and drop the page cache between runs.

    Returns whether the page cache was dropped, or None if it wasn't attempted.
    """
    # 1. GC
    gc.collect()
    trim_malloc()
    gc.collect()
    trim_malloc()

    if not tune_system or sys.platform != "linux":
        return None

    # 2. Sync
    subprocess.run(["sync"])

    # 3. Drop Caches
    path = "/proc/sys/vm/drop_caches"
    cache_success = False

    # Attempt 1: Sudo Shell
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", f"echo 3 > {path}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cache_success = True
    except (subprocess.CalledProcessError, OSError):
        pass

    # Attempt 2: Direct Write
    if not cache_success:
        try:
            with open(path, "w") as f:
                f.write("3")
            cache_success = True
        except (PermissionError, FileNotFoundError):
            pass  # non-Linux or unprivileged — not critical

    # 4. Allow compaction to move unevictable (mlocked) pages
    path_unevictable = "/proc/sys/vm/compact_unevictable_allowed"
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", f"echo 1 > {path_unevictable}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        try:
            with open(path_unevictable, "w") as f:
                f.write("1")
        except (PermissionError, FileNotFoundError):
            pass

    # 5. Compact Memory
    path_compact = "/proc/sys/vm/compact_memory"
    compact_success = False

    # Attempt 1: Sudo Shell
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", f"echo 1 > {path_compact}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        compact_success = True
    except (subprocess.CalledProcessError, OSError):
        pass

    # Attempt 2: Direct Write
    if not compact_success:
        try:
            with open(path_compact, "w") as f:
                f.write("1")
        except (PermissionError, FileNotFoundError):
            pass  # Not critical

    time.sleep(3)
    return cache_success
