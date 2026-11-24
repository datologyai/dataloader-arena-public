"""Storage, checkpoint-state summaries and RSS sampling for checkpoint benchmarks."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import psutil
import torch

from dataloader_arena.harness import write_json_output


class ArtifactStore:
    def __init__(self) -> None:
        self._client = None

    def _s3(self, uri: str):
        import boto3

        parsed = urlparse(uri)
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError(f"expected s3://bucket/key, got {uri!r}")
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client, parsed.netloc, parsed.path.lstrip("/")

    def read(self, uri: str) -> bytes:
        if uri.startswith("s3://"):
            client, bucket, key = self._s3(uri)
            response = client.get_object(Bucket=bucket, Key=key)
            with response["Body"] as body:
                return body.read()
        return Path(uri).read_bytes()

    def write(self, uri: str, payload: bytes) -> None:
        if uri.startswith("s3://"):
            client, bucket, key = self._s3(uri)
            client.put_object(Bucket=bucket, Key=key, Body=payload)
        else:
            path = Path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    def write_json(self, uri: str, value: object) -> None:
        self.write(uri, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


# One artifact client per process.
@lru_cache(maxsize=1)
def _artifact_store() -> ArtifactStore:
    return ArtifactStore()


def read_bytes(uri: str) -> bytes:
    return _artifact_store().read(uri)


def write_bytes(uri: str, payload: bytes) -> None:
    _artifact_store().write(uri, payload)


def write_json(uri: str, value: dict) -> None:
    if "://" not in uri:
        write_json_output(Path(uri), value)
    else:
        _artifact_store().write_json(uri, value)


def prepare_cache(base: Path, namespace: str, rank: int | None, *, cold: bool) -> Path:
    """Never clear the user's dataset or a caller-supplied arbitrary directory.

    Cold runs require a new empty namespace, instead of deleting existing files.
    A None rank selects the host-shared cache; its caller coordinates initialization.
    """
    if not namespace or Path(namespace).name != namespace or namespace in {".", ".."}:
        raise ValueError("cache namespace must be one non-special path component")
    leaf = "host" if rank is None else f"rank-{rank}"
    root = base.resolve() / ".arena-checkpoint-cache" / namespace / leaf
    if root.is_symlink():
        raise ValueError(f"symlinked cache directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if cold and any(root.iterdir()):
        raise ValueError(f"cold cache is not empty: {root}; choose a new namespace")
    return root


def cache_summary(root: Path) -> dict[str, int]:
    """Walk the cache outside measurement intervals."""
    files = size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                size += path.stat().st_size
                files += 1
        except FileNotFoundError:
            pass  # An async download may have renamed a temporary file.
    return {"resident_files": files, "resident_bytes": size}


def fingerprint(batch: Any) -> str:
    """Hash payload contents, shapes and dtypes; exclude checkpoint cursor metadata."""
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays cannot be fingerprinted reproducibly")
            digest.update(f"array:{value.dtype}:{value.shape}:".encode())
            digest.update(value.tobytes(order="C"))
        elif isinstance(value, np.generic):
            visit(value.item())
        elif isinstance(value, Mapping):
            digest.update(b"mapping[")
            for key in sorted(value):
                visit(key)
                visit(value[key])
            digest.update(b"]")
        elif isinstance(value, (list, tuple)):
            digest.update(f"{type(value).__name__}[".encode())
            for item in value:
                visit(item)
            digest.update(b"]")
        elif hasattr(value, "records"):
            visit([record.payload for record in value.records])
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(json.dumps(value, allow_nan=False).encode() + b";")
        else:
            raise TypeError(f"unsupported continuation payload: {type(value).__name__}")

    visit(batch)
    return digest.hexdigest()


def summarize_checkpoint(
    state: Mapping[str, Any], dp_degree: int, mapping_strategy: str = "contiguous"
) -> dict[str, Any]:
    """Count serialized source records, not packed outputs or replay-filter work."""
    world = state["world"]
    lanes = int(world["canonical_replicas"])
    if int(world.get("dp_degree", dp_degree)) != dp_degree:
        raise ValueError("checkpoint DP degree differs from benchmark")
    per_lane = {lane: {"chunks": 0, "source_records": 0} for lane in range(lanes)}
    for raw_lane, chunks in state["inflight"].items():
        counts = per_lane[int(raw_lane)]
        counts["chunks"] = len(chunks)
        for chunk in chunks.values():
            count = chunk.get("total_samples")
            if count is None:
                count = sum(len(samples) for _, samples in chunk["components"])
            counts["source_records"] += int(count)
    per_dp = {rank: Counter(chunks=0, source_records=0) for rank in range(dp_degree)}
    if "mapping" in world:
        mapping = {int(rank): lane_ids for rank, lane_ids in world["mapping"].items()}
        mapping_source = "checkpoint"
    else:
        # Zephon's aggregated checkpoint omits the rank mapping; rebuild it from config.
        if mapping_strategy not in {"contiguous", "interleaved"} or lanes < dp_degree:
            raise ValueError("unsupported lane mapping")
        base, extra = divmod(lanes, dp_degree)
        mapping = {}
        start = 0
        for rank in range(dp_degree):
            count = base + (rank < extra)
            mapping[rank] = (
                list(range(rank, lanes, dp_degree))
                if mapping_strategy == "interleaved"
                else list(range(start, start + count))
            )
            start += count
        mapping_source = "configuration (aggregated checkpoint omits mapping)"
    owners = {
        int(lane): rank for rank, lane_ids in mapping.items() for lane in lane_ids
    }
    if set(owners) != set(per_lane):
        raise ValueError("checkpoint mapping does not cover all canonical lanes")
    for lane, counts in per_lane.items():
        per_dp[owners[lane]].update(counts)
    return {
        "canonical_lanes": lanes,
        "mapping": mapping,
        "mapping_source": mapping_source,
        "per_lane": per_lane,
        "per_dp_rank": {rank: dict(counts) for rank, counts in per_dp.items()},
        "total": {
            key: sum(c[key] for c in per_lane.values())
            for key in ("chunks", "source_records")
        },
    }


def _rss_bytes() -> int:
    parent = psutil.Process()
    total = parent.memory_info().rss
    for child in parent.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


@dataclass
class RssSample:
    baseline_bytes: int
    peak_bytes: int

    @property
    def incremental_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


class RssSampler:
    """Poll peak process-tree RSS over a checkpoint save or restore interval."""

    def __init__(self, interval_s: float = 0.01) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.sample: RssSample | None = None

    def __enter__(self) -> "RssSampler":
        baseline = _rss_bytes()
        self.sample = RssSample(baseline_bytes=baseline, peak_bytes=baseline)

        def poll() -> None:
            assert self.sample is not None
            while not self._stop.wait(self._interval_s):
                current = _rss_bytes()
                with self._lock:
                    self.sample.peak_bytes = max(self.sample.peak_bytes, current)

        self._thread = threading.Thread(
            target=poll,
            name="checkpoint-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def snapshot(self) -> RssSample:
        """Return a current copy, including a synchronous sample."""
        current = _rss_bytes()
        with self._lock:
            assert self.sample is not None
            self.sample.peak_bytes = max(self.sample.peak_bytes, current)
            return RssSample(
                baseline_bytes=self.sample.baseline_bytes,
                peak_bytes=self.sample.peak_bytes,
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self.snapshot()

    def result(self) -> dict[str, int]:
        assert self.sample is not None
        return {
            "rss_baseline_bytes": self.sample.baseline_bytes,
            "rss_peak_bytes": self.sample.peak_bytes,
            "rss_incremental_bytes": self.sample.incremental_bytes,
        }
