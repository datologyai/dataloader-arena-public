"""Prepare public text and VLM datasets with local or S3 storage.

Run with ``uv run --locked --extra data-preparation --inexact python scripts/prepare_data.py --help``.
Preparation uses arena's root uv.lock. No model weights or GPU are needed.
Local output needs no AWS credentials.

Defaults support throughput runs through 64 workers, including read-ahead
and a 20% margin: 524,288 text records of 2,048 uint32 tokens (about 4 GiB),
or 262,144 original VLM conversations (about 120 GiB), in 64 MiB chunks.
Use --samples for smaller setup checks or larger measurement windows.
For startup experiments, --shard-samples accepts a JSON list of record counts
to preserve a reference export's shard metadata shape. It determines the total
record count and replaces byte-target chunking; records are never repeated or
padded to fill a shard. Paper-specific presets belong to the experiment repo.
Mammoth first caches 21.8 GB of annotations and builds a reusable disk index;
allow about 30 GiB of cache space in addition to the VLM output.
A small valid-record quota is streamed from each image archive, within the
download cap. This gives broad coverage without a full image-corpus download;
it is not a uniform sample of every image record.
VLM records are then mixed locally before chunking, using the same seed. This
downloads no extra images and temporarily needs space for a second VLM export.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import random
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from typing import Any

HUB = "https://huggingface.co"
TEXT_REPO = "HuggingFaceFW/fineweb"
TEXT_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
VLM_REPO = "MAmmoTH-VL/MAmmoTH-VL-Instruct-12M"
VLM_REVISION = "bac8f77cb8a8f9c4d0de407c6e3a589bd722562a"
VLM_ANNOTATIONS = "mammoth_si_10M.json"
TOKENIZER_REPO = "unsloth/Llama-3.2-1B"
TOKENIZER_REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
MIB = 1024**2
UINT32_MAX = 2**32 - 1
PACKAGES = (
    "numpy",
    "pillow",
    "pyarrow",
    "httpx2",
    "tokenizers",
    "huggingface-hub",
    "ijson",
)
_TOKENIZER: Any = None


class PreparationError(RuntimeError):
    """A source, resource limit, or output prevents completing the requested data."""


class ArchivePrefixLimit(PreparationError):
    """Stop an unproductive archive prefix without consuming the whole budget."""


class PublicHub:
    """Use Hub metadata, its download cache, and hf:// files anonymously."""

    def __init__(self, limit: int, cache_dir: Path) -> None:
        from huggingface_hub import HfApi, HfFileSystem, set_client_factory
        from huggingface_hub.utils import httpx

        # Preserve anonymous requests without environment proxies or netrc auth.
        set_client_factory(lambda: httpx.Client(trust_env=False, follow_redirects=True))
        self.api = HfApi(endpoint=HUB, token=False)
        self.fs = HfFileSystem(endpoint=HUB, token=False, skip_instance_cache=True)
        self.cache_dir = cache_dir.expanduser()
        self.limit = limit
        self.received = 0

    def check_budget(self, size: int) -> None:
        if self.received + size > self.limit:
            raise PreparationError(
                "Download limit reached; increase --max-download-mib or request fewer samples."
            )

    def repo(self, repo: str, revision: str, kind: str = "dataset") -> Any:
        info = self.api.repo_info(
            repo, revision=revision, repo_type=kind, expand=["sha", "private", "gated"]
        )
        if info.private or info.gated not in (False, None):
            raise PreparationError(
                "The source repository must allow anonymous access without gating."
            )
        return info

    def parquet_files(self, repo: str, revision: str, prefix: str) -> list[dict]:
        from huggingface_hub.hf_api import RepoFile

        entries = self.api.list_repo_tree(
            repo, prefix, repo_type="dataset", revision=revision
        )
        return sorted(
            [
                {"path": item.path, "size": item.size}
                for item in entries
                if isinstance(item, RepoFile) and item.path.endswith(".parquet")
            ],
            key=lambda item: item["path"],
        )

    def download(self, repo: str, revision: str, filename: str, kind: str) -> str:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs = {
            "repo_id": repo,
            "revision": revision,
            "filename": filename,
            "repo_type": kind,
            "cache_dir": self.cache_dir,
            "endpoint": HUB,
            "token": False,
        }
        try:
            return hf_hub_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            metadata = get_hf_file_metadata(
                hf_hub_url(
                    repo, filename, revision=revision, repo_type=kind, endpoint=HUB
                ),
                token=False,
            )
            if metadata.size is None:
                raise PreparationError(
                    "Hub did not report the download size."
                ) from None
            self.check_budget(metadata.size)
            print(
                f"Caching {filename}: {metadata.size / MIB:,.1f} MiB (reused on later runs)",
                file=sys.stderr,
                flush=True,
            )
            result = hf_hub_download(**kwargs)
            self.received += metadata.size
            return result

    def close(self) -> None:
        from huggingface_hub import close_session

        close_session()


class BudgetedReader(io.RawIOBase):
    """Bound bytes requested through the Hub's seekable file reader."""

    def __init__(
        self, hub: PublicHub, remote: Any, max_bytes: int | None = None
    ) -> None:
        super().__init__()
        self.hub = hub
        self.remote = remote
        self.max_bytes = max_bytes
        self.received = 0
        self.started = self.last_progress = time.monotonic()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.remote.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.remote.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.remote.size - self.tell())
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        if self.max_bytes is not None:
            available = self.max_bytes - self.received
            if not available:
                raise ArchivePrefixLimit("Archive prefix read limit reached.")
            size = min(size, available)
        self.hub.check_budget(size)
        data = self.remote.read(size)
        self.hub.received += len(data)
        self.received += len(data)
        if len(data) != size:
            raise PreparationError("Hub returned an incorrect byte range.")
        now = time.monotonic()
        if now - self.last_progress >= 5:
            print(
                f"Reading source data: {self.received / MIB:,.1f} MiB read in {now - self.started:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            self.last_progress = now
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def tree_spec(keys: list[str], children: list[dict]) -> dict:
    return {
        "type": "builtins.dict",
        "context": json.dumps(keys),
        "children_spec": children,
    }


def leaf_spec() -> dict:
    return {"type": None, "context": None, "children_spec": []}


def encode_leaves(leaves: list[bytes]) -> bytes:
    return struct.pack(f"<{len(leaves)}I", *(len(leaf) for leaf in leaves)) + b"".join(
        leaves
    )


class LitWriter:
    """Write the text and VLM schemas using the LitData PyTree wire format.

    Per item: uint32 leaf lengths, followed by leaf bytes. Per chunk: uint32
    item count, uint32 absolute offsets (N+1), followed by items.
    See Lightning-AI/litdata streaming/{writer,item_loader,serializers}.py.
    """

    def __init__(
        self,
        output: Path,
        mode: str,
        chunk_bytes: int,
        shard_samples: list[int] | None = None,
    ) -> None:
        self.output = output
        self.chunk_bytes = chunk_bytes
        self.items: list[bytes] = []
        self.payload_bytes = 0
        self.chunks: list[dict] = []
        self.count = 0
        self.shard_samples = shard_samples
        self.expected_samples = (
            sum(shard_samples) if shard_samples is not None else None
        )
        spec = leaf_spec()
        formats = ["no_header_numpy:6"]
        if mode == "vlm":
            metadata = tree_spec(
                ["document_id", "url", "num_images"], [leaf_spec() for _ in range(3)]
            )
            spec = tree_spec(
                ["images", "text", "metadata"], [leaf_spec(), leaf_spec(), metadata]
            )
            formats = ["bytes", "str", "str", "str", "int"]
        self.config = {
            "chunk_bytes": chunk_bytes,
            "chunk_size": None,
            "compression": None,
            "data_format": formats,
            "data_spec": json.dumps([1, spec]),
            "encryption": None,
            "item_loader": "PyTreeLoader",
        }

    def add(self, item: bytes) -> None:
        if self.expected_samples is not None and self.count >= self.expected_samples:
            raise PreparationError("More records than requested by --shard-samples.")
        next_size = 4 + 4 * (len(self.items) + 2) + self.payload_bytes + len(item)
        if self.shard_samples is None and self.items and next_size > self.chunk_bytes:
            self.flush()
            next_size = len(item) + 12
        if next_size > UINT32_MAX:
            raise PreparationError("Chunk exceeds the LitData uint32 offset limit.")
        self.items.append(item)
        self.payload_bytes += len(item)
        self.count += 1
        if (
            self.shard_samples is not None
            and len(self.items) == self.shard_samples[len(self.chunks)]
        ):
            self.flush()

    def flush(self) -> None:
        if not self.items:
            return
        offsets = [4 + 4 * (len(self.items) + 1)]
        for item in self.items:
            offsets.append(offsets[-1] + len(item))
        filename = f"chunk-0-{len(self.chunks)}.bin"
        with (self.output / filename).open("wb") as handle:
            handle.write(
                struct.pack(f"<{len(offsets) + 1}I", len(self.items), *offsets)
            )
            for item in self.items:
                handle.write(item)
        self.chunks.append(
            {
                "filename": filename,
                "chunk_bytes": offsets[-1],
                "chunk_size": len(self.items),
                "dim": None,
            }
        )
        self.items.clear()
        self.payload_bytes = 0

    def finish(self) -> None:
        if self.expected_samples is not None and self.count != self.expected_samples:
            raise PreparationError("Too few records to fill --shard-samples.")
        self.flush()
        (self.output / "index.json").write_text(
            json.dumps({"config": self.config, "chunks": self.chunks}) + "\n"
        )


def shuffle_vlm_export(
    source: Path,
    output: Path,
    seed: int,
    chunk_bytes: int,
    shard_samples: list[int] | None = None,
) -> LitWriter:
    """Mix already-downloaded records before chunking, without decoding images.

    Only record offsets and provenance are held in memory. Image payloads are
    copied unchanged, using at most 32 open source chunks. Source and output
    directories must differ; the source remains available until this completes.
    """
    if source.resolve() == output.resolve():
        raise PreparationError("Record shuffling needs a separate output directory.")
    index = json.loads((source / "index.json").read_text())
    records = []
    for chunk in index["chunks"]:
        path = source / chunk["filename"]
        count = chunk["chunk_size"]
        with path.open("rb") as handle:
            header = handle.read(4 * (count + 2))
        if len(header) != 4 * (count + 2):
            raise PreparationError(f"Truncated chunk header: {path}")
        values = struct.unpack(f"<{count + 2}I", header)
        offsets = values[1:]
        if (
            values[0] != count
            or offsets[0] != len(header)
            or offsets[-1] != path.stat().st_size
            or any(a >= b for a, b in zip(offsets, offsets[1:]))
        ):
            raise PreparationError(f"Invalid chunk offsets: {path}")
        records.extend((path, a, b - a) for a, b in zip(offsets, offsets[1:]))
    provenance = [
        json.loads(line) for line in (source / "sources.jsonl").read_text().splitlines()
    ]
    if len(provenance) != len(records) or any(
        row["index"] != i for i, row in enumerate(provenance)
    ):
        raise PreparationError("Source provenance does not match the exported records.")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    writer = LitWriter(output, "vlm", chunk_bytes, shard_samples)
    readers: OrderedDict[Path, Any] = OrderedDict()
    last_progress = time.monotonic()
    try:
        with (output / "sources.jsonl").open("w") as sources:
            for original in order:
                path, start, size = records[original]
                handle = readers.pop(path, None)
                if handle is None:
                    handle = path.open("rb")
                readers[path] = handle
                if len(readers) > 32:
                    readers.popitem(last=False)[1].close()
                handle.seek(start)
                item = handle.read(size)
                if len(item) != size:
                    raise PreparationError(f"Truncated record in {path}")
                writer.add(item)
                sources.write(
                    json.dumps({**provenance[original], "index": writer.count - 1})
                    + "\n"
                )
                if time.monotonic() - last_progress >= 5:
                    print(
                        f"Mixing VLM records: {writer.count:,}/{len(records):,}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = time.monotonic()
        writer.finish()
    finally:
        for handle in readers.values():
            handle.close()
    return writer


def source_batches(
    hub: PublicHub,
    repo: str,
    revision: str,
    files: list[dict],
    columns: list[str],
    batch_size: int,
    max_rows: int,
    visited: list[dict],
) -> Iterator[list[dict]]:
    seen = 0
    for file in files:
        filename = file["path"]
        path = f"datasets/{repo}@{revision}/{filename}"
        print(f"Reading hf://{path}", file=sys.stderr, flush=True)
        before = hub.received
        entry = {"file": filename, "rows_read": 0, "file_bytes_read": 0}
        visited.append(entry)
        with (
            hub.fs.open(
                path, "rb", size=file["size"], block_size=64 * 1024, cache_type="none"
            ) as remote,
            BudgetedReader(hub, remote) as source,
        ):
            yield from parquet_batches(
                source,
                repo,
                revision,
                filename,
                columns,
                batch_size,
                max_rows - seen,
                entry,
                hub,
                before,
            )
        seen += entry["rows_read"]
        if seen >= max_rows:
            return


def parquet_batches(
    source: BudgetedReader,
    repo: str,
    revision: str,
    filename: str,
    columns: list[str],
    batch_size: int,
    max_rows: int,
    entry: dict,
    hub: PublicHub,
    before: int,
) -> Iterator[list[dict]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source, pre_buffer=False)
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise PreparationError(f"Source {filename} does not have columns {columns}.")
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=columns, use_threads=False
    ):
        rows = batch.to_pylist()[: max_rows - entry["rows_read"]]
        for i, row in enumerate(rows):
            row["_source_id"] = (
                f"{repo}@{revision}/{filename}#row={entry['rows_read'] + i}"
            )
        entry["rows_read"] += len(rows)
        entry["file_bytes_read"] = hub.received - before
        if rows:
            yield rows
        if entry["rows_read"] >= max_rows:
            return


def init_worker(tokenizer_file: str | None) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if tokenizer_file:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(tokenizer_file)
        _TOKENIZER.no_truncation()
        _TOKENIZER.no_padding()


def tokenize_batch(rows: list[dict]) -> list[tuple[str, list[int]]]:
    texts = [row["text"] for row in rows]
    if not all(isinstance(text, str) for text in texts):
        raise PreparationError("Text source contains a non-string text field.")
    encodings = _TOKENIZER.encode_batch(texts, add_special_tokens=False)
    return [(row["_source_id"], item.ids) for row, item in zip(rows, encodings)]


def encode_vlm_row(row: dict) -> bytes | None:
    from PIL import Image, UnidentifiedImageError

    images = row["_images"]
    turns = row.get("conversations")
    if not isinstance(turns, list) or not turns:
        return None
    try:
        for content in images:
            with Image.open(io.BytesIO(content)) as opened:
                opened.verify()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return None
    messages = []
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    for turn in turns:
        role, content = roles.get(turn.get("from")), turn.get("value")
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
    # Preserve original turn boundaries and image markers; never invent a Q/A pair.
    if sum(message["content"].count("<image>") for message in messages) != len(images):
        return None
    blob = struct.pack(">I", len(images)) + b"".join(
        struct.pack(">I", len(content)) + content for content in images
    )
    source_id = row["_source_id"]
    return encode_leaves(
        [
            blob,
            json.dumps(messages, ensure_ascii=False).encode("utf-8"),
            hashlib.sha256(source_id.encode()).hexdigest().encode(),
            source_id.encode(),
            struct.pack("<q", len(images)),
        ]
    )


def mammoth_index(hub: PublicHub, revision: str) -> Path:
    """Build a reusable disk index; upstream annotations are one large JSON array."""
    import ijson

    hub.cache_dir.mkdir(parents=True, exist_ok=True)
    index = hub.cache_dir / f"mammoth-si-index-v1-{revision}.sqlite"
    if index.exists():
        print("Reusing cached Mammoth annotation index.", file=sys.stderr, flush=True)
        return index
    print(
        "Warning: an uncached Mammoth setup downloads about 22 GB of annotations and needs about "
        "30 GiB of disk for the JSON plus lookup index. Allow several minutes; "
        "cached files and the completed index are reused on later runs.",
        file=sys.stderr,
        flush=True,
    )
    annotations = hub.download(VLM_REPO, revision, VLM_ANNOTATIONS, "dataset")
    descriptor, filename = tempfile.mkstemp(
        prefix="mammoth-index-", suffix=".sqlite", dir=hub.cache_dir
    )
    os.close(descriptor)
    temporary = Path(filename)
    try:
        with (
            closing(sqlite3.connect(temporary)) as db,
            open(annotations, "rb") as source,
        ):
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute(
                "CREATE TABLE records (image TEXT NOT NULL, row_number INTEGER NOT NULL, payload BLOB NOT NULL)"
            )
            pending = []
            count = text_only = 0
            started = last_progress = time.monotonic()
            size = os.fstat(source.fileno()).st_size
            print(
                "Indexing original Mammoth annotations...",
                file=sys.stderr,
                flush=True,
            )
            for number, row in enumerate(ijson.items(source, "item", use_float=True)):
                image = row.get("image") or ""
                if isinstance(image, list) and len(image) == 1:
                    image = image[0]
                if not isinstance(image, str):
                    raise PreparationError("Expected single-image Mammoth annotations.")
                payload = zlib.compress(
                    json.dumps(row, ensure_ascii=False).encode(), level=1
                )
                pending.append((image, number, payload))
                count += 1
                text_only += not image
                if len(pending) == 5000:
                    db.executemany("INSERT INTO records VALUES (?, ?, ?)", pending)
                    db.commit()
                    pending.clear()
                now = time.monotonic()
                if now - last_progress >= 5:
                    read = source.tell()
                    print(
                        f"Indexing annotations: {100 * read / size:.1f}% "
                        f"({read / MIB:,.1f}/{size / MIB:,.1f} MiB read, "
                        f"{count:,} conversations, {now - started:.0f}s elapsed)",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = now
            db.executemany("INSERT INTO records VALUES (?, ?, ?)", pending)
            print(
                f"Indexed {count:,} conversations; finalizing image lookup...",
                file=sys.stderr,
                flush=True,
            )
            db.execute("CREATE INDEX by_image ON records (image)")
            db.execute("CREATE TABLE stats (total INTEGER, text_only INTEGER)")
            db.execute("INSERT INTO stats VALUES (?, ?)", (count, text_only))
            db.commit()
        temporary.replace(index)
        print(
            f"Mammoth annotation index ready ({time.monotonic() - started:.0f}s); reused on later runs.",
            file=sys.stderr,
            flush=True,
        )
    finally:
        temporary.unlink(missing_ok=True)
    return index


def mammoth_batches(
    hub: PublicHub,
    revision: str,
    args: argparse.Namespace,
    max_rows: int,
    visited: list[dict],
) -> Iterator[list[tuple[str, bytes | None]]]:
    """Use the shared quota sampler; local and S3 preparation select identically."""
    from data_preparation.mammoth import batches

    yield from batches(hub, revision, args, max_rows, visited, helpers=globals())


def parallel_batches(
    fn: Callable,
    batches: Iterable[list[dict]],
    workers: int,
    tokenizer_file: str | None,
) -> Iterator[list]:
    """Bound both work submitted to processes and source data read ahead."""
    if workers == 1:
        init_worker(tokenizer_file)
        for batch in batches:
            yield fn(batch)
        return
    iterator = iter(batches)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(tokenizer_file,),
    ) as pool:
        pending: deque = deque()
        try:
            for _ in range(2 * workers):
                batch = next(iterator, None)
                if batch is None:
                    break
                pending.append(pool.submit(fn, batch))
            while pending:
                yield pending.popleft().result()
                batch = next(iterator, None)
                if batch is not None:
                    pending.append(pool.submit(fn, batch))
        finally:
            for future in pending:
                future.cancel()


def prepare(args: argparse.Namespace, output: Path, hub: PublicHub) -> dict:
    import numpy as np

    repo = TEXT_REPO if args.mode == "text" else VLM_REPO
    revision = args.revision or (TEXT_REVISION if args.mode == "text" else VLM_REVISION)
    info = hub.repo(repo, revision)
    revision = info.sha
    tokenizer_file = None
    tokenizer_info = None
    if args.mode == "text":
        requested = args.tokenizer_revision or (
            TOKENIZER_REVISION if args.tokenizer == TOKENIZER_REPO else "main"
        )
        model = hub.repo(args.tokenizer, requested, "model")
        tokenizer_info = {
            "repo": args.tokenizer,
            "revision": model.sha,
            "add_special_tokens": False,
        }
        tokenizer_file = hub.download(
            args.tokenizer, model.sha, "tokenizer.json", "model"
        )
        tokenizer_info["sha256"] = hashlib.sha256(
            Path(tokenizer_file).read_bytes()
        ).hexdigest()
    visited: list[dict] = []
    max_rows = args.max_source_rows or max(
        1024, args.samples * (32 if args.mode == "text" else 4)
    )
    if args.mode == "text":
        files = hub.parquet_files(repo, revision, "sample/10BT")
        if not files:
            raise PreparationError("No FineWeb Parquet files found.")
        random.Random(args.seed).shuffle(files)
        batches = source_batches(
            hub, repo, revision, files, ["text"], args.batch_size, max_rows, visited
        )
    else:
        batches = mammoth_batches(hub, revision, args, max_rows, visited)
    processed = (
        parallel_batches(tokenize_batch, batches, args.workers, tokenizer_file)
        if args.mode == "text"
        else batches
    )
    source_dir = output
    if args.mode == "vlm":
        source_dir = output / ".vlm-records"
        source_dir.mkdir()
    shard_samples = getattr(args, "shard_samples_counts", None)
    writer = LitWriter(
        source_dir,
        args.mode,
        int(args.chunk_mib * MIB),
        shard_samples if args.mode == "text" else None,
    )
    buffer: list[int] = []
    skipped = 0
    last_progress = time.monotonic()
    with (
        (source_dir / "sources.jsonl").open("w") as sources,
        closing(batches),
        closing(processed),
    ):
        for batch in processed:
            for source_id, value in batch:
                if args.mode == "text":
                    sources.write(
                        json.dumps({"source": source_id, "tokens": len(value)}) + "\n"
                    )
                    buffer.extend(value)
                    consumed = 0
                    while (
                        len(buffer) - consumed >= args.sequence_length
                        and writer.count < args.samples
                    ):
                        tokens = np.asarray(
                            buffer[consumed : consumed + args.sequence_length],
                            dtype="<u4",
                        )
                        writer.add(encode_leaves([tokens.tobytes()]))
                        consumed += args.sequence_length
                    buffer = buffer[consumed:]
                elif value is None:
                    skipped += 1
                else:
                    writer.add(value)
                    sources.write(
                        json.dumps({"index": writer.count - 1, "source": source_id})
                        + "\n"
                    )
                if writer.count >= args.samples:
                    break
            if time.monotonic() - last_progress >= 5 or writer.count >= args.samples:
                print(
                    f"Prepared {writer.count:,}/{args.samples:,} samples; read {hub.received / MIB:.1f} MiB",
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = time.monotonic()
            if writer.count >= args.samples:
                break
    if writer.count != args.samples:
        raise PreparationError(
            f"Only {writer.count} valid samples available within the source row limit; "
            "request fewer samples or increase --max-source-rows."
        )
    writer.finish()
    if args.mode == "vlm":
        writer = shuffle_vlm_export(
            source_dir, output, args.seed, writer.chunk_bytes, shard_samples
        )
        shutil.rmtree(source_dir)
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
    }
    return {
        "format_version": 1,
        "mode": args.mode,
        "samples": writer.count,
        "chunks": len(writer.chunks),
        **({"shard_samples": shard_samples} if shard_samples is not None else {}),
        "schema": "uint32[sequence_length]"
        if args.mode == "text"
        else "encoded_images_conversation_metadata",
        "sequence_length": args.sequence_length if args.mode == "text" else None,
        "source": {
            "repo": repo,
            "revision": revision,
            "subset": "sample-10BT"
            if args.mode == "text"
            else "single-image (mammoth_si_10M.json)",
        },
        "tokenizer": tokenizer_info,
        "seed": args.seed,
        "shard_order": "seeded shuffle, rows in source order"
        if args.mode == "text"
        else "bounded valid-record prefix quotas across image archives, missing quotas redistributed; uniform text-only sample; global seeded record shuffle before chunking",
        "workers": args.workers if args.mode == "text" else 1,
        "chunk_bytes": writer.chunk_bytes,
        "compression": None,
        "file_bytes_read": hub.received,
        "max_file_bytes": hub.limit,
        "source_files": visited,
        "skipped_invalid_rows": skipped,
        "package_versions": {name: version(name) for name in PACKAGES},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_files": generator_files(),
        "sha256": checksums,
        "vlm_loader_settings": {"decoded_image_input": False}
        if args.mode == "vlm"
        else None,
        "notes": "Source revisions, sample selection, tokenizer and shard layout are recorded for reproducibility.",
    }


def generator_files():
    scripts = Path(__file__).resolve().parent
    root = scripts.parent
    paths = [
        root / "pyproject.toml",
        root / "uv.lock",
        Path(__file__),
        *sorted((scripts / "data_preparation").glob("*.py")),
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("text", "vlm"))
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory or s3://bucket/prefix; existing datasets are never replaced",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Throughput input records (default: text=524288, vlm=262144)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Text tokenization processes; VLM archives are streamed sequentially",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per text task or VLM write batch",
    )
    parser.add_argument("--chunk-mib", type=float, default=None)
    parser.add_argument(
        "--shard-samples",
        type=Path,
        help="JSON list of positive per-shard record counts; replaces byte-target chunking and supplies --samples",
    )
    parser.add_argument(
        "--max-download-mib",
        type=float,
        help="Cap source reads plus uncached Hub downloads (default: text=8192, vlm=262144 MiB; API metadata excluded)",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        help="Maximum source rows considered, including bounded process read-ahead",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--revision", help="Dataset revision (default: pinned commit for each mode)"
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Ungated tokenizer with tokenizer.json; no model weights are fetched",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help="Tokenizer revision (default: pinned commit for the default tokenizer)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hub cache plus reusable Mammoth annotation index (VLM first run: about 30 GiB disk)",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        help="Optional JSON settings supplied by an experiment launcher",
    )
    parser.add_argument("--images", choices=("encoded", "decoded-rgb"))
    parser.add_argument("--compression", choices=("none", "zstd"))
    parser.add_argument(
        "--shuffle-storage",
        help="Local directory or S3 prefix for temporary VLM records; defaults to local scratch",
    )
    parser.add_argument("--scratch", type=Path, help="Bounded local working directory")
    parser.add_argument(
        "--journal", type=Path, help="Persistent preparation journal for resume"
    )
    parser.add_argument("--scratch-gib", type=float)
    parser.add_argument("--max-spool-gib", type=float)
    parser.add_argument("--max-output-gib", type=float)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--abandon", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--all-objects", action="store_true")
    parser.add_argument("--keep-spool", action="store_true")
    parser.add_argument(
        "--receipt", type=Path, help="Save plan/validation receipt as JSON"
    )
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text()) if args.recipe else {}
    for key, target, unit in (
        ("samples", "samples", 1),
        ("seed", "seed", 1),
        ("workers", "workers", 1),
        ("revision", "revision", 1),
        ("max_source_rows", "max_source_rows", 1),
        ("sequence_length", "sequence_length", 1),
        ("tokenizer", "tokenizer", 1),
        ("tokenizer_revision", "tokenizer_revision", 1),
        ("batch_size", "batch_size", 1),
        ("cache_dir", "cache_dir", 1),
        ("chunk_bytes", "chunk_mib", MIB),
        ("max_download_bytes", "max_download_mib", MIB),
    ):
        if getattr(args, target) is None and key in recipe:
            setattr(args, target, recipe[key] if unit == 1 else recipe[key] / unit)
    args.sequence_length = (
        args.sequence_length if args.sequence_length is not None else 2048
    )
    args.tokenizer = args.tokenizer or TOKENIZER_REPO
    args.batch_size = args.batch_size if args.batch_size is not None else 16
    args.cache_dir = Path(args.cache_dir or ".cache/arena-throughput-data")
    args.images = args.images or recipe.get("image_representation", "encoded")
    args.compression = args.compression or recipe.get("compression", "none")
    args.seed = args.seed if args.seed is not None else 42
    args.workers = (
        args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    )
    args.chunk_mib = args.chunk_mib if args.chunk_mib is not None else 64
    if not args.output.startswith("s3://"):
        if "://" in args.output:
            parser.error("Output must be a local path or s3://bucket/prefix")
        args.output = Path(args.output)
    if args.mode == "text" and args.images != "encoded":
        parser.error("--images applies to VLM datasets")
    args.shard_samples_counts = None
    if args.shard_samples is not None:
        try:
            counts = json.loads(args.shard_samples.read_text())
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read --shard-samples: {error}")
        if (
            not isinstance(counts, list)
            or not counts
            or any(type(n) is not int or n <= 0 for n in counts)
        ):
            parser.error(
                "--shard-samples must contain a nonempty JSON list of positive integers"
            )
        if args.samples is not None and args.samples != sum(counts):
            parser.error("--samples must equal the sum of --shard-samples")
        args.samples = sum(counts)
        args.shard_samples_counts = counts
    if args.max_download_mib is None:
        args.max_download_mib = 8192 if args.mode == "text" else 262144
    if args.samples is None:
        args.samples = 524288 if args.mode == "text" else 262144
    for name in (
        "samples",
        "workers",
        "batch_size",
        "chunk_mib",
        "max_download_mib",
        "sequence_length",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_source_rows is not None and args.max_source_rows <= 0:
        parser.error("--max-source-rows must be positive")
    if args.chunk_mib * MIB >= UINT32_MAX:
        parser.error("--chunk-mib must be less than 4096 (LitData uses uint32 offsets)")
    if sys.byteorder != "little":
        parser.error(
            "The LitData wire format used here requires a little-endian machine"
        )
    return args


def prepare_local(args) -> None:
    args.cache_dir = args.cache_dir or Path(".cache/arena-throughput-data")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise PreparationError(
            "Output already exists; choose a new --output directory."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        os.environ.setdefault("HF_XET_CACHE", str(args.cache_dir.expanduser() / "xet"))
        with closing(
            PublicHub(int(args.max_download_mib * MIB), args.cache_dir)
        ) as hub:
            manifest = prepare(args, temporary, hub)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise PreparationError(
                "Output appeared during generation; choose a new --output directory."
            )
        temporary.rename(output)
        print(
            f"Wrote {manifest['samples']:,} samples in {manifest['chunks']} chunks to {output}"
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    args = parse_args(argv)
    if (
        isinstance(args.output, Path)
        and args.images == "encoded"
        and args.compression == "none"
        and not any(
            (
                args.recipe,
                args.shuffle_storage,
                args.journal,
                args.scratch,
                args.plan,
                args.check_only,
                args.abandon,
                args.receipt,
                args.keep_spool,
                args.all_objects,
                args.existing,
                args.scratch_gib is not None,
                args.max_spool_gib is not None,
                args.max_output_gib is not None,
            )
        )
    ):
        return prepare_local(args)
    from data_preparation.runner import run

    return run(args)


if __name__ == "__main__":
    # Helpers importing prepare_data reuse this exact module, including its
    # exception classes and sampling functions, when invoked as a script.
    sys.modules["prepare_data"] = sys.modules[__name__]
    try:
        main()
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
