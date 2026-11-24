"""Durable selection journal, exact permutation and bounded dataset conversion."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import sqlite3
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .format import chunk, config, decoded_record, decoded_size
from .s3 import Scratch, digest, json_bytes


def implementation():
    from importlib.metadata import PackageNotFoundError, version

    from prepare_data import generator_files

    packages = {}
    for name in (
        "numpy",
        "pyarrow",
        "tokenizers",
        "pillow",
        "boto3",
        "zstd",
        "huggingface-hub",
        "ijson",
        "httpx2",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    import platform

    from PIL import features

    return {
        "files": generator_files(),
        "packages": packages,
        "python": platform.python_version(),
        "image_codecs": {
            name: features.version_codec(name)
            for name in features.get_supported_codecs()
        },
    }


class Job:
    def __init__(
        self, store, scratch, settings, journal=None, abandon=False, spool_store=None
    ):
        self.store = store
        self.spool_store = spool_store or store
        self.settings = {
            "mode": "vlm",
            "image_representation": "decoded-rgb",
            "compression": "zstd",
            **settings,
        }
        settings = self.settings
        self.scratch = Scratch(scratch, settings["scratch_bytes"])
        if (
            hasattr(self.spool_store, "path")
            and self.spool_store.path != self.scratch.path
        ):
            self.scratch.roots.append(self.spool_store.path)
        if settings.get("cache_dir"):
            cache = Path(settings["cache_dir"])
            cache.mkdir(parents=True, exist_ok=True)
            self.scratch.roots.append(cache)
        self.journal = Path(journal or self.scratch.path).expanduser().resolve()
        self.journal.mkdir(parents=True, exist_ok=True)
        if self.journal != self.scratch.path:
            self.scratch.roots.append(self.journal)
        self.lock = (self.journal / "prepare.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(self.journal / "journal.sqlite")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS completed (name TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS selected (number INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS spools (id INTEGER PRIMARY KEY, receipt TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY, source_number INTEGER UNIQUE, kind TEXT,
                spool INTEGER, start INTEGER, length INTEGER, sha TEXT, decoded_bytes INTEGER);
            CREATE TABLE IF NOT EXISTS layout (
                id INTEGER PRIMARY KEY, first_record INTEGER, stop_record INTEGER, raw_bytes INTEGER);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, first_record INTEGER, stop_record INTEGER,
                raw_bytes INTEGER, receipt TEXT NOT NULL);
        """)
        data_keys = (
            "mode",
            "samples",
            "seed",
            "revision",
            "chunk_bytes",
            "shard_samples",
            "image_representation",
            "compression",
            "spool_bytes",
            "sequence_length",
            "tokenizer",
            "tokenizer_revision",
            "batch_size",
        )
        identity = {
            "settings": {key: settings[key] for key in data_keys if key in settings},
            "destination": store.uri,
            "shuffle_storage": self.spool_store.uri,
            "implementation": implementation(),
        }
        if abandon:
            if (
                self.get("identity") is None
                or self.get("identity")["destination"] != store.uri
            ):
                raise ValueError(
                    "Abandon requires the existing journal and matching S3 destination"
                )
            identity = self.get("identity")
            if identity["shuffle_storage"] != self.spool_store.uri:
                raise ValueError("Abandon requires the original shuffle storage")
        if self.get("abandoned"):
            raise ValueError("This preparation was abandoned; start with a new journal")
        if self.get("abandoning") and not abandon:
            raise ValueError(
                "Cleanup already began; rerun --abandon with this journal before starting a new preparation"
            )
        if self.get("identity") is None:
            if store.head("index.json") or store.head("manifest.json"):
                raise ValueError(
                    "Destination already contains a dataset; choose a new S3 prefix"
                )
            self.set("identity", identity)
        elif self.get("identity") != identity:
            raise ValueError(
                "Recipe, generator, dependencies or destination changed; use separate scratch and S3 prefixes"
            )
        history = self.get("attempt_settings", [])
        if not history or history[-1] != settings:
            history.append(settings)
            self.set("attempt_settings", history)
        if self.get("owner") is None:
            self.set("owner", str(uuid.uuid4()))
        store.owner = self.get("owner")
        self.spool_store.owner = self.get("owner")
        # A different scratch directory cannot silently adopt an existing prefix.
        store.put(
            "PREPARING.json",
            json_bytes({"identity": identity, "owner": self.get("owner")}),
        )
        self.pending = []
        self.pending_ids = set()
        self.pending_bytes = 0
        self.committed = self.db.execute("SELECT count(*) FROM records").fetchone()[0]
        self.sampler = self.get("sampler", {})

    def close(self):
        self.db.close()
        self.lock.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value))
        )
        self.db.commit()

    def seen(self, number):
        return (
            number in self.pending_ids
            or self.db.execute(
                "SELECT 1 FROM records WHERE source_number=?", (number,)
            ).fetchone()
            is not None
        )

    @property
    def record_count(self):
        return self.committed + len(self.pending)

    def accept(self, number, item, kind):
        from .format import MAX_RECORD_BYTES

        if len(item) > MAX_RECORD_BYTES:
            raise ValueError(
                "Selected encoded record exceeds the preparation bound; selection was not substituted"
            )
        if self.seen(number):
            existing = self.db.execute(
                "SELECT sha FROM records WHERE source_number=?", (number,)
            ).fetchone()
            expected = (
                existing[0]
                if existing
                else next(digest(p[1]) for p in self.pending if p[0] == number)
            )
            if expected != digest(item):
                raise ValueError(
                    "A repeated source record changed; refusing nondeterministic preparation"
                )
            return
        if self.settings["image_representation"] == "decoded-rgb":
            decoded_size(item, validate=True)
        self.pending.append((number, item, kind))
        self.pending_ids.add(number)
        self.pending_bytes += len(item)
        if self.pending_bytes >= self.settings["spool_bytes"]:
            self.flush()

    def flush(self):
        if not self.pending:
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                ("sampler", json.dumps(self.sampler)),
            )
            self.db.commit()
            return
        self.scratch.check(self.pending_bytes * 2 + 16 * 1024**2)
        self.scratch.check(16 * 1024**2, destination=self.journal)
        used = self.get("spool_total_bytes", 0)
        if used + self.pending_bytes > self.settings["max_spool_bytes"]:
            raise ValueError("Temporary shuffle storage cap reached")
        index = self.db.execute("SELECT count(*) FROM spools").fetchone()[0]
        name = f"_preparation/{self.get('owner')}/spool-{index:07d}.bin"
        data = b"".join(p[1] for p in self.pending)
        receipt = self.spool_store.put(name, data)
        # A failed upload leaves the journal uncommitted; immutable source rows are replayed.
        with self.db:
            self.db.execute(
                "INSERT INTO spools VALUES (?,?)", (index, json.dumps(receipt))
            )
            offset = 0
            for i, (number, item, kind) in enumerate(self.pending):
                self.db.execute(
                    "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
                    (
                        self.committed + i,
                        number,
                        kind,
                        index,
                        offset,
                        len(item),
                        digest(item),
                        decoded_size(item)
                        if self.settings["image_representation"] == "decoded-rgb"
                        else len(item),
                    ),
                )
                offset += len(item)
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                ("spool_total_bytes", json.dumps(used + len(data))),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                ("sampler", json.dumps(self.sampler)),
            )
        self.committed += len(self.pending)
        self.pending.clear()
        self.pending_ids.clear()
        self.pending_bytes = 0

    def permutation(self):
        """Same random.Random(seed).shuffle ordering as the local VLM generator.

        A memory-mapped uint32 array replaces millions of Python integers. An
        incomplete shuffle is discarded, never mistaken for a completed order.
        """
        import numpy as np

        target = self.journal / "order.u32"
        count = self.committed
        if count >= 2**32:
            raise ValueError("This permutation supports fewer than 2**32 records")
        if target.exists():
            with target.open("rb") as handle:
                sha = hashlib.file_digest(handle, "sha256").hexdigest()
            if target.stat().st_size != count * 4 or sha != self.get("order_sha256"):
                raise ValueError("Permutation changed")
        else:
            self.scratch.check(count * 4, destination=self.journal)
            temporary = target.with_suffix(".partial")
            order = np.memmap(temporary, dtype="<u4", mode="w+", shape=(count,))
            # Chunked initialization avoids a second corpus-sized metadata array.
            for start in range(0, count, 1_000_000):
                stop = min(count, start + 1_000_000)
                order[start:stop] = np.arange(start, stop, dtype="<u4")
            if self.settings["mode"] == "vlm":
                random.Random(self.settings["seed"]).shuffle(order)
            order.flush()
            del order
            with temporary.open("rb") as handle:
                self.set(
                    "order_sha256", hashlib.file_digest(handle, "sha256").hexdigest()
                )
                os.fsync(handle.fileno())
            temporary.replace(target)
        return np.memmap(target, dtype="<u4", mode="r", shape=(count,))

    def fetch(self, row):
        name, etag, start, length, expected = row
        item = self.spool_store.get(name, start=start, length=length, etag=etag)
        if digest(item) != expected:
            raise ValueError("Temporary source record checksum changed")
        return (
            decoded_record(item)
            if self.settings["image_representation"] == "decoded-rgb"
            else item
        )

    def convert(self):
        if self.committed != self.settings["samples"]:
            raise ValueError("Record target was not reached; no final shards uploaded")
        order = self.permutation()
        self.plan_layout(order)
        start = self.db.execute(
            "SELECT coalesce(max(stop_record),0) FROM chunks"
        ).fetchone()[0]
        # Verify completed shard receipts before resuming past their records.
        with ThreadPoolExecutor(max_workers=16) as pool:
            cursor = self.db.execute("SELECT receipt FROM chunks ORDER BY id")
            while rows := cursor.fetchmany(256):
                list(pool.map(self.store.verify, [json.loads(row[0]) for row in rows]))
        items, first = [], start
        stops = iter(
            self.db.execute(
                "SELECT stop_record FROM layout WHERE stop_record > ? ORDER BY id",
                (start,),
            )
        )
        stop = next(stops, (None,))[0]
        # Only `workers` encoded records are prefetched; no unbounded map queue.
        pending = deque()
        with ThreadPoolExecutor(max_workers=self.settings["workers"]) as pool:
            position = start
            while position < len(order) or pending:
                while position < len(order) and len(pending) < self.settings["workers"]:
                    row = self.db.execute(
                        """SELECT s.receipt,r.start,r.length,r.sha
                        FROM records r JOIN spools s ON s.id=r.spool WHERE r.id=?""",
                        (int(order[position]),),
                    ).fetchone()
                    receipt = json.loads(row[0])
                    pending.append(
                        pool.submit(
                            self.fetch, (receipt["name"], receipt["etag"], *row[1:])
                        )
                    )
                    position += 1
                item = pending.popleft().result()
                items.append(item)
                if first + len(items) == stop:
                    self.write_chunk(first, items)
                    first += len(items)
                    items = []
                    stop = next(stops, (None,))[0]
            if items:
                raise ValueError("Conversion did not reach its planned shard boundary")

    def plan_layout(self, order):
        """Check exact physical shard count before uploading decoded payloads."""
        if not self.get("layout_complete"):
            self.db.execute("DELETE FROM layout")
            first, size, index = 0, 8, 0
            counts = self.settings.get("shard_samples")
            if counts and sum(counts) != len(order):
                raise ValueError("Shard counts do not match the requested record count")
            for position, record in enumerate(order):
                length = self.db.execute(
                    "SELECT decoded_bytes FROM records WHERE id=?", (int(record),)
                ).fetchone()[0]
                if position > first and (
                    position - first == counts[index]
                    if counts
                    else size + 4 + length > self.settings["chunk_bytes"]
                ):
                    self.db.execute(
                        "INSERT INTO layout VALUES (?,?,?,?)",
                        (index, first, position, size),
                    )
                    index += 1
                    first, size = position, 8
                size += 4 + length
            self.db.execute(
                "INSERT INTO layout VALUES (?,?,?,?)", (index, first, len(order), size)
            )
            if (
                self.db.execute("SELECT max(raw_bytes) FROM layout").fetchone()[0]
                > 1023 * 1024**2
            ):
                raise ValueError(
                    "Planned shard exceeds the bounded 1023 MiB reader limit"
                )
            self.set("layout_complete", True)
        count = self.db.execute("SELECT count(*) FROM layout").fetchone()[0]
        if (
            not self.settings["minimum_shards"]
            <= count
            <= self.settings["maximum_shards"]
        ):
            raise ValueError(
                f"Predicted physical shard count {count:,} is outside recipe bounds; no decoded shards uploaded"
            )

    def write_chunk(self, first, items):
        raw_size = 8 + 4 * len(items) + sum(map(len, items))
        self.scratch.check(raw_size * 3 + 16 * 1024**2)
        data, raw_bytes = chunk(items, self.settings["compression"])
        used = self.get("final_total_bytes", 0)
        if used + len(data) > self.settings["max_output_bytes"]:
            raise ValueError("Final output storage cap reached")
        index = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        expected = self.db.execute(
            "SELECT first_record,stop_record,raw_bytes FROM layout WHERE id=?", (index,)
        ).fetchone()
        if expected != (first, first + len(items), raw_bytes):
            raise ValueError(
                "RGB conversion differs from the precomputed physical layout"
            )
        suffix = ".zstd.bin" if self.settings["compression"] == "zstd" else ".bin"
        receipt = self.store.put(f"chunk-0-{index}{suffix}", data)
        with self.db:
            self.db.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?)",
                (
                    index,
                    first,
                    first + len(items),
                    raw_bytes,
                    json.dumps(receipt),
                ),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                ("final_total_bytes", json.dumps(used + len(data))),
            )
        if index % 100 == 0:
            print(
                f"Published {index + 1:,} shards / {first + len(items):,} records",
                flush=True,
            )

    def publish(self):
        count = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if self.committed != self.settings["samples"]:
            raise ValueError("Record target was not reached")
        if (
            not self.settings["minimum_shards"]
            <= count
            <= self.settings["maximum_shards"]
        ):
            raise ValueError(
                f"Physical shard count {count:,} is outside configured bounds; index NOT published"
            )
        path = self.scratch.path / "publication-index.json"
        inventory = self.scratch.path / "publication-inventory.jsonl"
        with path.open("wb") as out, inventory.open("wb") as receipts:
            out.write(
                b'{"config":'
                + json_bytes(
                    config(
                        self.settings["chunk_bytes"],
                        self.settings["compression"],
                        self.settings["mode"],
                    )
                ).strip()
                + b',"chunks":['
            )
            for i, first, stop, raw, value in self.db.execute(
                "SELECT * FROM chunks ORDER BY id"
            ):
                receipt = json.loads(value)
                info = {
                    "filename": receipt["name"],
                    "chunk_bytes": raw,
                    "chunk_size": stop - first,
                    "dim": None,
                }
                out.write((b"," if i else b"") + json_bytes(info).strip())
                receipts.write(json_bytes({**receipt, "records": stop - first}))
            out.write(b"]}\n")
        # Metadata are O(shards), not O(payload). These are typically ~100 MiB.
        inventory_receipt = self.store.put("inventory.jsonl", inventory.read_bytes())
        index_data = path.read_bytes()
        manifest = {
            "version": 1,
            "purpose": "data-preparation",
            "mode": self.settings["mode"],
            "shuffle_storage": self.spool_store.uri,
            "destination": self.store.uri,
            "settings": self.settings,
            "records": self.committed,
            "physical_shards": count,
            "image_representation": self.settings["image_representation"],
            "compression": "zstd:4" if self.settings["compression"] == "zstd" else None,
            "stored_bytes": self.get("final_total_bytes"),
            "uncompressed_bytes": self.db.execute(
                "SELECT sum(raw_bytes) FROM chunks"
            ).fetchone()[0],
            "download_bytes": json.loads(
                (self.journal / "source-budget.json").read_text()
            )
            if (self.journal / "source-budget.json").exists()
            else 0,
            "spool_bytes": self.get("spool_total_bytes"),
            "index_sha256": digest(index_data),
            "order_sha256": self.get("order_sha256"),
            "inventory": inventory_receipt,
            "selection": "archive-prefix quotas/redistribution, text-only sampling, seeded record shuffle"
            if self.settings["mode"] == "vlm"
            else "seeded source file order, source-order rows, continuous token stream",
            "source_files": self.sampler.get("visited", []),
            "attempted_rows": self.sampler.get("attempts"),
            "validation": "server SHA256 receipts for every uploaded object; no full-corpus download",
        }
        manifest["attempt_settings"] = self.get("attempt_settings")
        manifest["tokenizer"] = self.sampler.get("tokenizer")
        manifest["selection_image_budget_bytes"] = self.sampler.get("image_budget")
        manifest["selection_prefix_byte_limit"] = self.sampler.get("prefix_limit")
        manifest["implementation"] = self.get("identity")["implementation"]
        manifest["vlm_loader_settings"] = (
            {
                "decoded_image_input": self.settings["image_representation"]
                == "decoded-rgb"
            }
            if self.settings["mode"] == "vlm"
            else None
        )
        manifest["skipped_invalid_rows"] = (
            self.sampler.get("attempts", self.committed) - self.committed
        )
        frozen = self.get("publication_manifest")
        if frozen is None:
            self.set("publication_manifest", manifest)
        else:
            if frozen["index_sha256"] != manifest["index_sha256"]:
                raise ValueError("Publication index changed")
            manifest = frozen
        self.store.put("manifest.json", json_bytes(manifest))
        # The index is the final commit marker. Failed preparation is not readable
        # as a complete LitData dataset.
        self.store.put("index.json", index_data)
        self.set("published", True)
        return manifest

    def abandon(self):
        """Remove only this unfinished job's objects, including a lost DB commit."""
        if self.get("published") or self.store.head("index.json") is not None:
            raise ValueError("Cannot abandon a published dataset")
        # Commit the destructive transition before any delete. A partial cleanup
        # must never be mistaken for an intact, resumable preparation.
        self.set("abandoning", True)
        removed = 0
        for table, storage in (("spools", self.spool_store), ("chunks", self.store)):
            for (value,) in self.db.execute(f"SELECT receipt FROM {table} ORDER BY id"):
                removed += storage.delete_owned(json.loads(value)["name"])
        next_spool = self.db.execute("SELECT count(*) FROM spools").fetchone()[0]
        next_chunk = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        removed += self.spool_store.delete_owned(
            f"_preparation/{self.get('owner')}/spool-{next_spool:07d}.bin"
        )
        suffix = ".zstd.bin" if self.settings["compression"] == "zstd" else ".bin"
        for name in (
            f"chunk-0-{next_chunk}{suffix}",
            "inventory.jsonl",
            "manifest.json",
        ):
            removed += self.store.delete_owned(name)
        self.store.delete_owned("PREPARING.json")
        self.set("abandoned", True)
        return dict(
            abandoned=True, destination=self.store.uri, deleted_stored_bytes=removed
        )

    def cleanup(self):
        if not self.get("published"):
            raise ValueError("Cannot delete temporary data before publication")
        for (value,) in self.db.execute("SELECT receipt FROM spools ORDER BY id"):
            receipt = json.loads(value)
            if self.spool_store.head(receipt["name"]) is not None:
                self.spool_store.verify(receipt)
                self.spool_store.delete_owned(
                    receipt["name"], version_id=receipt.get("version_id")
                )
        self.set("spool_cleaned", True)
