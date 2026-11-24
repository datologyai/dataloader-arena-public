"""Check the binary rewrite and provenance on small, independently read exports."""

import importlib.util
import io
import json
import random
import sqlite3
import struct
import tarfile
import tempfile
import unittest
import zlib
from contextlib import closing, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "prepare_data",
    Path(__file__).resolve().parents[1] / "scripts/prepare_data.py",
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def read_export(path):
    records = []
    index = json.loads((path / "index.json").read_text())
    for chunk in index["chunks"]:
        data = (path / chunk["filename"]).read_bytes()
        count = struct.unpack_from("<I", data)[0]
        offsets = struct.unpack_from(f"<{count + 1}I", data, 4)
        assert offsets[0] == 4 * (count + 2) and offsets[-1] == len(data)
        records.extend(data[a:b] for a, b in zip(offsets, offsets[1:]))
    return records


class ShuffleTests(unittest.TestCase):
    def test_reference_boundaries_preserve_every_record_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = prepare.LitWriter(root, "text", 1, [2, 1, 3])
            originals = [bytes([i]) * (i + 1) for i in range(6)]
            for item in originals:
                writer.add(item)
            writer.finish()
            self.assertEqual(read_export(root), originals)
            self.assertEqual([c["chunk_size"] for c in writer.chunks], [2, 1, 3])
            with self.assertRaisesRegex(prepare.PreparationError, "More records"):
                writer.add(b"extra")

    def test_reference_layout_must_be_filled_and_cannot_overflow_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = prepare.LitWriter(root, "vlm", 64 * prepare.MIB, [2])
            writer.add(b"a")
            with self.assertRaisesRegex(prepare.PreparationError, "Too few"):
                writer.finish()
            with patch.object(prepare, "UINT32_MAX", 20):
                with self.assertRaisesRegex(prepare.PreparationError, "uint32"):
                    writer.add(b"12345")
            self.assertFalse((root / "index.json").exists())

    def test_reference_layout_arguments_supply_count_and_reject_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "shape.json"
            path.write_text("[2, 1, 3]")
            base = [
                "text",
                "--output",
                str(root / "data"),
                "--shard-samples",
                str(path),
            ]
            args = prepare.parse_args(base)
            self.assertEqual(args.samples, 6)
            self.assertEqual(args.shard_samples_counts, [2, 1, 3])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                prepare.parse_args(base + ["--samples", "7"])
            for value in ["[]", "[0]", "[-1]", "[true]", "[1.5]", "{}", "not json"]:
                path.write_text(value)
                with (
                    self.subTest(value=value),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    prepare.parse_args(base)
            self.assertEqual(
                prepare.parse_args(["text", "--output", "unused"]).samples, 524288
            )
            self.assertEqual(
                prepare.parse_args(["vlm", "--output", "unused"]).samples, 262144
            )

    def test_text_preparation_preserves_reference_shape_and_wire_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}")
            output = root / "text"
            output.mkdir()
            args = prepare.parse_args(
                [
                    "text",
                    "--output",
                    str(output),
                    "--samples",
                    "6",
                    "--sequence-length",
                    "2",
                    "--workers",
                    "1",
                ]
            )
            args.shard_samples_counts = [2, 1, 3]
            hub = SimpleNamespace(
                repo=lambda *a: SimpleNamespace(sha="pinned"),
                download=lambda *a: str(tokenizer),
                parquet_files=lambda *a: [dict(path="data.parquet", size=1)],
                received=0,
                limit=100,
            )
            with (
                patch.object(
                    prepare, "source_batches", return_value=([] for _ in range(1))
                ),
                patch.object(
                    prepare,
                    "parallel_batches",
                    return_value=([("row", list(range(12)))] for _ in range(1)),
                ),
                patch.object(prepare, "version", return_value="test"),
            ):
                metadata = prepare.prepare(args, output, hub)
            self.assertEqual(metadata["samples"], 6)
            self.assertEqual(metadata["shard_samples"], [2, 1, 3])
            self.assertEqual(
                [
                    c["chunk_size"]
                    for c in json.loads((output / "index.json").read_text())["chunks"]
                ],
                [2, 1, 3],
            )
            records = read_export(output)
            self.assertEqual(
                [struct.unpack("<3I", data) for data in records],
                [(8, i, i + 1) for i in range(0, 12, 2)],
            )

    def test_mixing_applies_reference_boundaries_after_shuffle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            originals = [bytes([i]) * 25 for i in range(6)]
            writer = prepare.LitWriter(source, "vlm", 64)
            for item in originals:
                writer.add(item)
            writer.finish()
            (source / "sources.jsonl").write_text(
                "".join(
                    json.dumps(dict(index=i, source=str(i))) + "\n" for i in range(6)
                )
            )
            mixed = prepare.shuffle_vlm_export(source, output, 42, 1, [2, 1, 3])
            expected = originals.copy()
            random.Random(42).shuffle(expected)
            self.assertEqual(read_export(output), expected)
            self.assertEqual([c["chunk_size"] for c in mixed.chunks], [2, 1, 3])
            self.assertEqual(read_export(source), originals)

    def test_payloads_provenance_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            writer = prepare.LitWriter(source, "vlm", 180)
            originals = [
                prepare.encode_leaves(
                    [
                        bytes([i]) * 31,
                        f"text {i}".encode(),
                        str(i).encode(),
                        b"url",
                        struct.pack("<q", 1),
                    ]
                )
                for i in range(29)
            ]
            for item in originals:
                writer.add(item)
            writer.finish()
            (source / "sources.jsonl").write_text(
                "".join(
                    json.dumps({"index": i, "source": str(i)}) + "\n" for i in range(29)
                )
            )
            outputs = []
            for name, seed in [("a", 42), ("b", 42), ("c", 67)]:
                dest = root / name
                dest.mkdir()
                prepare.shuffle_vlm_export(source, dest, seed, 180)
                actual = read_export(dest)
                outputs.append(actual)
                self.assertCountEqual(actual, originals)
                provenance = [
                    json.loads(line)
                    for line in (dest / "sources.jsonl").read_text().splitlines()
                ]
                for i, (row, payload) in enumerate(zip(provenance, actual)):
                    self.assertEqual(row["index"], i)
                    self.assertEqual(payload, originals[int(row["source"])])
            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(outputs[0], outputs[2])
            self.assertNotEqual(outputs[0], originals)
            self.assertEqual(read_export(source), originals)

    def test_invalid_offsets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            dest = root / "dest"
            dest.mkdir()
            writer = prepare.LitWriter(source, "vlm", 180)
            writer.add(b"example")
            writer.finish()
            (source / "chunk-0-0.bin").write_bytes(
                struct.pack("<3I", 1, 12, 999) + b"example"
            )
            with self.assertRaisesRegex(
                prepare.PreparationError, "Invalid chunk offsets"
            ):
                prepare.shuffle_vlm_export(source, dest, 42, 180)

    def test_preparation_mixes_records_without_fetching_more(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            originals = [
                prepare.encode_leaves(
                    [
                        b"images",
                        str(i).encode(),
                        str(i).encode(),
                        b"url",
                        struct.pack("<q", 1),
                    ]
                )
                for i in range(29)
            ]
            args = SimpleNamespace(
                mode="vlm",
                revision=None,
                max_source_rows=None,
                samples=29,
                workers=1,
                chunk_mib=180 / prepare.MIB,
                seed=42,
                sequence_length=2048,
                shard_samples_counts=[9, 1, 19],
            )
            hub = SimpleNamespace(
                repo=lambda *args: SimpleNamespace(sha="pinned"),
                received=123,
                limit=456,
            )
            with (
                patch.object(
                    prepare,
                    "mammoth_batches",
                    return_value=(
                        [(str(i), v) for i, v in enumerate(originals)] for _ in range(1)
                    ),
                ),
                patch.object(
                    prepare,
                    "parallel_batches",
                    side_effect=AssertionError("VLM is already encoded"),
                ),
                patch.object(prepare, "version", return_value="test"),
            ):
                manifest = prepare.prepare(args, output, hub)
            self.assertCountEqual(read_export(output), originals)
            self.assertNotEqual(read_export(output), originals)
            self.assertEqual(manifest["file_bytes_read"], 123)
            self.assertEqual(hub.received, 123)
            self.assertFalse((output / ".vlm-records").exists())
            self.assertEqual(manifest["samples"], 29)
            self.assertEqual(manifest["shard_samples"], [9, 1, 19])
            self.assertEqual(
                [
                    c["chunk_size"]
                    for c in json.loads((output / "index.json").read_text())["chunks"]
                ],
                [9, 1, 19],
            )

    def test_archive_reader_returns_a_short_final_read(self):
        remote = io.BytesIO(b"x" * prepare.MIB)
        remote.size = prepare.MIB
        hub = SimpleNamespace(received=0, check_budget=lambda size: None)
        with prepare.BudgetedReader(hub, remote, max_bytes=4096) as reader:
            # tarfile requests large blocks even when only a small prefix is
            # permitted. Useful bytes must be returned before stopping.
            self.assertEqual(reader.read(8 * prepare.MIB), b"x" * 4096)
            self.assertEqual(remote.tell(), 4096)
            with self.assertRaises(prepare.ArchivePrefixLimit):
                reader.read(8 * prepare.MIB)
        self.assertEqual(hub.received, 4096)

    def test_archive_quotas_backfill_invalid_rows_and_enforce_budget(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index.sqlite"
            image = io.BytesIO()
            Image.new("RGB", (8, 8)).save(image, format="PNG")
            content = image.getvalue()
            archives = {}
            with closing(sqlite3.connect(index)) as db, db:
                db.execute(
                    "CREATE TABLE records(image TEXT,row_number INTEGER,payload BLOB)"
                )
                db.execute("CREATE TABLE stats(total INTEGER,text_only INTEGER)")
                number = 0
                for archive in range(3):
                    data = io.BytesIO()
                    with tarfile.open(fileobj=data, mode="w:gz") as tar:
                        for row in range(4):
                            name = f"archive-{archive}/{row}.png"
                            info = tarfile.TarInfo(name)
                            info.size = len(content)
                            tar.addfile(info, io.BytesIO(content))
                            turns = [
                                {
                                    "from": "invalid" if row == 0 else "human",
                                    "value": "<image> question",
                                },
                                {"from": "gpt", "value": "answer"},
                            ]
                            db.execute(
                                "INSERT INTO records VALUES(?,?,?)",
                                (
                                    name,
                                    number,
                                    zlib.compress(
                                        json.dumps({"conversations": turns}).encode()
                                    ),
                                ),
                            )
                            number += 1
                    archives[f"shard-{archive}.tar.gz"] = data.getvalue()
                for row in range(4):
                    turns = [
                        {
                            "from": "invalid" if row == 2 else "human",
                            "value": "question",
                        },
                        {"from": "gpt", "value": "answer"},
                    ]
                    db.execute(
                        "INSERT INTO records VALUES(?,?,?)",
                        (
                            "",
                            number,
                            zlib.compress(
                                json.dumps({"conversations": turns}).encode()
                            ),
                        ),
                    )
                    number += 1
                db.execute("INSERT INTO stats VALUES(16,4)")

            class Hub:
                def __init__(self, limit):
                    self.received = 0
                    self.limit = limit
                    self.api = SimpleNamespace(
                        list_repo_tree=lambda *a, **k: [
                            SimpleNamespace(path=n, size=len(d))
                            for n, d in archives.items()
                        ]
                    )
                    self.fs = SimpleNamespace(open=self.open)

                def open(self, path, *a, **k):
                    data = archives[path.rsplit("/", 1)[-1]]
                    file = io.BytesIO(data)
                    file.size = len(data)
                    return file

                def check_budget(self, size):
                    if self.received + size > self.limit:
                        raise prepare.PreparationError("Download limit reached")

            args = SimpleNamespace(samples=12, seed=42, batch_size=2)

            def run(hub):
                visited = []
                with patch.object(prepare, "mammoth_index", return_value=index):
                    rows = [
                        row
                        for batch in prepare.mammoth_batches(
                            hub, "pinned", args, 64, visited
                        )
                        for row in batch
                    ]
                return rows, visited

            hub = Hub(1024**2)
            rows, visited = run(hub)
            valid = [
                (source, payload) for source, payload in rows if payload is not None
            ]
            self.assertEqual(len(valid), 12)
            self.assertEqual(sum(payload is None for _, payload in rows), 4)
            self.assertEqual([v["rows_read"] for v in visited], [3, 3, 3])
            self.assertEqual(len(visited), 3)
            self.assertLessEqual(hub.received, hub.limit)
            self.assertEqual(sum(v["file_bytes_read"] for v in visited), hub.received)
            self.assertEqual(run(Hub(1024**2))[0], rows)
            # An archive without matching annotations cannot monopolize the
            # budget. Its missing quota is filled from productive prefixes.
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w:gz") as tar:
                unmatched = random.Random(42).randbytes(64000)
                info = tarfile.TarInfo("no-annotations/image.bin")
                info.size = len(unmatched)
                tar.addfile(info, io.BytesIO(unmatched))
            archives["unmatched.tar.gz"] = data.getvalue()
            bounded = Hub(100000)
            rows, visits = run(bounded)
            valid = [
                (source, payload) for source, payload in rows if payload is not None
            ]
            self.assertEqual(len(valid), 12)
            self.assertEqual(len({source for source, _ in valid}), 12)
            missing = next(v for v in visits if v["file"] == "unmatched.tar.gz")
            self.assertTrue(missing["prefix_limit_reached"])
            self.assertEqual(missing["rows_read"], 0)
            self.assertLessEqual(
                missing["file_bytes_read"], missing["prefix_byte_limit"]
            )
            self.assertTrue(any(v["sampling_round"] == 2 for v in visits))
            self.assertLessEqual(bounded.received, bounded.limit)
            self.assertEqual(
                sum(v["file_bytes_read"] for v in visits), bounded.received
            )
            self.assertEqual(run(Hub(100000))[0], rows)

            limited = Hub(10)
            with self.assertRaisesRegex(
                prepare.PreparationError,
                "bounded archive prefixes|Download limit reached",
            ):
                run(limited)
            self.assertLessEqual(limited.received, 10)


if __name__ == "__main__":
    unittest.main()
