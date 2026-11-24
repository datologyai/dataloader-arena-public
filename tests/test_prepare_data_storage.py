"""Differential sampling, real reader compatibility, and interrupted S3 writes."""

import base64
import io
import json
import random
import sqlite3
import struct
import tarfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import prepare_data as cli
import prepare_data as original
import pytest
import reference_mammoth_sampler as reference
from data_preparation.format import chunk, decoded_record, leaves, verify_chunk
from data_preparation.job import Job
from data_preparation.mammoth import batches
from data_preparation.s3 import S3, digest, location
from data_preparation.source import SelectedRows
from data_preparation.validate import check
from PIL import Image


class CloudError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class Cloud:
    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.fail = None
        self.events = []

    def head_object(self, Bucket, Key, **kwargs):
        if Key not in self.objects:
            raise CloudError("404")
        data = self.objects[Key]
        return {
            "ContentLength": len(data),
            "ETag": digest(data),
            "Metadata": self.metadata.get(Key, {}),
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(digest(data))).decode(),
        }

    def put_object(self, Bucket, Key, Body, IfNoneMatch, ChecksumSHA256, **kwargs):
        if self.fail and self.fail in Key:
            raise RuntimeError("injected upload failure")
        if Key in self.objects:
            raise CloudError("412")
        assert IfNoneMatch == "*"
        assert ChecksumSHA256 == base64.b64encode(bytes.fromhex(digest(Body))).decode()
        self.objects[Key] = bytes(Body)
        self.metadata[Key] = kwargs.get("Metadata", {})
        self.events.append(("put", Key))

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        data = self.objects[Key]
        if IfMatch and IfMatch != digest(data):
            raise CloudError("412")
        if Range:
            first, last = map(int, Range.removeprefix("bytes=").split("-"))
            data = data[first : last + 1]
        self.events.append(("get", Key))
        return {"Body": io.BytesIO(data)}

    def delete_object(self, Bucket, Key, **kwargs):
        self.objects.pop(Key, None)
        self.metadata.pop(Key, None)
        self.events.append(("delete", Key))


def settings(count=12):
    return dict(
        samples=count,
        seed=42,
        revision="a" * 40,
        chunk_bytes=1024,
        spool_bytes=1024,
        scratch_bytes=128 * 1024**3,
        workers=3,
        max_download_bytes=100000,
        max_spool_bytes=100000,
        max_output_bytes=100000,
        max_source_rows=100,
        minimum_records=count,
        maximum_records=count,
        minimum_shards=1,
        maximum_shards=count,
        reference_shards_lower_bound=1,
    )


def item(number, modes=("RGBA",)):
    images = []
    for mode in modes:
        out = io.BytesIO()
        Image.new(mode, (3 + number % 3, 4), number % 255).save(out, format="PNG")
        images.append(out.getvalue())
    return original.encode_vlm_row(
        {
            "_images": images,
            "_source_id": f"source#row={number}",
            "conversations": [
                {"from": "human", "value": "<image> " * len(images) + "question"},
                {"from": "gpt", "value": f"answer {number}"},
            ],
        }
    )


def source(root):
    index = root / "annotations.sqlite"
    archives = {}
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    content = image.getvalue()
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE records(image TEXT,row_number INTEGER,payload BLOB)")
        db.execute("CREATE TABLE stats(total INTEGER,text_only INTEGER)")
        number = 0
        for archive in range(3):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w:gz") as tar:
                for row in range(4):
                    name = f"a{archive}/{row}.png"
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
            archives[f"s{archive}.tar.gz"] = data.getvalue()
        for row in range(4):
            turns = [{"from": "invalid" if row == 2 else "human", "value": "question"}]
            db.execute(
                "INSERT INTO records VALUES(?,?,?)",
                (
                    "",
                    number,
                    zlib.compress(json.dumps({"conversations": turns}).encode()),
                ),
            )
            number += 1
        db.execute("INSERT INTO stats VALUES(16,4)")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        unmatched = random.Random(42).randbytes(64000)
        info = tarfile.TarInfo("missing/image.bin")
        info.size = len(unmatched)
        tar.addfile(info, io.BytesIO(unmatched))
    archives["unmatched.tar.gz"] = data.getvalue()

    class Hub:
        def __init__(self):
            self.received, self.limit = 0, 100000
            self.opened = []
            self.api = SimpleNamespace(
                list_repo_tree=lambda *a, **k: [
                    SimpleNamespace(path=n, size=len(d)) for n, d in archives.items()
                ]
            )
            self.fs = SimpleNamespace(open=self.open)

        def open(self, path, *args, **kwargs):
            name = path.rsplit("/", 1)[-1]
            self.opened.append(name)
            remote = io.BytesIO(archives[name])
            remote.size = len(archives[name])
            return remote

        def check_budget(self, size):
            if self.received + size > self.limit:
                raise original.PreparationError("Download limit reached")

    return index, Hub


@pytest.mark.parametrize("seed", [7, 42, 123, 1337])
@pytest.mark.parametrize("batch_size", [1, 2, 9])
def test_sampler_matches_frozen_original_including_invalid_and_redistributed_rows(
    tmp_path, seed, batch_size
):
    index, Hub = source(tmp_path)
    args = SimpleNamespace(samples=12, seed=seed, batch_size=batch_size)
    old_hub, new_hub = Hub(), Hub()
    old_visits, new_visits = [], []
    with (
        patch.object(reference, "mammoth_index", return_value=index),
        patch.object(original, "mammoth_index", return_value=index),
    ):
        expected = list(
            reference.mammoth_batches(old_hub, "pin", args, 100, old_visits)
        )
        actual = list(batches(new_hub, "pin", args, 100, new_visits))
    assert actual == expected  # exact batches, IDs, encoded bytes, and invalids
    assert new_visits == old_visits
    assert new_hub.received == old_hub.received
    if seed == 42:
        assert any(v["sampling_round"] == 2 for v in new_visits)


@pytest.mark.parametrize("interrupt_after", [1, 2, 3, 4, 6, 8, 10, 11])
def test_sampler_resume_preserves_selected_ids_order_and_encoded_bytes(
    tmp_path, interrupt_after
):
    index, Hub = source(tmp_path)
    args = SimpleNamespace(samples=12, seed=42, batch_size=1)
    with patch.object(reference, "mammoth_index", return_value=index):
        expected = [
            p
            for batch in reference.mammoth_batches(Hub(), "pin", args, 100, [])
            for p in batch
            if p[1] is not None
        ]
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    scratch = tmp_path / "scratch"
    job = Job(store, scratch, settings())
    with patch.object(original, "mammoth_index", return_value=index):
        stream = batches(
            Hub(),
            "pin",
            args,
            100,
            [],
            state=job.sampler,
            selected=SelectedRows(job.db),
            checkpoint=job.flush,
        )
        for batch in stream:
            for source_id, value in batch:
                if value:
                    job.accept(int(source_id.rsplit("=", 1)[-1]), value, "image")
            if job.record_count >= interrupt_after:
                break
        stream.close()
    job.close()  # uncommitted progress/invalid selected IDs roll back together
    job = Job(store, scratch, settings())
    with patch.object(original, "mammoth_index", return_value=index):
        for batch in batches(
            Hub(),
            "pin",
            args,
            100,
            [],
            state=job.sampler,
            selected=SelectedRows(job.db),
            checkpoint=job.flush,
        ):
            for source_id, value in batch:
                if value:
                    job.accept(int(source_id.rsplit("=", 1)[-1]), value, "image")
    job.flush()
    actual = []
    for number, value, start, length in job.db.execute(
        "SELECT r.source_number,s.receipt,r.start,r.length FROM records r JOIN spools s ON r.spool=s.id ORDER BY r.id"
    ):
        receipt = json.loads(value)
        actual.append((number, store.get(receipt["name"], start=start, length=length)))
    assert actual == [(int(k.rsplit("=", 1)[-1]), v) for k, v in expected]
    assert job.sampler["phase"] == "complete"
    job.close()


@pytest.mark.parametrize("count", [1, 2, 17, 1003])
def test_memmap_shuffle_is_exactly_list_shuffle(tmp_path, count):
    job = Job(S3("s3://bucket/data", Cloud()), tmp_path, settings(count))
    job.committed = count
    expected = list(range(count))
    random.Random(42).shuffle(expected)
    assert list(job.permutation()) == expected
    assert list(job.permutation()) == expected
    job.close()


def build(job, count=12):
    for number in range(count):
        job.accept(
            number, item(number, () if number % 3 == 0 else ("RGBA", "L")), "image"
        )
    job.flush()
    job.convert()
    return job.publish()


def test_s3_roundtrip_real_litdata_reader_format_order_and_publication(tmp_path):
    from litdata import StreamingDataset

    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    job = Job(store, tmp_path / "scratch", settings())
    manifest = build(job)
    receipt = check(store, settings(), all_objects=True)
    assert receipt["records"] == 12 and receipt["manifest_checked"]
    assert cloud.events[-1][0] == "get"
    index_put = next(
        i for i, e in enumerate(cloud.events) if e == ("put", "data/index.json")
    )
    assert all(
        i < index_put
        for i, e in enumerate(cloud.events)
        if e[0] == "put" and "chunk-" in e[1]
    )
    local = tmp_path / "reader"
    local.mkdir()
    for key, data in cloud.objects.items():
        name = key.removeprefix("data/")
        if "/" not in name:
            (local / name).write_bytes(data)
    dataset = StreamingDataset(input_dir=str(local), shuffle=False)
    order = list(range(12))
    random.Random(42).shuffle(order)
    for position, number in enumerate(order):
        actual = dataset[position]
        expected = leaves(
            decoded_record(item(number, () if number % 3 == 0 else ("RGBA", "L")))
        )
        assert actual["images"] == expected[0]
        assert actual["text"] == expected[1].decode()
        assert actual["metadata"]["url"] == expected[3].decode()
    assert manifest["stored_bytes"] < manifest["uncompressed_bytes"]
    from zephon.debug import DatasetInspector
    from zephon.io import Dataset

    with DatasetInspector(
        Dataset.from_path("rgb", str(local), fmt="litdata")
    ) as reader:
        position = 0
        for shard, entry in enumerate(
            json.loads((local / "index.json").read_text())["chunks"]
        ):
            for row in range(entry["chunk_size"]):
                assert reader.read(shard, row) == dataset[position]
                position += 1
    assert position == 12
    cloud.objects["data/unrelated"] = b"keep"
    job.cleanup()
    assert cloud.objects["data/unrelated"] == b"keep"
    assert not any("/spool-" in k for k in cloud.objects)
    job.close()


def test_failed_spool_upload_preserves_source_and_resume(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    spec = settings(1)
    cloud.fail = "spool-"
    job = Job(store, tmp_path, spec)
    job.accept(0, item(0), "image")
    with pytest.raises(RuntimeError, match="upload failure"):
        job.flush()
    assert job.committed == 0
    assert job.db.execute("SELECT count(*) FROM records").fetchone()[0] == 0
    assert store.head("index.json") is None
    job.close()
    cloud.fail = None
    job = Job(store, tmp_path, spec)
    build(job, 1)
    assert check(store, spec)["valid"]
    job.close()


def test_changed_record_conflict_scale_and_caps_fail_without_index(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    store.put("object", b"first")
    with pytest.raises(ValueError, match="differs"):
        store.put("object", b"second")
    spec = settings(1)
    spec["max_spool_bytes"] = 1
    job = Job(store, tmp_path, spec)
    job.accept(0, item(0), "image")
    with pytest.raises(ValueError, match="storage cap"):
        job.flush()
    assert store.head("index.json") is None
    job.close()


@pytest.mark.parametrize(
    "uri", ["/local", "file:///local", "s3://bucket", "s3://b/a/../x", "s3://b/x?q=1"]
)
def test_storage_validation(uri):
    with pytest.raises(ValueError):
        location(uri)


def test_encoded_images_do_not_pass_decoded_validation():
    data, _ = chunk([item(0)])
    with pytest.raises((ValueError, struct.error)):
        verify_chunk(data, 1)


def test_plan_and_decline_do_not_create_cloud_client_or_scratch(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(settings()))
    argv = ["vlm", "--recipe", str(recipe), "--output", "s3://bucket/data"]
    with patch(
        "data_preparation.storage.open_store",
        side_effect=AssertionError("must not contact cloud"),
    ):
        assert cli.main(argv + ["--plan"])["records"] == 12
        with (
            patch.object(cli.sys.stdin, "isatty", return_value=False),
            pytest.raises(ValueError, match="explicit confirmation"),
        ):
            cli.main(argv + ["--scratch", str(tmp_path / "scratch")])
    assert not (tmp_path / "scratch").exists()


def test_recipe_validation_is_not_disabled_by_yes(tmp_path):
    spec = settings()
    spec["samples"] = 0
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(spec))
    with pytest.raises((ValueError, SystemExit)):
        cli.main(["vlm", "--recipe", str(recipe), "--output", "s3://b/data", "--yes"])


def test_exact_layout_guard_fails_before_decoded_uploads(tmp_path):
    spec = settings()
    spec.update(chunk_bytes=1024**2, minimum_shards=2)
    cloud = Cloud()
    job = Job(S3("s3://bucket/data", cloud), tmp_path, spec)
    for number in range(12):
        job.accept(number, item(number), "image")
    job.flush()
    with pytest.raises(ValueError, match="Predicted physical shard count"):
        job.convert()
    assert not any("/chunk-" in key for key in cloud.objects)
    assert job.store.head("index.json") is None
    job.close()


def test_failed_decoded_upload_resumes_without_replacing_completed_shards(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    job = Job(store, tmp_path, settings())
    for number in range(12):
        job.accept(number, item(number), "image")
    job.flush()
    cloud.fail = "chunk-0-2."
    with pytest.raises(RuntimeError, match="upload failure"):
        job.convert()
    first = {k: v for k, v in cloud.objects.items() if "/chunk-" in k}
    assert len(first) == 2
    assert job.store.head("index.json") is None
    job.close()
    cloud.fail = None
    job = Job(store, tmp_path, settings())
    job.convert()
    job.publish()
    assert check(store, settings())["valid"]
    assert all(cloud.objects[k] == v for k, v in first.items())
    job.close()


@pytest.mark.parametrize("modes", [(), ("RGBA",), ("L", "RGB")])
def test_raw_pixels_equal_the_existing_encoded_image_reader(modes):
    from dataloader_arena.benchmarks.vlm.transforms.decode import (
        deserialize_encoded_image_array,
        deserialize_pil_array,
    )

    encoded = item(4, modes)
    old = deserialize_encoded_image_array(leaves(encoded)[0])
    new = deserialize_pil_array(leaves(decoded_record(encoded))[0])
    assert [(im.size, im.mode, im.tobytes()) for im in new] == [
        (im.size, im.mode, im.tobytes()) for im in old
    ]


def test_persistent_metadata_survives_loss_of_scratch(tmp_path):
    import shutil

    store = S3("s3://bucket/data", Cloud())
    scratch, journal = tmp_path / "scratch", tmp_path / "persistent"
    job = Job(store, scratch, settings(), journal=journal)
    for number in range(12):
        job.accept(number, item(number), "image")
    job.flush()
    job.close()
    shutil.rmtree(scratch)
    job = Job(store, scratch, settings(), journal=journal)
    assert job.committed == 12
    job.convert()
    job.publish()
    assert check(store, settings())["valid"]
    job.close()


def test_failed_response_body_retries_the_same_conditional_range():
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    receipt = store.put("spool", b"abcdefgh")
    calls = []
    get = cloud.get_object

    class BrokenBody(io.BytesIO):
        def read(self, *args):
            raise OSError("interrupted body")

    def response(**kwargs):
        calls.append(kwargs)
        result = get(**kwargs)
        if len(calls) == 1:
            result["Body"] = BrokenBody()
        return result

    with patch.object(cloud, "get_object", side_effect=response), patch("time.sleep"):
        assert store.get("spool", start=2, length=3, etag=receipt["etag"]) == b"cde"
    assert len(calls) == 2 and calls[0] == calls[1]


def test_small_journal_volume_does_not_need_annotation_allocation(tmp_path):
    from data_preparation.s3 import Scratch

    scratch = Scratch(tmp_path / "scratch", 128 * 1024**3)
    journal = tmp_path / "journal"
    journal.mkdir()
    scratch.roots.append(journal)

    def usage(root):
        return SimpleNamespace(free=(2 if Path(root) == journal else 100) * 1024**3)

    with patch("shutil.disk_usage", side_effect=usage):
        scratch.check(32 * 1024**3)
        with pytest.raises(ValueError, match="free disk"):
            scratch.check(3 * 1024**3, destination=journal)


def test_prepopulated_visited_list_matches_original(tmp_path):
    index, Hub = source(tmp_path)
    args = SimpleNamespace(samples=12, seed=42, batch_size=2)
    a, b = [{"previous_source": "unchanged"}], [{"previous_source": "unchanged"}]
    with (
        patch.object(reference, "mammoth_index", return_value=index),
        patch.object(original, "mammoth_index", return_value=index),
    ):
        expected = list(reference.mammoth_batches(Hub(), "pin", args, 100, a))
        actual = list(batches(Hub(), "pin", args, 100, b))
    assert actual == expected and a == b


def test_durable_sampler_rejects_multirow_batches(tmp_path):
    with pytest.raises(ValueError, match="batch_size=1"):
        list(
            batches(
                None,
                "pin",
                SimpleNamespace(batch_size=2),
                100,
                [],
                checkpoint=lambda: None,
            )
        )


def test_layout_and_chunk_bytes_match_existing_litwriter(tmp_path):
    import zstd

    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    spec = settings()
    job = Job(store, tmp_path / "scratch", spec)
    build(job)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    writer = original.LitWriter(reference_dir, "vlm", spec["chunk_bytes"])
    order = list(range(12))
    random.Random(42).shuffle(order)
    for number in order:
        writer.add(
            decoded_record(item(number, () if number % 3 == 0 else ("RGBA", "L")))
        )
    writer.finish()
    index = json.loads(store.get("index.json"))
    assert len(index["chunks"]) == len(writer.chunks)
    for actual, expected in zip(index["chunks"], writer.chunks, strict=True):
        assert {k: v for k, v in actual.items() if k != "filename"} == {
            k: v for k, v in expected.items() if k != "filename"
        }
        assert (
            zstd.decompress(store.get(actual["filename"]))
            == (reference_dir / expected["filename"]).read_bytes()
        )
    job.close()


def test_budget_reservation_is_persistent_and_caps_survive_restart(tmp_path):
    from data_preparation.source import BudgetedHub

    job = Job(S3("s3://bucket/data", Cloud()), tmp_path, settings())

    def init(hub, limit, cache_dir):
        hub.limit, hub.received, hub.cache_dir = limit, 0, cache_dir

    with patch.object(original.PublicHub, "__init__", init):
        hub = BudgetedHub(job)
        hub.check_budget(1000)
        hub.received += 1000
        restarted = BudgetedHub(job)
        assert restarted.charged == 1000 and restarted.limit == 99000
        restarted.check_budget(99000)
        restarted.received += 99000
        with pytest.raises(original.PreparationError, match="Download limit"):
            restarted.check_budget(1)
    assert json.loads((job.journal / "source-budget.json").read_text()) == 100000
    job.close()


def test_ingest_reuses_shared_sampler_with_durable_state(tmp_path):
    import data_preparation.source as source_module

    index, Hub = source(tmp_path)
    hub = Hub()
    hub.repo = lambda *args: SimpleNamespace(sha="a" * 40)
    hub.close = lambda: None
    job = Job(S3("s3://bucket/data", Cloud()), tmp_path / "scratch", settings())
    with (
        patch.object(source_module, "BudgetedHub", return_value=hub),
        patch.object(original, "mammoth_index", return_value=index),
        patch.object(job.scratch, "check"),
    ):
        source_module.ingest(job)
    assert job.committed == 12 and job.sampler["phase"] == "complete"
    assert job.sampler["attempts"] == 16
    job.close()


def test_abandon_removes_owned_objects_and_uncommitted_upload_only(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    cloud.objects["data/unrelated"] = b"keep"
    job = Job(store, tmp_path, settings())
    for number in range(4):
        job.accept(number, item(number), "image")
    job.flush()
    next_id = job.db.execute("SELECT count(*) FROM spools").fetchone()[0]
    orphan = f"_preparation/{job.get('owner')}/spool-{next_id:07d}.bin"
    store.put(orphan, b"uploaded before receipt commit")
    assert job.abandon()["abandoned"]
    assert set(cloud.objects) == {"data/unrelated"}
    job.close()


def test_abandon_refuses_published_dataset_and_foreign_object(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    job = Job(store, tmp_path, settings())
    build(job)
    with pytest.raises(ValueError, match="published"):
        job.abandon()
    cloud.objects["data/foreign"] = b"keep"
    with pytest.raises(ValueError, match="another preparation"):
        store.delete_owned("foreign")
    job.close()


def test_interrupted_abandon_blocks_resume_but_can_finish_cleanup(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    job = Job(store, tmp_path, settings())
    for number in range(4):
        job.accept(number, item(number), "image")
    job.flush()
    delete = store.delete_owned
    calls = 0

    def fail_after_first_delete(name):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted cleanup")
        return delete(name)

    with patch.object(store, "delete_owned", side_effect=fail_after_first_delete):
        with pytest.raises(OSError, match="interrupted cleanup"):
            job.abandon()
    assert job.get("abandoning") and not job.get("abandoned")
    job.close()
    with pytest.raises(ValueError, match="Cleanup already began"):
        Job(S3("s3://bucket/data", cloud), tmp_path, settings())
    resumed = Job(S3("s3://bucket/data", cloud), tmp_path, settings(), abandon=True)
    assert resumed.abandon()["abandoned"]
    assert not cloud.objects
    resumed.close()


def test_owned_delete_preserves_s3_version_identity(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    store.owner = "our-preparation"
    store.put("chunk", b"content")
    head = cloud.head_object

    def versioned_head(**kwargs):
        return head(**kwargs) | {"VersionId": "exact-version"}

    with (
        patch.object(cloud, "head_object", side_effect=versioned_head),
        patch.object(cloud, "delete_object", wraps=cloud.delete_object) as delete,
    ):
        assert store.delete_owned("chunk") == len(b"content")
    delete.assert_called_once_with(
        Bucket="bucket", Key="data/chunk", VersionId="exact-version"
    )
