"""Local/S3 stores and format choices do not change source order or payloads."""

import json
import random
from pathlib import Path
from unittest.mock import patch

import prepare_data as prep
import pytest
from data_preparation.format import decoded_record
from data_preparation.job import Job
from data_preparation.s3 import S3
from data_preparation.storage import Local
from data_preparation.validate import check
from test_prepare_data_storage import Cloud, item, settings


@pytest.mark.parametrize("output_remote", [False, True])
@pytest.mark.parametrize("temporary_remote", [False, True])
@pytest.mark.parametrize("images", ["encoded", "decoded-rgb"])
@pytest.mark.parametrize("compression", ["none", "zstd"])
@pytest.mark.parametrize("counts", [None, [3, 1, 8]])
def test_storage_and_format_combinations(
    tmp_path, output_remote, temporary_remote, images, compression, counts
):
    cloud = Cloud()
    output = (
        S3("s3://bucket/output", cloud) if output_remote else Local(tmp_path / "output")
    )
    temporary = (
        S3("s3://bucket/shuffle", cloud)
        if temporary_remote
        else Local(tmp_path / "shuffle")
    )
    spec = settings() | dict(
        mode="vlm",
        image_representation=images,
        compression=compression,
        shard_samples=counts,
    )
    job = Job(output, tmp_path / "scratch", spec, spool_store=temporary)
    originals = [item(n, () if n % 3 == 0 else ("RGBA",)) for n in range(12)]
    for number, record in enumerate(originals):
        job.accept(number, record, "image")
    job.flush()
    job.convert()
    job.publish()
    result = check(output, spec, all_objects=True)
    assert result["records"] == 12
    # Compare complete raw shards with the original local writer, including
    # exact shuffle, headers and chunk boundaries. No storage-dependent order.
    import zstd

    reference = tmp_path / "reference"
    reference.mkdir()
    writer = prep.LitWriter(reference, "vlm", 1024, counts)
    order = list(range(12))
    random.Random(42).shuffle(order)
    for number in order:
        writer.add(
            decoded_record(originals[number])
            if images == "decoded-rgb"
            else originals[number]
        )
    writer.finish()
    assert result["physical_shards"] == len(writer.chunks)
    actual = json.loads(output.get("index.json"))
    for got, expected in zip(actual["chunks"], writer.chunks, strict=True):
        data = output.get(got["filename"])
        raw = zstd.decompress(data) if compression == "zstd" else data
        assert raw == (reference / expected["filename"]).read_bytes()
    spool_names = [
        json.loads(row[0])["name"]
        for row in job.db.execute("SELECT receipt FROM spools")
    ]
    job.cleanup()
    assert all(temporary.head(name) is None for name in spool_names)
    assert output.head("index.json") is not None
    job.close()


def test_local_output_and_shuffle_need_no_s3_client(tmp_path):
    with patch(
        "data_preparation.storage.S3", side_effect=AssertionError("unexpected S3")
    ):
        from data_preparation.storage import open_store

        output, spool = open_store(tmp_path / "out"), open_store(tmp_path / "spool")
        job = Job(output, tmp_path / "scratch", settings(), spool_store=spool)
        for number in range(12):
            job.accept(number, item(number), "image")
        job.flush()
        job.convert()
        job.publish()
        assert check(output, settings())["valid"]
        job.close()


def test_local_spool_upload_failure_and_resume(tmp_path):
    cloud = Cloud()
    output = S3("s3://bucket/output", cloud)
    spool = Local(tmp_path / "spool")
    spec = settings()
    job = Job(output, tmp_path / "scratch", spec, spool_store=spool)
    for number in range(12):
        job.accept(number, item(number), "image")
    job.flush()
    cloud.fail = "chunk-"
    with pytest.raises(RuntimeError, match="injected"):
        job.convert()
    assert output.head("index.json") is None
    job.close()
    cloud.fail = None
    resumed = Job(output, tmp_path / "scratch", spec, spool_store=spool)
    resumed.convert()
    resumed.publish()
    assert check(output, spec)["valid"]
    resumed.close()


def test_text_record_order_is_not_shuffled_and_layout_is_explicit(tmp_path):
    import numpy as np

    spec = settings(12) | dict(
        mode="text",
        image_representation="encoded",
        compression="none",
        shard_samples=[2, 10],
    )
    output = Local(tmp_path / "out")
    job = Job(
        output, tmp_path / "scratch", spec, spool_store=Local(tmp_path / "shuffle")
    )
    for number in range(12):
        job.accept(
            number,
            prep.encode_leaves(
                [np.asarray([number, number + 1], dtype="<u4").tobytes()]
            ),
            "text",
        )
    job.flush()
    assert list(job.permutation()) == list(range(12))
    job.convert()
    job.publish()
    assert check(output, spec, all_objects=True)["physical_shards"] == 2
    from litdata import StreamingDataset

    dataset = StreamingDataset(input_dir=str(output.path), shuffle=False)
    assert [int(dataset[n][0]) for n in range(12)] == list(range(12))
    job.close()


def test_local_store_refuses_foreign_objects_and_detects_changes(tmp_path):
    store = Local(tmp_path)
    store.owner = "ours"
    receipt = store.put("object", b"original")
    Path(tmp_path / "object").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        store.verify(receipt)
    Path(tmp_path / "foreign").write_bytes(b"keep")
    with pytest.raises(ValueError, match="another preparation"):
        store.delete_owned("foreign")


def test_unified_cli_runs_same_source_with_local_output_and_s3_shuffle(tmp_path):
    import shutil

    import data_preparation.source as source_module
    from test_prepare_data_storage import source

    index, Hub = source(tmp_path)
    hub = Hub()
    hub.repo = lambda *args: type("Revision", (), {"sha": "a" * 40})()
    hub.close = lambda: None
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy2(index, cache / ("mammoth-si-index-v1-" + "a" * 40 + ".sqlite"))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            settings()
            | dict(mode="vlm", image_representation="decoded-rgb", compression="zstd")
        )
    )
    cloud = Cloud()
    output = tmp_path / "output"
    arguments = [
        "vlm",
        "--output",
        str(output),
        "--recipe",
        str(recipe),
        "--cache-dir",
        str(cache),
        "--scratch",
        str(tmp_path / "scratch"),
        "--journal",
        str(tmp_path / "journal"),
        "--shuffle-storage",
        "s3://bucket/temporary",
        "--yes",
    ]
    with (
        patch.object(source_module, "BudgetedHub", return_value=hub),
        patch.object(prep, "mammoth_index", return_value=index),
        patch("data_preparation.storage.S3", side_effect=lambda uri: S3(uri, cloud)),
    ):
        assert prep.main(arguments)["records"] == 12
    assert not cloud.objects  # temporary S3 records cleaned after local validation
    assert prep.main(arguments + ["--check-only", "--all-objects"])["valid"]


def test_recipe_text_options_win_over_defaults_and_explicit_cli_wins(tmp_path):
    from data_preparation.runner import settings as merged

    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            dict(
                mode="text",
                samples=12,
                sequence_length=8,
                batch_size=7,
                tokenizer="example/custom",
                tokenizer_revision="b" * 40,
                cache_dir=str(tmp_path / "cached"),
            )
        )
    )
    args = prep.parse_args(
        ["text", "--recipe", str(recipe), "--output", "s3://bucket/data"]
    )
    spec = merged(args)
    assert (
        spec["sequence_length"],
        spec["batch_size"],
        spec["tokenizer"],
        spec["tokenizer_revision"],
    ) == (8, 7, "example/custom", "b" * 40)
    assert spec["cache_dir"] == str(tmp_path / "cached")
    assert (
        merged(
            prep.parse_args(
                [
                    "text",
                    "--output",
                    "s3://bucket/data",
                    "--tokenizer",
                    "example/custom",
                ]
            )
        )["tokenizer_revision"]
        == "main"
    )
    override = prep.parse_args(
        [
            "text",
            "--recipe",
            str(recipe),
            "--output",
            "s3://bucket/data",
            "--sequence-length",
            "16",
        ]
    )
    assert merged(override)["sequence_length"] == 16


def test_scratch_estimate_rejects_default_local_spool_before_cloud_calls():
    with patch(
        "data_preparation.storage.open_store",
        side_effect=AssertionError("must not touch storage"),
    ):
        result = prep.main(["vlm", "--output", "s3://bucket/data", "--plan"])
        assert not result["fits_scratch_estimate"]
        with pytest.raises(ValueError, match="scratch cap"):
            prep.main(["vlm", "--output", "s3://bucket/data", "--yes"])


def test_check_only_does_not_create_missing_local_directory(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        prep.main(["vlm", "--output", str(missing), "--check-only"])
    assert not missing.exists()


def test_local_output_may_be_relocated_after_publication(tmp_path):
    import shutil

    original = Local(tmp_path / "out")
    job = Job(
        original,
        tmp_path / "scratch",
        settings(),
        spool_store=Local(tmp_path / "spool"),
    )
    for n in range(12):
        job.accept(n, item(n), "image")
    job.flush()
    job.convert()
    job.publish()
    job.close()
    relocated = tmp_path / "relocated"
    shutil.move(original.path, relocated)
    assert check(Local(relocated, create=False), settings(), all_objects=True)["valid"]


def test_overlapping_output_is_rejected_before_storage_writes(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlap"):
        prep.main(["vlm", "--output", str(output), "--scratch", str(output), "--yes"])
    assert not output.exists()


def test_shards_larger_than_reader_bound_fail_before_upload(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/out", cloud)
    spec = settings(1)
    job = Job(store, tmp_path, spec)
    job.accept(0, item(0), "image")
    job.flush()
    job.db.execute("UPDATE records SET decoded_bytes=?", (1024**3,))
    job.db.commit()
    with pytest.raises(ValueError, match="reader limit"):
        job.convert()
    assert not any("chunk-" in key for key in cloud.objects)
    job.close()


def test_vlm_budget_can_increase_without_changing_frozen_prefix_or_rows(tmp_path):
    import argparse

    from data_preparation.mammoth import batches
    from data_preparation.source import SelectedRows
    from test_prepare_data_storage import source

    index, Hub = source(tmp_path)
    args = argparse.Namespace(samples=12, seed=42, batch_size=1)
    cloud = Cloud()
    store = S3("s3://bucket/data", cloud)
    initial = settings() | dict(max_spool_bytes=100)
    job = Job(store, tmp_path / "scratch", initial)

    def ingest(hub):
        with patch.object(prep, "mammoth_index", return_value=index):
            for batch in batches(
                hub,
                "pin",
                args,
                100,
                [],
                state=job.sampler,
                selected=SelectedRows(job.db),
                checkpoint=job.flush,
            ):
                for sid, value in batch:
                    if value:
                        job.accept(int(sid.rsplit("=", 1)[-1]), value, "image")
        job.flush()

    with pytest.raises(ValueError, match="storage cap"):
        ingest(Hub())
    original_budget = job.get("sampler")["image_budget"]
    job.close()
    raised = settings() | dict(
        max_download_bytes=200000,
        scratch_bytes=256 * 1024**3,
        workers=2,
        maximum_shards=100,
    )
    job = Job(store, tmp_path / "scratch", raised)
    bigger = Hub()
    bigger.limit = 200000
    ingest(bigger)
    assert job.sampler["image_budget"] == original_budget
    with patch.object(prep, "mammoth_index", return_value=index):
        expected = [
            int(sid.rsplit("=", 1)[-1])
            for batch in batches(Hub(), "pin", args, 100, [])
            for sid, value in batch
            if value
        ]
    assert [
        row[0]
        for row in job.db.execute("SELECT source_number FROM records ORDER BY id")
    ] == expected
    assert len(job.get("attempt_settings")) == 2
    job.close()


def test_decoded_image_load_failure_is_detected_before_spooling(tmp_path):
    import io

    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (32, 32)).save(data, format="JPEG")
    truncated = data.getvalue()[:-2]
    record = prep.encode_vlm_row(
        dict(
            _images=[truncated],
            _source_id="fixture",
            conversations=[dict(**{"from": "human", "value": "<image> question"})],
        )
    )
    assert record is not None  # The historical header/verify check accepts it.
    cloud = Cloud()
    job = Job(S3("s3://bucket/out", cloud), tmp_path, settings(1))
    with pytest.raises(OSError):
        job.accept(0, record, "image")
    assert job.committed == 0 and not any("spool-" in key for key in cloud.objects)
    job.close()


def test_text_ingestion_resumes_with_larger_cap_and_pinned_custom_tokenizer(tmp_path):
    from types import SimpleNamespace

    import data_preparation.source as source_module

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("deterministic tokenizer fixture")
    requests = []

    class Hub:
        def repo(self, repo, revision, kind="dataset"):
            requests.append((repo, revision, kind))
            return SimpleNamespace(sha="b" * 40 if kind == "model" else "a" * 40)

        def download(self, *args):
            return str(tokenizer)

        def parquet_files(self, *args):
            return []

        def close(self):
            pass

    spec = settings(4) | dict(
        mode="text",
        image_representation="encoded",
        compression="none",
        sequence_length=4,
        tokenizer="custom/tokenizer",
        tokenizer_revision="main",
        batch_size=2,
    )
    output = S3("s3://bucket/data", Cloud())
    job = Job(output, tmp_path / "scratch", spec)
    original_accept = job.accept

    def interrupted(number, record, kind):
        original_accept(number, record, kind)
        job.flush()
        if number == 1:
            raise OSError("test interruption")

    def tokenized(*args):
        yield [("a", list(range(9))), ("b", list(range(9, 18)))]

    with (
        patch.object(source_module, "BudgetedHub", return_value=Hub()),
        patch.object(prep, "source_batches", return_value=(x for x in ())),
        patch.object(prep, "parallel_batches", side_effect=tokenized),
        patch.object(job, "accept", side_effect=interrupted),
    ):
        with pytest.raises(OSError, match="interruption"):
            source_module.ingest(job)
    assert job.committed == 2
    job.close()
    requests.clear()
    job = Job(output, tmp_path / "scratch", spec | dict(max_download_bytes=200000))
    with (
        patch.object(source_module, "BudgetedHub", return_value=Hub()),
        patch.object(prep, "source_batches", return_value=(x for x in ())),
        patch.object(prep, "parallel_batches", side_effect=tokenized),
    ):
        source_module.ingest(job)
    assert job.committed == 4
    assert ("custom/tokenizer", "b" * 40, "model") in requests
    job.convert()
    manifest = job.publish()
    assert manifest["tokenizer"]["revision"] == "b" * 40
    assert check(output, spec)["valid"]
    job.close()


def test_failed_manifest_publication_retries_with_changed_resource_caps(tmp_path):
    cloud = Cloud()
    store = S3("s3://bucket/out", cloud)
    job = Job(store, tmp_path, settings())
    for n in range(12):
        job.accept(n, item(n), "image")
    job.flush()
    job.convert()
    cloud.fail = "index.json"
    with pytest.raises(RuntimeError, match="injected"):
        job.publish()
    frozen = store.get("manifest.json")
    job.close()
    cloud.fail = None
    retry = Job(
        store,
        tmp_path,
        settings() | dict(max_output_bytes=200000, workers=2, max_source_rows=200),
    )
    retry.publish()
    assert store.get("manifest.json") == frozen
    assert check(store, settings())["valid"]
    retry.close()


def test_annotation_retry_traffic_does_not_shrink_selection_budget(tmp_path):
    from data_preparation.source import BudgetedHub

    spec = settings() | dict(max_download_bytes=100000)
    store = S3("s3://bucket/out", Cloud())
    job = Job(store, tmp_path, spec)

    def init(hub, limit, cache_dir):
        hub.limit, hub.received, hub.cache_dir = limit, 0, cache_dir

    with patch.object(prep.PublicHub, "__init__", init):
        hub = BudgetedHub(job)
        hub.in_annotation_download = True
        hub.check_budget(20000)
    job.close()
    resumed = Job(store, tmp_path, spec | dict(max_download_bytes=200000))
    with patch.object(prep.PublicHub, "__init__", init):
        retry = BudgetedHub(resumed)
        retry.in_annotation_download = True
        retry.check_budget(20000)
        assert retry.charged == 40000
        assert retry.image_selection_budget == 80000
    resumed.close()


def test_public_hub_is_anonymous_with_environment_credentials_and_can_reopen(
    tmp_path, monkeypatch
):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import httpx

    requests = []
    clients = []
    original_client = httpx.Client

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "public/source",
                "sha": "a" * 40,
                "private": False,
                "gated": False,
            },
        )

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        kwargs["transport"] = httpx.MockTransport(respond)
        result = original_client(**kwargs)
        clients.append(result)
        return result

    monkeypatch.setenv("HF_TOKEN", "ignored-credential")
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid-proxy:1")
    monkeypatch.setattr(httpx, "Client", client)
    # A second preparation in the same interpreter must not reuse a closed client.
    for _ in range(2):
        hub = prep.PublicHub(1024, tmp_path)
        try:
            assert isinstance(hub.api, HfApi)
            assert hub.repo("public/source", "a" * 40).sha == "a" * 40
        finally:
            hub.close()
    assert (
        len([request for request in requests if "/api/datasets/" in request.url.path])
        == 2
    )
    assert all("authorization" not in request.headers for request in requests)
    assert all(client.is_closed for client in clients)
