"""Checkpoint continuation and measurement boundaries, with real Zephon."""

import json
from copy import deepcopy

import pytest
import torch
from conftest import local_tokenizer

from dataloader_arena.checkpoint import (
    batches_for_steps,
    identity,
    parse_args,
    run_checkpoint,
)
from dataloader_arena.utils.checkpointing import (
    fingerprint,
    prepare_cache,
    read_bytes,
    summarize_checkpoint,
    write_json,
)


def test_step_progress_is_per_dp_rank():
    assert batches_for_steps(100, 128, 4, 8) == 400
    with pytest.raises(ValueError, match="whole batches"):
        batches_for_steps(1, 127, 4, 8)


def test_cache_never_deletes_existing_data(tmp_path):
    (tmp_path / "important").write_text("keep")
    root = prepare_cache(tmp_path, "campaign", 0, cold=True)
    (root / "cached").write_text("keep this too")
    with pytest.raises(ValueError, match="not empty"):
        prepare_cache(tmp_path, "campaign", 0, cold=True)
    assert prepare_cache(tmp_path, "campaign", 0, cold=False) == root
    assert (tmp_path / "important").read_text() == "keep"
    for invalid in ("..", ".", "nested/path", ""):
        with pytest.raises(ValueError):
            prepare_cache(tmp_path, invalid, 0, cold=True)


def test_source_counts_use_saved_mapping_and_all_512_lanes():
    state = {
        "world": {
            "canonical_replicas": 512,
            "dp_degree": 2,
            "mapping": {0: list(range(0, 512, 2)), 1: list(range(1, 512, 2))},
        },
        "inflight": {
            "511": {"0": {"total_samples": 7}},
            "0": {"1": {"components": [(0, [1, 2, 3])]}},
        },
    }
    summary = summarize_checkpoint(state, 2)
    assert len(summary["per_lane"]) == 512
    assert summary["total"] == {"chunks": 2, "source_records": 10}
    assert summary["per_dp_rank"][0]["source_records"] == 3
    assert summary["per_dp_rank"][1]["source_records"] == 7


def test_continuation_digest_detects_content_shape_and_dtype_changes():
    batch = {"tokens": torch.arange(4), "image": [torch.ones(3, 2, 2)], "mask": None}
    assert fingerprint(batch) == fingerprint(deepcopy(batch))
    changed = deepcopy(batch)
    changed["image"][0][0, 0, 0] = 2
    assert fingerprint(batch) != fingerprint(changed)
    assert fingerprint(torch.ones(2, 2)) != fingerprint(torch.ones(4))
    assert fingerprint(torch.ones(4)) != fingerprint(torch.ones(4, dtype=torch.int32))


def test_restore_identity_allows_new_aggregation_but_rejects_recipe_changes():
    def workload(options):
        return identity(
            "factory", options, dataset="same", batch_size=4, seed=42, ranks=8
        )

    options = {
        "pipeline": {"run_id": "a", "aggregate_dir": "a", "flush_every_k_chunks": 8}
    }
    updated = deepcopy(options)
    updated["pipeline"].update(run_id="b", aggregate_dir="b")
    assert workload(options) == workload(updated)
    updated["pipeline"]["flush_every_k_chunks"] = 16
    assert workload(options) != workload(updated)


@pytest.mark.parametrize(
    "workload,attention,execution",
    [
        ("text", None, "threads"),
        ("vlm", "sdpa", "threads"),
        ("vlm", "flash_attention_2", "threads"),
        ("vlm", "sdpa", "process_mtp"),
        ("vlm", "sdpa", "process_mtp_per_lane"),
    ],
)
def test_real_pipeline_checkpoint_roundtrip(
    tmp_path, text_data, vlm_data, monkeypatch, workload, attention, execution
):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: local_tokenizer()
    )
    options = {
        "runner": "threads",
        "skip_loader": True,
        "fetch_parallelism": 1,
        "work_source": {"chunk_size": 2, "exhausted_policy": "repeat"},
        "pipeline": {
            "canonical_replicas": 1,
            "prefetch_batches": 1,
            "default_stage_prefetch": 1,
            "op_queue_capacity": 2,
        },
    }
    if execution.startswith("process_mtp"):
        options["runner"] = "process"
        options["pipeline"].update(mtp_mode=True, mtp_buffer=1, mp_context="spawn")
    if workload == "vlm":
        factory = "dataloader_arena.benchmarks.vlm.zephon:create_loader"
        directory = vlm_data
        options.update(
            vlm={"sequence_length": 512, "attn_implementation": attention},
            packing={"window_size": 2, "min_sequence_length": 1},
            vlm_zephon={
                "fused_transforms": True,
                "process_parallelism": 1,
                "batch_metadata": "preserve_contributors",
            },
        )
        options["pipeline"]["flush_every_k_chunks"] = 2
        if execution == "process_mtp_per_lane":
            options["vlm_zephon"]["shuffle_pack_mode"] = "per_lane"
            options["pipeline"]["canonical_replicas"] = 2
    else:
        factory = "dataloader_arena.benchmarks.load_and_batch.zephon:create_loader"
        directory = text_data
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"loaders": [{"name": "zephon", "factory": factory, "config": options}]}
        )
    )
    checkpoint = str(tmp_path / "checkpoint.pkl")
    common = [
        "--config",
        str(config),
        "--local",
        str(directory),
        "--checkpoint-uri",
        checkpoint,
        "--batch-size",
        "2",
        "--batches",
        "3",
        "--verify-batches",
        "2",
    ]
    for mode in ("fresh", "save", "restore"):
        result = run_checkpoint(
            parse_args(
                common
                + ["--mode", mode, "--json-output", str(tmp_path / f"{mode}.json")]
            )
        )
        assert result["world_size"] == 1
        host = result["ranks"][0]["provenance"]["environment"]["host"]
        assert host["hostname"]
        assert host["memory"]["total"] > 0
        assert "ec2" in host and "system_files" in host
        assert (
            result["ranks"][0]["memory"]["rss_peak_bytes"]
            >= result["ranks"][0]["memory"]["rss_baseline_bytes"]
        )
        if mode == "restore":
            assert (
                result["ranks"][0]["memory_through_replay_completion"]["status"]
                == "unavailable"
            )
            assert result["ranks"][0]["continuation"]["matches"]
            assert (
                result["ranks"][0]["replay_metrics_after_validation"]["status"]
                == "unavailable"
            )
            assert result["max_across_ranks_s"]["restore_to_first_batch"] > 0
        elif mode == "save":
            assert result["ranks"][0]["checkpoint_bytes"] > 0
            assert (
                result["max_across_ranks_s"]["save_end_to_end"]
                >= result["max_across_ranks_s"]["pipeline_checkpoint"]
            )
    manifest = json.loads(read_bytes(checkpoint + ".manifest.json"))
    assert len(manifest["continuation_hashes"][0]) == 2
    assert manifest["versions"]["packages"]["zephon"]["version"]
    if workload == "text":
        index = directory / "index.json"
        index.write_bytes(index.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="dataset index changed"):
            run_checkpoint(
                parse_args(
                    common
                    + [
                        "--mode",
                        "restore",
                        "--json-output",
                        str(tmp_path / "changed.json"),
                    ]
                )
            )


def test_two_rank_cpu_roundtrip(tmp_path, text_data):
    """Exercise real collective aggregation, not mocked distributed bookkeeping."""
    import os
    import subprocess
    import sys

    config = tmp_path / "config.json"
    options = {
        "skip_loader": True,
        "runner": "threads",
        "fetch_parallelism": 1,
        "work_source": {"chunk_size": 2, "exhausted_policy": "repeat"},
        "pipeline": {
            "canonical_replicas": 4,
            "prefetch_batches": 1,
            "default_stage_prefetch": 1,
            "op_queue_capacity": 2,
        },
    }
    config.write_text(
        json.dumps(
            {
                "loaders": [
                    {
                        "name": "zephon",
                        "factory": "dataloader_arena.benchmarks.load_and_batch.zephon:create_loader",
                        "config": options,
                    }
                ]
            }
        )
    )
    env = dict(os.environ)
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    for mode in ("fresh", "save", "restore"):
        report = tmp_path / f"{mode}.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0",
            "--local-addr=127.0.0.1",
            "--rdzv-id=arena-checkpoint-test",
            "--nproc-per-node=2",
            "--module",
            "--",
            "dataloader_arena.checkpoint",
            "--mode",
            mode,
            "--config",
            str(config),
            "--local",
            str(text_data),
            "--batch-size",
            "2",
            "--batches",
            "3",
            "--verify-batches",
            "2",
            "--aggregate-dir",
            str(tmp_path / "aggregation"),
            "--checkpoint-uri",
            str(tmp_path / "state.pkl"),
            "--json-output",
            str(report),
        ]
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=90, env=env
        )
        assert result.returncode == 0, result.stdout + result.stderr
        observations = json.loads(report.read_text())
        assert len(observations["ranks"]) == 2
        if mode == "restore":
            assert all(r["continuation"]["matches"] for r in observations["ranks"])


def test_s3_artifacts_retain_original_boto3_client(monkeypatch):
    import io
    from unittest.mock import Mock

    import dataloader_arena.utils.checkpointing as helpers

    payloads = {}
    client = Mock()
    client.put_object.side_effect = lambda *, Bucket, Key, Body: payloads.update(
        {(Bucket, Key): Body}
    )
    client.get_object.side_effect = lambda *, Bucket, Key: {
        "Body": io.BytesIO(payloads[(Bucket, Key)])
    }
    create_client = Mock(return_value=client)
    monkeypatch.setattr("boto3.client", create_client)
    helpers._artifact_store.cache_clear()
    try:
        write_json("s3://example/checkpoint/manifest.json", {"value": 3})
        assert json.loads(read_bytes("s3://example/checkpoint/manifest.json")) == {
            "value": 3
        }
        create_client.assert_called_once_with("s3")
        client.put_object.assert_called_once()
        client.upload_file.assert_not_called()
    finally:
        helpers._artifact_store.cache_clear()


def test_rss_sampler_retains_first_batch_snapshot_and_later_peak(monkeypatch):
    import dataloader_arena.utils.checkpointing as helpers

    observations = iter([100, 140, 160])
    monkeypatch.setattr(helpers, "_rss_bytes", lambda: next(observations))
    with helpers.RssSampler(interval_s=60) as rss:
        first = rss.snapshot()
    assert first.baseline_bytes == 100
    assert first.incremental_bytes == 40
    assert rss.result()["rss_incremental_bytes"] == 60


def test_checkpoint_uses_common_loader_selection(tmp_path):
    config = tmp_path / "loaders.json"
    config.write_text(
        json.dumps(
            {
                "loaders": [
                    {
                        "name": name,
                        "factory": "dataloader_arena.benchmarks.load_and_batch.zephon:create_loader",
                        "config": {},
                    }
                    for name in ("first", "second")
                ]
            }
        )
    )
    common = [
        "--config",
        str(config),
        "--local",
        str(tmp_path),
        "--mode",
        "fresh",
        "--json-output",
        str(tmp_path / "result.json"),
    ]
    for selection in ([], ["--only", "missing"]):
        with pytest.raises(ValueError, match="select exactly one"):
            run_checkpoint(parse_args(common + selection))


def test_host_cache_is_initialized_once_and_cold_reuse_is_rejected(
    tmp_path, monkeypatch
):
    from argparse import Namespace

    from dataloader_arena.checkpoint import prepare_remote_cache

    args = Namespace(cache_scope="host", cache_policy="cold")
    monkeypatch.setenv("LOCAL_RANK", "0")
    leader = prepare_remote_cache(args, tmp_path, "host-campaign", 0)
    assert leader.name == "host"
    (leader / "data").write_text("shared")
    monkeypatch.setenv("LOCAL_RANK", "1")
    follower = prepare_remote_cache(args, tmp_path, "host-campaign", 1)
    assert follower == leader
    assert (follower / "data").read_text() == "shared"
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(RuntimeError, match="not empty"):
        prepare_remote_cache(args, tmp_path, "host-campaign", 0)
