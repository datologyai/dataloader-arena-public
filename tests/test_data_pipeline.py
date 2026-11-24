"""Preparation schemas are readable by the loaders and VLM pipelines."""

import pytest
from conftest import local_tokenizer

from dataloader_arena import throughput, ttfb
from dataloader_arena.context import LoaderConfig, RunParameters
from dataloader_arena.harness import _shutdown_loader, import_factory


@pytest.mark.parametrize(
    "backend,execution",
    [
        ("litdata", "inline"),
        ("litdata", "process"),
        ("grain", "inline"),
        ("grain", "process"),
        ("zephon", "threads"),
        ("zephon", "process"),
    ],
)
def test_prepared_text_supports_both_metrics(text_data, backend, execution):
    factory = import_factory(
        f"dataloader_arena.benchmarks.load_and_batch.{backend}:create_loader"
    )
    options = {"skip_loader": True} if backend == "zephon" else {}
    workers = 1 if execution == "process" and backend != "zephon" else 0
    if backend == "zephon":
        options["runner"] = execution
    if backend == "grain" and execution == "process":
        options["pipeline"] = {"prefetch": {"mode": "process", "buffer_size": 1}}
    config = LoaderConfig(
        backend, RunParameters(text_data, None, 2, workers, seed=42), options
    )
    result, _ = throughput.run_benchmark(
        factory, config, warmup_steps=1, measure_steps=2
    )
    assert result.steps == 2
    assert result.total_time > 0
    trial, _ = ttfb.measure_ttfb(factory, config)
    assert trial.total_ttfb >= trial.iter_to_batch > 0


@pytest.mark.parametrize(
    "backend,execution",
    [
        ("zephon", "threads"),
        ("zephon", "process"),
        ("grain", "inline"),
        ("grain", "process"),
    ],
)
def test_full_vlm_pipeline_reads_prepared_schema(
    vlm_data, backend, execution, monkeypatch
):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: local_tokenizer(),
    )
    factory = import_factory(f"dataloader_arena.benchmarks.vlm.{backend}:create_loader")
    options = {
        "vlm": {
            "sequence_length": 512,
            "decoded_image_input": False,
            "attn_implementation": "flash_attention_2",
        },
        "packing": {"window_size": 4, "min_sequence_length": 1},
    }
    if backend == "zephon":
        options.update(
            {
                "fetch_parallelism": 1,
                "runner": execution,
                "vlm_zephon": {"process_parallelism": 1, "fused_transforms": True},
            }
        )
    workers = 1 if execution == "process" and backend == "grain" else 0
    config = LoaderConfig(
        backend, RunParameters(vlm_data, None, 1, workers, seed=42), options
    )
    loader = iterator = None
    try:
        loader = factory(config)
        iterator = iter(loader)
        batch = next(iterator)
        assert batch["input_ids"].numel() > 0
        assert batch["images"][0].numel() > 0
    finally:
        _shutdown_loader(loader, iterator)
    result, _ = throughput.run_benchmark(
        factory, config, warmup_steps=0, measure_steps=1, packing_stats=True
    )
    assert result.steps == 1
    assert result.total_usable_tokens > 0
    trial, _ = ttfb.measure_ttfb(factory, config)
    assert trial.total_ttfb > 0
