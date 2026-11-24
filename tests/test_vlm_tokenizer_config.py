from unittest.mock import patch

from dataloader_arena.benchmarks.vlm.config import (
    VLMConfig,
    build_image_transform,
    build_preprocessing_config,
)


def test_preprocessing_passes_explicit_tokenizer_revision():
    revision = "a" * 40
    with patch("transformers.AutoTokenizer.from_pretrained") as load:
        config = build_preprocessing_config(VLMConfig(tokenizer_revision=revision))
    load.assert_called_once_with("Qwen/Qwen3-0.6B", revision=revision)
    assert config.tokenizer is load.return_value


def test_checkpoint_can_select_historical_slow_image_processor():
    with patch("transformers.AutoImageProcessor.from_pretrained") as load:
        transform = build_image_transform(
            VLMConfig(
                image_processor_name="google/siglip2-so400m-patch14-384",
                image_processor_revision="a" * 40,
                image_processor_use_fast=False,
            )
        )
    load.assert_called_once_with(
        "google/siglip2-so400m-patch14-384", revision="a" * 40, use_fast=False
    )
    assert transform.processor is load.return_value


def test_unspecified_image_processor_backend_keeps_library_default():
    with patch("transformers.AutoImageProcessor.from_pretrained") as load:
        build_image_transform(VLMConfig(image_processor_name="example/processor"))
    load.assert_called_once_with("example/processor", revision=None)
