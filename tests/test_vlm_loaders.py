"""Tests for the VLM preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataloader_arena.context import LoaderConfig, RunParameters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader_config(
    path: str = "/nonexistent", config: dict | None = None
) -> LoaderConfig:
    return LoaderConfig(
        loader_name="test",
        run=RunParameters(
            local_path=Path(path),
            remote_path=None,
            batch_size=2,
            num_workers=2,
        ),
        config=config or {},
    )


def test_vlm_config_defaults():
    from dataloader_arena.benchmarks.vlm.config import VLMConfig

    cfg = VLMConfig.parse({})
    assert cfg.sequence_length == 2048
    assert cfg.attn_implementation == "flash_attention_2"
    assert cfg.enable_dynamic_tiling is False


def test_vlm_config_overrides():
    from dataloader_arena.benchmarks.vlm.config import VLMConfig

    cfg = VLMConfig.parse(
        {
            "sequence_length": 4096,
            "enable_dynamic_tiling": True,
        }
    )
    assert cfg.sequence_length == 4096
    assert cfg.enable_dynamic_tiling is True


def test_vlm_config_rejects_unknown_keys():
    from pydantic import ValidationError

    from dataloader_arena.benchmarks.vlm.config import VLMConfig

    with pytest.raises(ValidationError):
        VLMConfig.parse({"unknown_key": 42})


def test_vlm_zephon_config_defaults():
    from dataloader_arena.benchmarks.vlm.config import VLMZephonConfig

    cfg = VLMZephonConfig.parse({})
    assert cfg.shuffle_pack_mode == "shared"
    assert cfg.fused_transforms is False
    assert cfg.process_parallelism == 24


def test_vlm_zephon_config_overrides():
    from dataloader_arena.benchmarks.vlm.config import VLMZephonConfig

    cfg = VLMZephonConfig.parse(
        {
            "fused_transforms": True,
            "process_parallelism": 8,
        }
    )
    assert cfg.fused_transforms is True
    assert cfg.process_parallelism == 8


def test_vlm_packing_config_defaults():
    from dataloader_arena.benchmarks.vlm.config import VLMPackingConfig

    cfg = VLMPackingConfig.parse({})
    assert cfg.window_size == 64
    assert cfg.seed == 42
    assert cfg.min_sequence_length == 50


def test_vlm_factory_modules_importable():
    """All factory modules can be imported without errors."""
    import importlib

    for name in ("zephon", "grain"):
        module = importlib.import_module(f"dataloader_arena.benchmarks.vlm.{name}")
        assert callable(module.create_loader)


def test_transforms_importable():
    """Arena-local transforms can be imported."""
    from dataloader_arena.benchmarks.vlm.transforms import (
        Conversation,
        Message,
        construct_batch_tensors,
        safe_parse_sample,
        shuffle_and_pack,
        tokenize_conversations,
        validate_batch,
    )

    assert Conversation is not None
    assert Message is not None
    assert all(
        callable(fn)
        for fn in (
            safe_parse_sample,
            tokenize_conversations,
            shuffle_and_pack,
            construct_batch_tensors,
            validate_batch,
        )
    )


def test_constants():
    """Constants are available."""
    from dataloader_arena.benchmarks.vlm.transforms.constants import (
        ASSISTANT_ROLE,
        IGNORE_TOKEN_ID,
        USER_ROLE,
        VISION_START_TOKEN,
    )

    assert USER_ROLE == "user"
    assert ASSISTANT_ROLE == "assistant"
    assert IGNORE_TOKEN_ID == -100
    assert VISION_START_TOKEN == "<|vision_start|>"


class TestPreprocessingConfig:
    def test_settings_and_tokenizer(self, monkeypatch):
        from types import SimpleNamespace

        import transformers

        from dataloader_arena.benchmarks.vlm.config import (
            VLMConfig,
            build_preprocessing_config,
        )

        tokenizer = SimpleNamespace(pad_token_id=0)
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: tokenizer,
        )
        config = build_preprocessing_config(VLMConfig())
        assert config.tokenizer is tokenizer
        assert config.tokens_per_image == config.tokens_per_tile == 243
        assert config.user_message_template.format(content="hello").endswith(
            "hello<|im_end|>\n"
        )
        assert config.enable_dynamic_tiling is False
        assert config.tiling_preprocessor is None
