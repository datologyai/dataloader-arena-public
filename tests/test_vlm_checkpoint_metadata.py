"""The checkpoint path closes every input; the `historical` mode stays selectable."""

from types import SimpleNamespace

import pytest
import torch
from zephon.types import SampleBatch, SampleMeta, SampleRecord

from dataloader_arena.benchmarks.vlm import ops
from dataloader_arena.benchmarks.vlm.config import VLMConfig, build_preprocessing_config


@pytest.mark.parametrize("preserve", [False, True])
def test_batch_collapse_contributor_contract(monkeypatch, preserve):
    payload = {"input_ids": torch.arange(6)}
    monkeypatch.setattr(ops, "construct_batch_tensors", lambda **kw: payload)
    monkeypatch.setattr(
        ops, "validate_batch", lambda **kw: SimpleNamespace(is_valid=True)
    )
    records = tuple(
        SampleRecord(
            SampleMeta(
                sample_id=i,
                lane_id=0,
                chunk_id=i,
                component_sample_counts={i: 1},
                component_token_counts={i: i + 2},
            ),
            {},
        )
        for i in range(3)
    )
    operator = ops.make_construct_and_validate_batch_fn(
        8, 0, False, "sdpa", None, preserve_contributors=preserve
    )
    output = operator(SampleBatch(records=records)).records[0]
    assert output.payload is payload
    assert output.meta.cursor == records[-1].meta.cursor
    expected = records if preserve else records[-1:]
    assert output.meta.contribution_refs() == tuple(
        ref for r in expected for ref in r.meta.contribution_refs()
    )
    assert output.meta.component_sample_counts == {
        r.meta.sample_id: 1 for r in expected
    }
    assert output.meta.component_token_counts == {
        r.meta.sample_id: r.meta.sample_id + 2 for r in expected
    }


def test_checkpoint_recipe_uses_configured_templates_tokens_and_special_tokens(
    monkeypatch,
):
    import transformers
    from conftest import local_tokenizer

    tokenizer = local_tokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    result = build_preprocessing_config(
        VLMConfig(
            tokens_per_image=729,
            tokens_per_tile=729,
            additional_special_tokens=["<|mm_bos|>", "<|vision_start|>"],
            start_sequence="<|mm_bos|>",
            user_message_template="U:{content}",
            assistant_message_template="A:{content}",
        )
    )
    assert result.tokens_per_image == 729
    assert result.start_sequence == "<|mm_bos|>"
    assert result.wrap_user_message("hi") == "U:hi"
    assert "<|mm_bos|>" in tokenizer.get_vocab()
    assert "<|mm_bos|>" in tokenizer.all_special_tokens


def test_image_processor_recipe_is_worker_safe_and_revision_pinned(monkeypatch):
    from unittest.mock import Mock

    import cloudpickle
    import transformers
    from PIL import Image

    from dataloader_arena.benchmarks.vlm.config import build_image_transform

    processor = Mock(return_value={"pixel_values": torch.ones(1, 3, 4, 4)})
    load = Mock(return_value=processor)
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", load)
    transform = build_image_transform(
        VLMConfig(image_processor_name="local-test", image_processor_revision="a" * 40)
    )
    result = cloudpickle.loads(cloudpickle.dumps(transform))(Image.new("RGB", (4, 4)))
    assert result.shape == (3, 4, 4)
    load.assert_called_once_with("local-test", revision="a" * 40)
