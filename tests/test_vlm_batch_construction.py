"""Lists and flat tensor payloads produce the same training batch."""

import pytest
import torch

from dataloader_arena.benchmarks.vlm.transforms.batch_construction import (
    construct_batch_tensors,
)


@pytest.mark.parametrize("tensor_inputs", [False, True])
def test_flat_batch_preserves_tokens_masking_positions_and_images(tensor_inputs):
    image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    samples = [
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, -100, 3],
            "doc_lengths": [2, 1],
            "images": [image],
        },
        {"input_ids": [4, 5], "labels": [4, 5], "doc_lengths": [2], "images": []},
    ]
    if tensor_inputs:
        samples = [
            dict(
                s,
                **{
                    k: torch.tensor(s[k], dtype=torch.int32)
                    for k in ("input_ids", "labels", "doc_lengths")
                },
            )
            for s in samples
        ]
    batch = construct_batch_tensors(
        samples,
        max_length=8,
        pad_token_id=0,
        allow_cross_document_attention=False,
        attn_implementation="flash_attention_2",
    )
    expected = {
        "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3, 4, 5], dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.long),
        "sequence_lengths": torch.tensor([3, 2], dtype=torch.long),
        "cu_seqlens_q": torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 2, 3, 5], dtype=torch.int32),
    }
    for name, value in expected.items():
        assert batch[name].dtype == value.dtype
        assert torch.equal(batch[name], value)
    assert torch.equal(batch["images"][0], image.unsqueeze(0))
    assert batch["images"][1].numel() == 0
    assert batch["attention_mask"] is None
    assert batch["max_seqlen_q"] == batch["max_seqlen_k"] == 2
    assert batch["batch_size"] == 2
