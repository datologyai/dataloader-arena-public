"""
Shared batch construction functions for VLM training.

Shared by the Grain and Zephon pipelines.
Split into per-sample extraction and batch-level construction.
"""

from typing import Any, Dict, List

import torch

from .constants import ASSISTANT_ROLE, IGNORE_TOKEN_ID
from .models import Conversation


def extract_sample_data_flat(flat_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract tokens, labels, doc_lengths from flat (tensor-based) payloads."""
    all_input_ids: list = []
    all_labels: list = []
    all_images: list = []
    doc_lengths: list = []

    for p in flat_payloads:
        token_ids = p["token_ids"]
        msg_lengths = p["msg_lengths"].tolist()
        roles = p["roles"]

        offset = 0
        for role, mlen in zip(roles, msg_lengths):
            if mlen > 0:
                chunk = token_ids[offset : offset + mlen]
                all_input_ids.append(chunk)
                if role == ASSISTANT_ROLE:
                    all_labels.append(chunk)
                else:
                    all_labels.append(
                        torch.full((mlen,), IGNORE_TOKEN_ID, dtype=torch.int64)
                    )
                offset += mlen
        doc_lengths.append(p["length"])

        images = p.get("images", [])
        if isinstance(images, list) and len(images) > 0:
            all_images.extend(images)
        elif isinstance(images, torch.Tensor) and images.numel() > 0:
            all_images.append(images)

    return {
        "input_ids": torch.cat(all_input_ids)
        if all_input_ids
        else torch.empty(0, dtype=torch.int64),
        "labels": torch.cat(all_labels)
        if all_labels
        else torch.empty(0, dtype=torch.int64),
        "images": all_images,
        "doc_lengths": torch.tensor(doc_lengths, dtype=torch.int64),
        "length": sum(doc_lengths),
    }


def extract_sample_data(conversations: List[Conversation]) -> Dict[str, Any]:
    """Extract tokens, labels, doc_lengths from packed conversations (per-sample).

    Extract token IDs, labels, images and document lengths for one packed bin.
    """
    packed_sample_input_ids: List[int] = []
    packed_sample_labels: List[int] = []
    packed_sample_images: List[Any] = []
    original_sample_lengths: List[int] = []
    current_pos = 0

    for conversation in conversations:
        conversation_start = current_pos

        for message in conversation.messages:
            token_ids = message.token_ids or []
            token_count = len(token_ids)

            if token_count > 0:
                packed_sample_input_ids.extend(token_ids)

                if message.role == ASSISTANT_ROLE:
                    packed_sample_labels.extend(token_ids)
                else:
                    packed_sample_labels.extend([IGNORE_TOKEN_ID] * token_count)

                current_pos += token_count

        if conversation.images is not None:
            packed_sample_images.extend(conversation.images)
        original_sample_lengths.append(current_pos - conversation_start)

    return {
        "input_ids": packed_sample_input_ids,
        "labels": packed_sample_labels,
        "images": packed_sample_images,
        "doc_lengths": original_sample_lengths,
        "length": len(packed_sample_input_ids),
    }


def construct_batch_tensors(
    samples: List[Dict[str, Any]],
    max_length: int,
    pad_token_id: int,
    allow_cross_document_attention: bool = False,
    attn_implementation: str = "flash_attention_2",
) -> Dict[str, Any]:
    """Stack per-sample data into batch tensors."""
    use_flat = (
        attn_implementation == "flash_attention_2"
        and not allow_cross_document_attention
    )
    if use_flat:
        return _construct_flat_batch(samples, max_length)
    else:
        return _construct_padded_batch(
            samples, max_length, pad_token_id, allow_cross_document_attention
        )


def _construct_padded_batch(
    samples: List[Dict[str, Any]],
    max_length: int,
    pad_token_id: int,
    allow_cross_document_attention: bool,
) -> Dict[str, Any]:
    """Construct padded batch tensors (SDPA format)."""
    batch_size = len(samples)
    batch_input_ids = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long
    )
    batch_labels = torch.full(
        (batch_size, max_length), IGNORE_TOKEN_ID, dtype=torch.long
    )
    batch_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    batch_images: List[Any] = []
    sequence_lengths: List[int] = []

    for batch_idx, sample in enumerate(samples):
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        doc_lengths = sample["doc_lengths"]
        images = sample.get("images", [])

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        if isinstance(doc_lengths, torch.Tensor):
            doc_lengths = doc_lengths.tolist()

        actual_length = len(input_ids)
        if actual_length > max_length:
            raise ValueError(
                f"Packed sample has {actual_length} tokens but max_length is {max_length}. "
                f"Upstream packing/filtering must ensure no sample exceeds max_length "
                f"(Oversized conversations should be filtered before packing; "
                f"Zephon should drop them in the tokenize operator)."
            )
        sequence_lengths.append(actual_length)

        if actual_length > 0:
            batch_input_ids[batch_idx, :actual_length] = input_ids
            batch_labels[batch_idx, :actual_length] = labels

            if not allow_cross_document_attention:
                doc_id, pos = 1, 0
                for doc_len in doc_lengths:
                    end = min(pos + doc_len, max_length)
                    if end > pos:
                        batch_attention_mask[batch_idx, pos:end] = doc_id
                        doc_id += 1
                    pos = end
                    if pos >= max_length:
                        break
            else:
                batch_attention_mask[batch_idx, :actual_length] = 1

        batch_images.append(_collect_images(images))

    total_actual = sum(sequence_lengths)
    total_possible = batch_size * max_length
    packing_efficiency = total_actual / total_possible if total_possible > 0 else 0.0

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "attention_mask": batch_attention_mask,
        "images": batch_images,
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        "packing_efficiency": packing_efficiency,
        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
        "max_seqlen_q": None,
        "max_seqlen_k": None,
    }


def _construct_flat_batch(
    samples: List[Dict[str, Any]], max_length: int
) -> Dict[str, Any]:
    """Construct flat batch tensors (FA2 varlen format)."""
    all_input_ids: List[int] = []
    all_labels: List[int] = []
    all_position_ids: List[int] = []
    cu_seqlens: List[int] = [0]
    batch_images: List[Any] = []
    sequence_lengths: List[int] = []
    max_seqlen = 0

    for sample in samples:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        doc_lengths = sample["doc_lengths"]
        images = sample.get("images", [])

        if isinstance(doc_lengths, torch.Tensor):
            doc_lengths = doc_lengths.tolist()

        for doc_len in doc_lengths:
            if doc_len > 0:
                cu_seqlens.append(cu_seqlens[-1] + doc_len)
                max_seqlen = max(max_seqlen, doc_len)
                all_position_ids.extend(range(doc_len))

        sample_len = len(input_ids)
        if sample_len > max_length:
            raise ValueError(
                f"Packed sample has {sample_len} tokens but max_length is {max_length}. "
                f"Upstream packing/filtering must ensure no sample exceeds max_length "
                f"(Oversized conversations should be filtered before packing; "
                f"Zephon should drop them in the tokenize operator)."
            )
        sequence_lengths.append(sample_len)
        all_input_ids.extend(
            input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        )
        all_labels.extend(
            labels.tolist() if isinstance(labels, torch.Tensor) else labels
        )

        batch_images.append(_collect_images(images))

    batch_size = len(samples)
    packing_efficiency = 1.0

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "attention_mask": None,
        "position_ids": torch.tensor(all_position_ids, dtype=torch.long).unsqueeze(0),
        "images": batch_images,
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        "packing_efficiency": packing_efficiency,
        "cu_seqlens_q": torch.tensor(cu_seqlens, dtype=torch.int32),
        "cu_seqlens_k": torch.tensor(cu_seqlens, dtype=torch.int32),
        "max_seqlen_q": max_seqlen,
        "max_seqlen_k": max_seqlen,
        "batch_size": batch_size,
    }


def _collect_images(images: List[Any]) -> Any:
    """Collect and stack images into tensor."""
    if isinstance(images, torch.Tensor) and images.numel() > 0:
        return images
    if images is not None and len(images) > 0:
        tensors = [
            img for img in images if isinstance(img, torch.Tensor) and img.numel() > 0
        ]
        if tensors:
            return torch.stack(tensors, dim=0)
    return torch.tensor([])
