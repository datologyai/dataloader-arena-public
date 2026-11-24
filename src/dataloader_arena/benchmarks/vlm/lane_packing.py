"""Windowed shuffle/packing that keeps canonical lanes independent."""

from collections import defaultdict

import torch

from .ops import _create_packed_record
from .transforms.packing import shuffle_and_pack


def content_seed(records, seed: int) -> int:
    """Match the checkpoint pipeline's order-independent sample-ID seed."""
    value = seed
    for record in records:
        sample_id = record.meta.sample_id
        if isinstance(sample_id, tuple):
            item_hash = 0
            for part in sample_id:
                item_hash = item_hash * 31 + hash(part)
        else:
            item_hash = hash(sample_id)
        value ^= item_hash
    return value & 0x7FFFFFFF


def add_lane_shuffle_pack(
    pipeline,
    *,
    window_size: int,
    max_length: int,
    length_fn,
    seed: int,
    min_sequence_length: int,
    pack_payloads_fn,
    parallelism: int,
):
    """Drain every complete window per lane; pack each window independently.

    A worker may receive several windows from several lanes in one dispatch.
    Regrouping by lane and window preserves the same packing boundaries even
    when upstream delivery combines or interleaves lanes differently.
    """
    if window_size < 1:
        raise ValueError("packing window_size must be positive")

    def init_state():
        return {"lane_buffers": defaultdict(list)}

    def push(state, records):
        buffers = state["lane_buffers"]
        outputs = []
        for record in records:
            if record.meta.tombstone:
                outputs.append(record)
            else:
                buffers[record.meta.lane_id].append(record)
        for lane, buffer in buffers.items():
            complete = len(buffer) // window_size * window_size
            outputs.extend(buffer[:complete])
            buffers[lane] = buffer[complete:]
        return state, outputs

    def flush(state):
        return [
            record for buffer in state["lane_buffers"].values() for record in buffer
        ]

    def transform(records):
        outputs = []
        lanes = defaultdict(list)
        for record in records:
            if record.meta.tombstone:
                outputs.append(record)
            else:
                lanes[record.meta.lane_id].append(record)
        for records in lanes.values():
            for start in range(0, len(records), window_size):
                window = records[start : start + window_size]
                generator = torch.Generator().manual_seed(content_seed(window, seed))
                bins, _ = shuffle_and_pack(
                    items=window,
                    max_length=max_length,
                    length_fn=length_fn,
                    generator=generator,
                    min_sequence_length=min_sequence_length,
                )
                outputs.extend(
                    _create_packed_record(bin, max_length, pack_payloads_fn, length_fn)
                    for bin in bins
                    if bin.items
                )
        return outputs

    return pipeline.stateful_transform(
        "lane_shuffle_pack",
        init_state=init_state,
        push=push,
        flush=flush,
        transform=transform,
        parallelism=parallelism,
        preserves_cursor_order=False,
    )
