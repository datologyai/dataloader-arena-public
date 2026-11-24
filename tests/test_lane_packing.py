"""Packing boundaries must survive cross-lane interleaving and combined dispatches."""

from zephon import SampleMeta, SampleRecord

from dataloader_arena.benchmarks.vlm.lane_packing import (
    add_lane_shuffle_pack,
    content_seed,
)


def record(lane, index, *, tombstone=False):
    meta = SampleMeta(
        sample_id=(0, lane, index), lane_id=lane, chunk_id=0, chunk_offset=index
    )
    if tombstone:
        meta = meta.with_tombstone(True)
    return SampleRecord(meta=meta, payload={"id": (lane, index), "length": 100})


def callbacks(window_size=3):
    captured = {}

    class Pipeline:
        def stateful_transform(self, name, **options):
            captured.update(options)
            return self

    add_lane_shuffle_pack(
        Pipeline(),
        window_size=window_size,
        max_length=512,
        length_fn=lambda r: r.payload["length"],
        seed=42,
        min_sequence_length=50,
        pack_payloads_fn=lambda items: {"ids": [x["id"] for x in items]},
        parallelism=4,
    )
    return captured


def test_seed_matches_sample_id_xor_and_ignores_delivery_order():
    items = [record(2, 5), record(3, 7)]
    expected = 42 ^ (2 * 31 + 5) ^ (3 * 31 + 7)
    assert content_seed(items, 42) == expected
    assert content_seed(items[::-1], 42) == expected


def test_all_complete_windows_drain_and_tombstones_do_not_fill_windows():
    fns = callbacks()
    items = [record(0, n) for n in range(7)] + [record(1, n) for n in range(4)]
    tombstone = record(0, 99, tombstone=True)
    state, ready = fns["push"](fns["init_state"](), [tombstone] + items)
    assert ready[0] == tombstone
    assert len(ready) == 10
    assert [r.payload["id"] for r in state["lane_buffers"][0]] == [(0, 6)]
    assert [r.payload["id"] for r in state["lane_buffers"][1]] == [(1, 3)]
    packed = fns["transform"](ready)
    assert packed[0] == tombstone
    for r in packed[1:]:
        assert {lane for lane, _ in r.payload["ids"]} == {r.meta.lane_id}
        assert len(r.payload["ids"]) == 3
    tail = fns["transform"](fns["flush"](state))
    assert {r.payload["ids"][0] for r in tail} == {(0, 6), (1, 3)}


def test_packing_is_independent_of_other_lanes_interleaving():
    fns = callbacks()
    lanes = [[record(lane, n) for n in range(6)] for lane in [0, 1]]
    grouped = lanes[0] + lanes[1]
    interleaved = [r for pair in zip(*lanes) for r in pair]

    def packed_ids(items):
        _, ready = fns["push"](fns["init_state"](), items)
        outputs = fns["transform"](ready)
        return sorted((r.meta.lane_id, tuple(r.payload["ids"])) for r in outputs)

    assert packed_ids(grouped) == packed_ids(interleaved)
