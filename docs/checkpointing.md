# Checkpoint benchmark

`dataloader-checkpoint` measures checkpointing of a CPU input pipeline. No model,
GPU or optimizer runs. It uses the same loader factories and `loaders` JSON as
throughput and TTFB. `--only NAME` selects one loader that can save and restore.

Each mode runs as its own process:

- `fresh` measures startup without a checkpoint.
- `save` pulls `--batches` batches, writes a checkpoint, then hashes the next
  `--verify-batches` batches (default 4) from the same iterator.
- `restore` loads the checkpoint and checks that its first batches match those
  hashes. A mismatch writes the report and exits with an error.

Hashing happens outside the timed windows. Checkpoints are Python pickles, so only
restore files you trust.

Under `torchrun`, each process owns one data-parallel partition. Model or tensor
parallelism and changing the number of ranks on restore are not supported.

## Smoke test

```sh
uv sync --locked --python 3.13 --inexact --extra checkpoint
for mode in fresh save restore; do
  uv run --no-sync dataloader-checkpoint \
    --mode "$mode" --config examples/checkpoint/text-smoke.json \
    --local /path/to/text --batch-size 2 --batches 10 \
    --checkpoint-uri results/checkpoint.pkl \
    --json-output "results/checkpoint-$mode.json"
done
```

`/path/to/text` is a text dataset from `scripts/prepare_data.py text`.

## VLM pipeline on several ranks

[`examples/checkpoint/vlm.json`](../examples/checkpoint/vlm.json) uses SigLIP2
384px preprocessing, 729 image tokens, the Qwen3 tokenizer, per-lane shuffling and
packing, and padded SDPA batches. `image_processor_use_fast=false` selects the PIL
image backend; without it, Transformers picks one itself.

The example doesn't pin the tokenizer or image-processor revisions. Set
`vlm.tokenizer_revision` and `vlm.image_processor_revision` for repeatable
results. zephon-vldb-experiments pins the ones the paper used.

```sh
for mode in fresh save restore; do
  uv run --no-sync torchrun --standalone --nproc-per-node=8 \
    --module -- dataloader_arena.checkpoint \
    --mode "$mode" --config examples/checkpoint/vlm.json \
    --local /path/to/cache-base --remote s3://YOUR_BUCKET/vlm-data \
    --batch-size 4 --optimizer-steps 100 --global-batch-size 128 --cache-scope host \
    --aggregate-dir s3://YOUR_BUCKET/checkpoint-benchmark/aggregation \
    --checkpoint-uri s3://YOUR_BUCKET/checkpoint-benchmark/state.pkl \
    --json-output "results/vlm-checkpoint-$mode.json"
done
```

- `--optimizer-steps 100 --global-batch-size 128` means 400 pulls per rank here
  (`100 × 128 / (4 × 8)`). You can pass `--batches` instead.
- Restore starts from the saved position; its progress options don't move it.
- `--aggregate-dir` must be reachable by every rank: a shared filesystem or an S3
  prefix. Each run uses a fresh subdirectory.
- Use new checkpoint and report paths for each seed or repetition.
- `--seed` sets both the source and packing seeds.

### Caches for remote data

By default each rank gets its own cache under
`LOCAL/.arena-checkpoint-cache/NAMESPACE/rank-RANK`. `--cache-scope host` shares
one cache per host. Each run uses a new namespace and refuses an occupied cache.
For a warm cache, pass `--cache-policy reuse --cache-namespace NAME`.

Local datasets are read directly and the page cache isn't cleared, so local runs
are not cold-cache measurements.

## Pipeline settings

The VLM checkpoint example already sets `batch_metadata=preserve_contributors`
and `shuffle_pack_mode=per_lane`. Batches retain every input's metadata, and each
canonical lane has its own shuffle and packing buffer. The options below let you
change those settings when building a different workload.

Throughput, TTFB and checkpointing share one VLM pipeline builder. Its defaults are
the throughput settings: a 224px image transform, 243 image tokens and flat
FlashAttention batches. The checkpoint example overrides them.

`vlm_zephon.batch_metadata`:

- `historical` (default) keeps only the last input's metadata.
- `preserve_contributors` keeps every input's metadata. Checkpointing requires it,
  or consumed inputs stay marked in flight. The checkpoint CLI selects it by
  default and rejects `historical`.

`vlm_zephon.shuffle_pack_mode`:

- `shared` (default) uses one window buffer.
- `per_lane` gives each canonical lane its own buffer and seeds each window from
  its content. The checkpoint example uses it.
- `zephon` uses Zephon's own shuffle and packing operators.

The two settings are independent, so you can change packing without changing
checkpoint metadata.

## Custom loaders

The factory must return an iterable with `checkpoint()` and `restore(state)`.
It must keep contributor metadata. Continuation hashes support tensors, NumPy
arrays, mappings, lists, tuples, scalars and Zephon `SampleBatch` payloads.

## Report contents

Each report has per-rank observations, the slowest rank per phase, the resolved
configuration, dependency versions, a hash of the input index, cache residency,
continuation hashes and an environment snapshot.

| Quantity | Measurement |
| --- | --- |
| Save latency | Checkpoint call, pickle serialization, rank-zero write and rank sync |
| Checkpoint capture | Wall time of `checkpoint()` |
| Checkpoint size | Uncompressed pickle bytes |
| Retained chunks and records | Counts from the saved in-flight state, per lane and rank |
| Fresh TTFB | Iterator construction plus first pull; graph construction excluded |
| Restore TTFB | Read, deserialize, restore, rank sync, iterator construction and first pull |
| Continuation | Batch hashes against the uninterrupted iterator |
| Memory | Per-rank process-tree RSS: baseline, peak and increase |

The `not_measured` list in each report names quantities that Zephon's public API
doesn't expose, such as aggregation sub-phases and when replay finishes.
Restore memory and latency stop at the first batch, not at the end of replay.

RSS is sampled every 10 ms, summed over the process and its children. It can count
shared pages twice and miss short peaks.

The input-index hash doesn't verify every data chunk. Verify frozen inputs
separately.
