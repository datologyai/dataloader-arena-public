# Dataloader Arena

Dataloader Arena is a benchmark harness for data loaders. It includes adapters for
[Zephon](https://github.com/datologyai/zephon),
[Grain](https://github.com/google/grain) and
[LitData](https://github.com/Lightning-AI/litdata), with text and multimodal
workloads. You can run an individual benchmark, sweep configurations, or compare
your own loader.

| Benchmark | Command | What it measures |
| --- | --- | --- |
| Throughput | `dataloader-bench` | How quickly a loader produces batches after warmup. |
| Time to first batch | `dataloader-ttfb` | Time from constructing the loader to receiving its first batch. |
| Checkpointing | `dataloader-checkpoint` | Checkpoint size, save and recovery time, memory use and correct continuation. |

The workloads cover fetching and batching records, and a complete VLM pipeline
with image processing, tokenization, shuffling, packing and batching.
Even checkpointing runs without a training model and therefore does not need GPU resources.

## Run a benchmark

Run the examples on a dedicated Linux machine with local NVMe, mounted at
`/mnt/local` below. The harness will make changes to the system configuration to ensure fair benchmarking.
Use `--no-system-tuning` to skip this.
The CPU settings are restored when the run ends.

To run a benchmark, first install [uv](https://docs.astral.sh/uv/) and clone the repository. This example
prepares a small text dataset and compares the three loaders:

```sh
git clone https://github.com/datologyai/dataloader-arena-public.git
cd dataloader-arena-public
uv sync --locked --python 3.13 \
  --extra zephon --extra litdata --extra grain --extra data-preparation

uv run --no-sync python scripts/prepare_data.py text \
  --output /mnt/local/arena-data/text --samples 8192

uv run --no-sync dataloader-bench \
  --config examples/loaders.json --local /mnt/local/arena-data/text \
  --num-workers 4 --batch-size 32 --warmup 1024 --steps 4096 \
  --strict --csv-output results/throughput.csv
```

`--warmup` and `--steps` count **samples**, not batches. The loader definitions are in
[`examples/loaders.json`](examples/loaders.json).
`--strict` makes skipped, failed or incomplete measurements fail the command.

For startup measurements, use the same loader configuration:

```sh
uv run --no-sync dataloader-ttfb \
  --config examples/loaders.json --local /mnt/local/arena-data/text \
  --num-workers 4 --batch-size 32 --repetitions 5 \
  --strict --csv-output results/ttfb.csv
```

Results print to the terminal and are saved at the selected CSV path. Add
`--json-output results/run.json` for detailed measurements and machine information.
Use either command's `--help` for its options.

## Prepare data and run the VLM pipeline

The same preparation script accepts `text` or `vlm`. Text uses FineWeb;
VLM uses MAmmoTH-VL conversations and images.

The default VLM dataset is about 120 GiB. With the annotation cache (about
30 GiB) and temporary shuffle space, allow roughly 275 GiB of free disk.
`--samples` makes a smaller dataset, but the annotation cache is still needed.
No GPU or model weights are needed.

Use `--inexact` when adding extras to keep the benchmark adapters already installed.

```sh
uv sync --locked --python 3.13 --inexact \
  --extra zephon --extra grain --extra vlm-pipeline --extra data-preparation
uv run --no-sync python scripts/prepare_data.py vlm \
  --output /mnt/local/arena-data/vlm
```

Save a loader configuration such as this as `vlm-loaders.json`:

```json
{
  "loaders": [
    {"name": "zephon-vlm", "factory": "dataloader_arena.benchmarks.vlm.zephon:create_loader", "config": {}},
    {"name": "grain-vlm", "factory": "dataloader_arena.benchmarks.vlm.grain:create_loader", "config": {}}
  ]
}
```

Then use it with either measurement command. For example:

```sh
uv run --no-sync dataloader-bench \
  --config vlm-loaders.json --local /mnt/local/arena-data/vlm \
  --batch-size 4 --num-workers 4 --warmup 200 --steps 6000 \
  --packing-stats --strict --csv-output results/vlm.csv
```

`--packing-stats` records usable-token throughput alongside samples/s.

[Data preparation](docs/data-preparation.md) covers S3 output, decoded images,
record selection and resuming.

## Checkpointing

Install the `checkpoint` extra, then run a small save/restore check on the text
dataset prepared above:

```sh
uv sync --locked --python 3.13 --inexact --extra checkpoint
for mode in fresh save restore; do
  uv run --no-sync dataloader-checkpoint \
    --mode "$mode" --config examples/checkpoint/text-smoke.json \
    --local /mnt/local/arena-data/text --batch-size 2 --batches 10 \
    --checkpoint-uri results/checkpoint.pkl \
    --json-output "results/checkpoint-$mode.json"
done
```

`fresh` measures startup without a checkpoint. `save` writes a checkpoint and
records the next batches from the uninterrupted loader; `restore` checks its
output against that continuation and fails on a mismatch.
The [checkpoint guide](docs/checkpointing.md) covers the VLM pipeline, multiple
CPU ranks and S3 inputs.

## Run a sweep

[`examples/sweep.json`](examples/sweep.json) shows how to vary batch size and
other settings. Set its dataset path, then run:

```sh
uv run --locked --extra sweep python sweep.py --config examples/sweep.json \
  --results-dir results/sweep --merge-csv results/merged.csv
```

A sweep can vary loaders, worker counts, batch sizes, seeds, Python versions and
datasets. For TTFB, set `bench_command` to `dataloader-ttfb` and use
`--repetitions` in place of throughput's `--warmup` and `--steps`.

`--num-workers` sets Grain and LitData worker counts; Zephon's operator parallelism
is set in its loader configuration. For seeded runs, Grain uses `pipeline.seed`
from its config, and a `seed` in a LitData config overrides `--seed`.

## Add your own loader

To benchmark your own loader, add its factory and options to a loaders JSON file,
then pass that file with `--config`. The same factories work with throughput,
TTFB and sweeps. Checkpointing additionally requires a loader that can save and
restore its position. The [Zephon text factory](src/dataloader_arena/benchmarks/load_and_batch/zephon.py)
is one example of the factory interface.

## Optional AWS setup

If you need a CPU benchmark machine, this repository includes an EC2 launcher:

```sh
uv run --locked --extra provisioning python scripts/launch_benchmark_instance.py \
  my-benchmark --security-group MY_SECURITY_GROUP --subnet-tag MY_SUBNET \
  --iam-instance-profile MY_INSTANCE_ROLE --key-pair-name MY_KEY --keep-on-stop
```

Use your own AWS resources and credentials. The launcher prepares local NVMe and
prints how to connect. It defaults to `r8id.48xlarge`; `--instance-type` picks
another. `--keep-on-stop` stops the instance on OS shutdown instead of
terminating it. Stopping erases instance-store data.

The instance has no idle shutdown; stop or terminate it in AWS when finished.
As with all cloud instances, stopping ends compute charges, but storage charges can remain.

## Reproduce the Zephon paper

You might have found this repository because of our Zephon paper.
If you want to quickly reproduce our experiments, the
[zephon-vldb-experiments](https://github.com/datologyai/zephon-vldb-experiments) repository
provides the paper recipes, machine setup and reproduction commands.
