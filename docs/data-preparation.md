# Data preparation

`scripts/prepare_data.py` prepares text and VLM datasets. It needs the
`data-preparation` extra. `--inexact` keeps any benchmark extras you already
installed.
Most likely, you will want to call this using the [zephon-vldb-experiments](https://github.com/datologyai/zephon-vldb-experiments) repository.

To supply a recipe, use `--recipe settings.json`.

## Local datasets

```sh
uv run --locked --extra data-preparation --inexact python scripts/prepare_data.py text --output /nvme/data/text
uv run --locked --extra data-preparation --inexact python scripts/prepare_data.py vlm --output /nvme/data/vlm
```

- **Text** comes from FineWeb. Each record is 2,048 uint32 tokens from one
  continuous token stream. Documents are joined without a separator token. The
  source files are read in a seeded order; the records themselves aren't shuffled.
- **VLM** comes from MAmmoTH-VL. Records hold encoded JPEG/PNG images and
  conversations, and are shuffled with the seed.

Output is uncompressed. `--samples`, `--seed`, `--chunk-mib` and the source and
tokenizer revisions control the output. `--shard-samples counts.json` sets exact
per-shard record counts; records are never repeated or padded to fill them.

### VLM record selection

VLM records aren't a uniform sample. Each image archive contributes a quota of
valid records from its start, within a byte budget, and unfilled quotas move to
other archives. Text-only records are sampled separately.

The annotation download counts against that budget the first time it runs. With
a warm annotation cache, the budget for images is larger, which can change the
selected records.

## Formats and storage

```sh
uv run --locked --extra data-preparation --inexact python scripts/prepare_data.py vlm \
  --samples 6144 --output s3://YOUR_BUCKET/data \
  --images decoded-rgb --compression zstd \
  --shuffle-storage /nvme/temporary-records \
  --scratch /nvme/preparation --journal /persistent/preparation --plan
```

`--plan` prints the plan and stops. Without it, the script asks for confirmation;
`--yes` skips the prompt.

- `--images encoded|decoded-rgb` and `--compression none|zstd` are independent.
  Decoding keeps dimensions, text, metadata and order. A decode failure stops the
  job. zstd uses level 4 with one worker, so output is deterministic.
- Output and temporary records can each be local or on S3. Use
  `--shuffle-storage s3://YOUR_BUCKET/temporary-records` when local disk is too
  small. No S3 client is created unless you pass an S3 location.
- Source and tokenizer revisions are recorded in the output.

## Resume preparation

If preparation is interrupted, rerun the same command with the same journal.
Keep the journal on persistent storage. The script needs it to resume or clean up
temporary objects.

Use the same recipe, code, lockfile and storage locations. You can adjust resource
caps, validation guards, worker count and cache location. Raising the download
cap doesn't change the VLM records already selected.

Writes refuse to overwrite different existing content. The final index is
written last, and temporary objects are removed after the output validates.

## Set resource limits

Use `--scratch-gib`, `--max-spool-gib` and `--max-output-gib` to limit storage,
and `--max-download-mib` and `--max-source-rows` to limit source reads. Local
temporary records and the `--cache-dir` annotation cache count toward the
scratch cap.

## Checking and discarding

- `--check-only` validates a dataset. It fully decodes the first, middle and last
  shards; it doesn't hash the whole corpus. `--all-objects` also checks every
  object's identity.
- `--existing` accepts a compatible dataset that has no generation manifest.
- `--abandon` deletes an unfinished preparation's own objects, after confirmation
  or with `--yes`. It refuses published datasets. Use one writer per journal.
