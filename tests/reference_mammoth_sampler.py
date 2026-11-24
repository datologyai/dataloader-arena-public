# Frozen quota sampler before the shared-storage refactor, for differential tests.
import argparse
import json
import random
import sqlite3
import sys
import tarfile
import zlib
from collections.abc import Iterator
from contextlib import closing

from prepare_data import (
    MIB,
    VLM_ANNOTATIONS,
    VLM_REPO,
    ArchivePrefixLimit,
    BudgetedReader,
    PreparationError,
    PublicHub,
    encode_vlm_row,
    mammoth_index,
)


def mammoth_batches(
    hub: PublicHub,
    revision: str,
    args: argparse.Namespace,
    max_rows: int,
    visited: list[dict],
) -> Iterator[list[tuple[str, bytes | None]]]:
    """Take a small valid-record quota from each archive, within the byte cap.

    This gives broad archive coverage, not a uniform image-record sample. Gzip
    archives are read from the beginning and closed when their quota or prefix
    byte limit is met. Missing quotas are redistributed across productive
    archives. Invalid records do not count toward quotas.
    """
    index = mammoth_index(hub, revision)
    files = hub.api.list_repo_tree(
        VLM_REPO, "single_image_data", repo_type="dataset", revision=revision
    )
    archives = sorted(
        [item for item in files if item.path.endswith(".tar.gz")],
        key=lambda item: item.path,
    )
    random.Random(args.seed).shuffle(archives)
    batch = []
    selected: set[int] = set()
    attempts = 0

    def encode(
        number: int, payload: bytes, content: bytes | None
    ) -> tuple[str, bytes | None]:
        nonlocal attempts
        attempts += 1
        if attempts > max_rows:
            raise PreparationError(
                "Source row limit reached before filling archive quotas."
            )
        row = json.loads(zlib.decompress(payload))
        row["_images"] = [content] if content is not None else []
        row["_source_id"] = f"{VLM_REPO}@{revision}/{VLM_ANNOTATIONS}#row={number}"
        return row["_source_id"], encode_vlm_row(row)

    with closing(sqlite3.connect(index)) as db:
        total, text_only = db.execute("SELECT total, text_only FROM stats").fetchone()
        text_count = round(args.samples * text_only / total)
        if text_count:
            # The lightweight row-id list avoids loading all text payloads.
            text_ids = [
                row[0]
                for row in db.execute(
                    "SELECT rowid FROM records WHERE image='' ORDER BY rowid"
                )
            ]
            random.Random(args.seed).shuffle(text_ids)
            accepted = 0
            for rowid in text_ids:
                number, payload = db.execute(
                    "SELECT row_number, payload FROM records WHERE rowid=?", (rowid,)
                ).fetchone()
                pair = encode(number, payload, None)
                batch.append(pair)
                accepted += pair[1] is not None
                if len(batch) >= args.batch_size:
                    yield batch
                    batch = []
                if accepted == text_count:
                    break
            if accepted != text_count:
                raise PreparationError("Not enough valid text-only candidates.")

        image_count = args.samples - text_count
        archives = archives[: min(len(archives), image_count)]
        if image_count and not archives:
            raise PreparationError("No image archives found.")
        remaining = image_count
        # Reserve an equal share of the image-read budget for every archive.
        # A missing annotation prefix must not consume the entire download cap.
        prefix_limit = (hub.limit - hub.received) // max(1, len(archives))
        sampling_round = 0
        while remaining and archives:
            sampling_round += 1
            round_start = remaining
            productive = []
            for position, archive in enumerate(archives):
                if not remaining:
                    break
                archives_left = len(archives) - position
                target = (remaining + archives_left - 1) // archives_left
                before = hub.received
                entry = {
                    "file": archive.path,
                    "sampling_round": sampling_round,
                    "rows_read": 0,
                    "target_rows": target,
                    "file_bytes_read": 0,
                    "prefix_byte_limit": prefix_limit,
                }
                visited.append(entry)
                path = f"datasets/{VLM_REPO}@{revision}/{archive.path}"
                print(
                    f"Reading up to {target:,} valid conversations from hf://{path}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    with (
                        hub.fs.open(
                            path, "rb", size=archive.size, cache_type="none"
                        ) as remote,
                        BudgetedReader(hub, remote, max_bytes=prefix_limit) as reader,
                        tarfile.open(
                            fileobj=reader, mode="r|gz", bufsize=8 * MIB
                        ) as tar,
                    ):
                        for member in tar:
                            if not member.isfile() or member.size > 64 * MIB:
                                continue
                            matches = db.execute(
                                "SELECT row_number, payload FROM records WHERE image=? ORDER BY row_number",
                                (member.name.removeprefix("./"),),
                            ).fetchall()
                            if not matches:
                                continue
                            with closing(tar.extractfile(member)) as image:
                                content = image.read()
                            for number, payload in matches:
                                if number in selected:
                                    continue
                                selected.add(number)
                                pair = encode(number, payload, content)
                                batch.append(pair)
                                if pair[1] is not None:
                                    entry["rows_read"] += 1
                                    remaining -= 1
                                if len(batch) >= args.batch_size:
                                    yield batch
                                    batch = []
                                if entry["rows_read"] == target:
                                    break
                            if entry["rows_read"] == target:
                                break
                except ArchivePrefixLimit:
                    entry["prefix_limit_reached"] = True
                    print(
                        f"Stopped {archive.path} at its prefix read limit "
                        f"({entry['rows_read']:,}/{target:,} valid conversations).",
                        file=sys.stderr,
                        flush=True,
                    )
                finally:
                    entry["file_bytes_read"] = hub.received - before
                if entry["rows_read"] == target:
                    productive.append(archive)
            # Redistribute missing quotas across productive archives. Previously
            # selected row IDs are skipped, so rereading a prefix adds no duplicates.
            archives = productive
            if remaining == round_start:
                break
        if remaining:
            raise PreparationError(
                "Not enough valid conversations within the bounded archive prefixes; "
                "increase --max-download-mib or request fewer samples."
            )
        if batch:
            yield batch
