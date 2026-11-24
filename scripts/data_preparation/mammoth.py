"""The shared MAmmoTH quota sampler, with optional durable preparation state.

Storage and checkpointing don't affect which records are chosen. The choice
depends on the source, seed, sample count and byte budget. A cold annotation
cache charges its download to the budget, so it can select different records
than a warm one.
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
import tarfile
import zlib
from contextlib import closing


def batches(
    hub,
    revision,
    args,
    max_rows,
    visited,
    *,
    state=None,
    selected=None,
    checkpoint=None,
    helpers=None,
):
    if helpers is None:
        import prepare_data

        helpers = vars(prepare_data)
    (
        MIB,
        VLM_ANNOTATIONS,
        VLM_REPO,
        ArchivePrefixLimit,
        BudgetedReader,
        PreparationError,
        encode_vlm_row,
        mammoth_index,
    ) = (
        helpers[key]
        for key in (
            "MIB",
            "VLM_ANNOTATIONS",
            "VLM_REPO",
            "ArchivePrefixLimit",
            "BudgetedReader",
            "PreparationError",
            "encode_vlm_row",
            "mammoth_index",
        )
    )

    state = state if state is not None else {}
    selected = selected if selected is not None else set()
    durable = checkpoint is not None
    if durable and args.batch_size != 1:
        raise ValueError(
            "Durable sampling requires batch_size=1 for atomic consumer checkpoints"
        )
    checkpoint = checkpoint or (lambda: None)
    index = mammoth_index(hub, revision)
    if not state:
        archives = sorted(
            [
                dict(path=f.path, size=f.size)
                for f in hub.api.list_repo_tree(
                    VLM_REPO,
                    "single_image_data",
                    repo_type="dataset",
                    revision=revision,
                )
                if f.path.endswith(".tar.gz")
            ],
            key=lambda f: f["path"],
        )
        random.Random(args.seed).shuffle(archives)
        state.update(
            phase="text",
            image_budget=getattr(
                hub, "image_selection_budget", hub.limit - hub.received
            ),
            archives=archives,
            text_position=0,
            text_accepted=0,
            attempts=0,
            visited=visited,
        )
        checkpoint()
    if visited is not state["visited"]:
        visited[:] = state["visited"]
        state["visited"] = visited
    batch = []

    def encode(number, payload, content):
        state["attempts"] += 1
        if state["attempts"] > max_rows:
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
        if state["phase"] == "text":
            if text_count:
                # uint64 array has the same elementwise shuffle as the original
                # Python-int list, with bounded metadata overhead at 10M rows.
                from array import array

                ids = array(
                    "Q",
                    (
                        r[0]
                        for r in db.execute(
                            "SELECT rowid FROM records WHERE image='' ORDER BY rowid"
                        )
                    ),
                )
                random.Random(args.seed).shuffle(ids)
                while (
                    state["text_position"] < len(ids)
                    and state["text_accepted"] < text_count
                ):
                    rowid = ids[state["text_position"]]
                    number, payload = db.execute(
                        "SELECT row_number,payload FROM records WHERE rowid=?", (rowid,)
                    ).fetchone()
                    pair = encode(number, payload, None)
                    state["text_position"] += 1
                    state["text_accepted"] += pair[1] is not None
                    batch.append(pair)
                    if len(batch) >= args.batch_size:
                        yield batch
                        batch = []
                if state["text_accepted"] != text_count:
                    raise PreparationError("Not enough valid text-only candidates.")
            image_count = args.samples - text_count
            state["archives"] = state["archives"][
                : min(len(state["archives"]), image_count)
            ]
            if image_count and not state["archives"]:
                raise PreparationError("No image archives found.")
            state.update(
                phase="images",
                remaining=image_count,
                position=0,
                sampling_round=1,
                round_start=image_count,
                productive=[],
                active=False,
                prefix_limit=state["image_budget"] // max(1, len(state["archives"])),
            )
            # At a boundary the consumer must have accepted every emitted row.
            if batch and durable:
                yield batch
                batch = []
            checkpoint()

        while state["phase"] == "images" and state["remaining"] and state["archives"]:
            if state["position"] == len(state["archives"]):
                if state["remaining"] == state["round_start"]:
                    break
                state.update(
                    archives=state["productive"],
                    productive=[],
                    position=0,
                    round_start=state["remaining"],
                    sampling_round=state["sampling_round"] + 1,
                )
                checkpoint()
                continue
            archive = state["archives"][state["position"]]
            if not state["active"]:
                left = len(state["archives"]) - state["position"]
                visited.append(
                    dict(
                        file=archive["path"],
                        sampling_round=state["sampling_round"],
                        rows_read=0,
                        target_rows=(state["remaining"] + left - 1) // left,
                        file_bytes_read=0,
                        prefix_byte_limit=state["prefix_limit"],
                    )
                )
                state["active"] = True
            entry = visited[-1]
            target = entry["target_rows"]
            before, previous_bytes = hub.received, entry["file_bytes_read"]
            if entry["rows_read"] < target and not entry.get("prefix_limit_reached"):
                path = f"datasets/{VLM_REPO}@{revision}/{archive['path']}"
                print(
                    f"Reading up to {target:,} valid conversations from hf://{path}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    with (
                        hub.fs.open(
                            path, "rb", size=archive["size"], cache_type="none"
                        ) as remote,
                        BudgetedReader(
                            hub, remote, max_bytes=state["prefix_limit"]
                        ) as reader,
                        tarfile.open(
                            fileobj=reader, mode="r|gz", bufsize=8 * MIB
                        ) as tar,
                    ):
                        for member in tar:
                            if not member.isfile() or member.size > 64 * MIB:
                                continue
                            matches = db.execute(
                                "SELECT row_number,payload FROM records WHERE image=? ORDER BY row_number",
                                (member.name.removeprefix("./"),),
                            )
                            # fetchone avoids loading every row for popular images.
                            first = matches.fetchone()
                            if first is None:
                                continue
                            with closing(tar.extractfile(member)) as image:
                                content = image.read()
                            from itertools import chain

                            for number, payload in chain([first], matches):
                                if number in selected:
                                    continue
                                selected.add(number)
                                pair = encode(number, payload, content)
                                if pair[1] is not None:
                                    entry["rows_read"] += 1
                                    state["remaining"] -= 1
                                entry["file_bytes_read"] = (
                                    previous_bytes + hub.received - before
                                )
                                batch.append(pair)
                                if len(batch) >= args.batch_size:
                                    yield batch
                                    batch = []
                                if entry["rows_read"] == target:
                                    break
                            if entry["rows_read"] == target:
                                break
                except ArchivePrefixLimit:
                    entry["prefix_limit_reached"] = True
                finally:
                    entry["file_bytes_read"] = previous_bytes + hub.received - before
            if entry["rows_read"] == target:
                state["productive"].append(archive)
            state.update(position=state["position"] + 1, active=False)
            if batch and durable:
                yield batch
                batch = []
            checkpoint()
        if state.get("remaining", 0):
            raise PreparationError(
                "Not enough valid conversations within the bounded archive prefixes; "
                "start a new preparation with a larger initial selection budget or fewer samples. "
                "Raising a retry's traffic cap does not widen frozen archive prefixes."
            )
        state["phase"] = "complete"
        if batch:
            yield batch
        checkpoint()
