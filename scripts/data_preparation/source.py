"""Run the shared sampler with durable selection and bounded S3 spooling."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import closing
from pathlib import Path

from prepare_data import PublicHub

from .mammoth import batches


class SelectedRows:
    def __init__(self, db):
        self.db = db

    def __contains__(self, number):
        return (
            self.db.execute(
                "SELECT 1 FROM selected WHERE number=?", (number,)
            ).fetchone()
            is not None
        )

    def add(self, number):
        self.db.execute("INSERT OR IGNORE INTO selected VALUES (?)", (number,))


class BudgetedHub(PublicHub):
    def __init__(self, job):
        self.job = job
        self.budget_path = job.journal / "source-budget.json"
        self.in_annotation_download = False
        self.charged = (
            json.loads(self.budget_path.read_text()) if self.budget_path.exists() else 0
        )
        super().__init__(
            job.settings["max_download_bytes"] - self.charged,
            Path(job.settings.get("cache_dir", job.scratch.path / "hub")),
        )

    @property
    def image_selection_budget(self):
        original = self.job.get("selection_source_cap")
        if original is None:
            original = self.job.settings["max_download_bytes"]
            self.job.set("selection_source_cap", original)
        return original - self.job.get("selection_annotation_bytes", 0)

    def download(self, repo, revision, filename, kind):
        from prepare_data import VLM_ANNOTATIONS, VLM_REPO

        annotation = repo == VLM_REPO and filename == VLM_ANNOTATIONS
        self.in_annotation_download = annotation
        try:
            result = super().download(repo, revision, filename, kind)
            if annotation and self.job.get("selection_annotation_bytes") is None:
                self.job.set("selection_annotation_bytes", 0)
            return result
        finally:
            self.in_annotation_download = False

    def check_budget(self, size):
        super().check_budget(size)
        if (
            self.in_annotation_download
            and self.job.get("selection_annotation_bytes") is None
        ):
            self.job.set("selection_annotation_bytes", size)
            self.job.set(
                "selection_source_cap", self.job.settings["max_download_bytes"]
            )
        if self.charged + size > self.job.settings["max_download_bytes"]:
            raise ValueError("Cumulative source-download cap reached")
        # Reserve before IO, independently of the selection transaction. A killed
        # request may be conservatively overcounted; never undercount retries.
        self.charged += size
        tmp = self.budget_path.with_suffix(".partial")
        with tmp.open("w") as handle:
            import os

            json.dump(self.charged, handle)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.budget_path)


def ingest(job):
    if job.settings["mode"] == "text":
        return ingest_text(job)
    settings = job.settings
    if job.sampler.get("phase") == "complete":
        return
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_CACHE"] = str(job.scratch.path / "hub" / "xet")
    cache = Path(settings.get("cache_dir", job.scratch.path / "hub"))
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / f"mammoth-si-index-v1-{settings['revision']}.sqlite").exists():
        job.scratch.check(
            32 * 1024**3,
            destination=cache,
        )
    with closing(BudgetedHub(job)) as hub:
        from prepare_data import VLM_REPO

        if hub.repo(VLM_REPO, settings["revision"]).sha != settings["revision"]:
            raise ValueError("Public source revision changed")
        args = argparse.Namespace(
            samples=settings["samples"], seed=settings["seed"], batch_size=1
        )
        visited = job.sampler.get("visited", [])
        stream = batches(
            hub,
            settings["revision"],
            args,
            settings["max_source_rows"],
            visited,
            state=job.sampler,
            selected=SelectedRows(job.db),
            checkpoint=job.flush,
        )
        with closing(stream):
            for batch in stream:
                for source_id, item in batch:
                    if item is None:
                        continue
                    number = int(source_id.rsplit("#row=", 1)[1])
                    import struct

                    from .format import leaves

                    kind = (
                        "image" if struct.unpack("<q", leaves(item)[-1])[0] else "text"
                    )
                    job.accept(number, item, kind)
    job.flush()


def ingest_text(job):
    """Use the existing FineWeb tokenizer/packing logic, replaying on resume."""
    import hashlib
    import random
    from pathlib import Path

    import numpy as np
    import prepare_data as prep

    spec = job.settings
    if job.sampler.get("phase") == "complete":
        return
    visited = []
    job.sampler["visited"] = visited
    with closing(BudgetedHub(job)) as hub:
        revision = hub.repo(prep.TEXT_REPO, spec["revision"]).sha
        if revision != spec["revision"]:
            raise ValueError("Public text source revision changed")
        pinned = job.get("resolved_tokenizer")
        model = hub.repo(
            spec["tokenizer"],
            pinned["revision"] if pinned else spec["tokenizer_revision"],
            "model",
        )
        tokenizer = hub.download(
            spec["tokenizer"], model.sha, "tokenizer.json", "model"
        )
        tokenizer_info = dict(
            repo=spec["tokenizer"],
            revision=model.sha,
            sha256=hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest(),
        )
        if pinned is not None and tokenizer_info != pinned:
            raise ValueError("Pinned tokenizer content changed")
        if pinned is None:
            job.set("resolved_tokenizer", tokenizer_info)
        job.sampler["tokenizer"] = tokenizer_info
        files = hub.parquet_files(prep.TEXT_REPO, revision, "sample/10BT")
        random.Random(spec["seed"]).shuffle(files)
        source = prep.source_batches(
            hub,
            prep.TEXT_REPO,
            revision,
            files,
            ["text"],
            spec["batch_size"],
            spec["max_source_rows"],
            visited,
        )
        processed = prep.parallel_batches(
            prep.tokenize_batch, source, spec["workers"], tokenizer
        )
        buffer, count = [], 0
        with closing(source), closing(processed):
            for batch in processed:
                for _, values in batch:
                    buffer.extend(values)
                    consumed = 0
                    while (
                        len(buffer) - consumed >= spec["sequence_length"]
                        and count < spec["samples"]
                    ):
                        tokens = np.asarray(
                            buffer[consumed : consumed + spec["sequence_length"]],
                            dtype="<u4",
                        )
                        job.accept(
                            count, prep.encode_leaves([tokens.tobytes()]), "text"
                        )
                        consumed += spec["sequence_length"]
                        count += 1
                    buffer = buffer[consumed:]
                    if count == spec["samples"]:
                        break
                if count == spec["samples"]:
                    break
        if count != spec["samples"]:
            raise ValueError("Not enough text tokens within the source budget")
    job.sampler["phase"] = "complete"
    job.flush()
