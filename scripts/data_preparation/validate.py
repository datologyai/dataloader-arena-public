"""Inspect scale, schema and sampled payloads before a checkpoint benchmark."""

from __future__ import annotations

import json

from .format import config, verify_chunk
from .s3 import digest


def check(store, settings, *, existing=False, all_objects=False):
    raw = store.get("index.json")
    index = json.loads(raw)
    chunks = index["chunks"]
    count = len(chunks)
    records = sum(c["chunk_size"] for c in chunks)
    if not settings["minimum_records"] <= records <= settings["maximum_records"]:
        raise ValueError(f"Wrong dataset scale: {records:,} records")
    if not settings["minimum_shards"] <= count <= settings["maximum_shards"]:
        raise ValueError(f"Wrong dataset layout: {count:,} physical shards")
    if any(type(c["chunk_size"]) is not int or c["chunk_size"] <= 0 for c in chunks):
        raise ValueError("Invalid shard counts")
    names = [c["filename"] for c in chunks]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate shard names")
    cfg = index["config"]
    compression = settings.get("compression", "zstd")
    images = settings.get("image_representation", "decoded-rgb")
    mode = settings.get("mode", "vlm")
    if cfg.get("compression") not in (
        ("zstd", "zstd:4") if compression == "zstd" else (None,)
    ):
        raise ValueError("Dataset compression differs from recipe")
    expected = config(settings["chunk_bytes"], compression, mode)
    if (
        cfg.get("data_format") != expected["data_format"]
        or cfg.get("item_loader") != "PyTreeLoader"
        or json.loads(cfg.get("data_spec", "null")) != json.loads(expected["data_spec"])
        or cfg.get("encryption") is not None
    ):
        raise ValueError("Unexpected LitData schema")
    manifest = None
    if not existing:
        manifest = json.loads(store.get("manifest.json"))
        if (
            manifest.get("purpose") != "data-preparation"
            or (
                manifest.get("destination") != store.uri
                and ("://" in str(manifest.get("destination")) or "://" in store.uri)
            )
            or manifest.get("index_sha256") != digest(raw)
            or manifest.get("records") != records
            or manifest.get("physical_shards") != count
            or manifest.get("image_representation") != images
            or manifest.get("compression")
            != ("zstd:4" if compression == "zstd" else None)
        ):
            raise ValueError("Dataset manifest/format/index identity mismatch")
        for key in ("revision", "samples", "seed", "chunk_bytes"):
            if manifest["settings"].get(key) != settings[key]:
                raise ValueError(f"Dataset recipe differs: {key}")
        receipt = manifest["inventory"]
        inventory = store.get(receipt["name"])
        if digest(inventory) != receipt["sha256"]:
            raise ValueError("Changed upload inventory")
        # Inventory certifies every referenced shard was uploaded with a server
        # checksum. --all-objects additionally HEADs their current identities.
        entries = [json.loads(line) for line in inventory.splitlines()]
        if len(entries) != count or [e["name"] for e in entries] != names:
            raise ValueError("Upload inventory does not cover the LitData index")
        if [e["records"] for e in entries] != [c["chunk_size"] for c in chunks]:
            raise ValueError("Inventory record counts differ from the LitData index")
        if sum(e["bytes"] for e in entries) != manifest["stored_bytes"]:
            raise ValueError("Stored-size manifest mismatch")
        if all_objects:
            from concurrent.futures import ThreadPoolExecutor

            # Bounded batches, rather than one future per half-million shards.
            with ThreadPoolExecutor(max_workers=16) as pool:
                for start in range(0, len(entries), 256):
                    list(pool.map(store.verify, entries[start : start + 256]))
    elif all_objects:
        raise ValueError(
            "Full checksum identity verification needs a generated upload inventory"
        )
    samples = []
    for position in sorted({0, count // 2, count - 1}):
        shard = chunks[position]
        data = store.get(shard["filename"])
        if manifest is not None and digest(data) != entries[position]["sha256"]:
            raise ValueError("Sampled shard checksum changed")
        verify_chunk(
            data,
            shard["chunk_size"],
            shard["chunk_bytes"],
            compression=compression,
            images=images,
            mode=mode,
        )
        samples.append(
            {"name": shard["filename"], "sha256": digest(data), "bytes": len(data)}
        )
    if digest(store.get("index.json")) != digest(raw):
        raise ValueError("Dataset index changed during validation")
    return {
        "valid": True,
        "destination": store.uri,
        "records": records,
        "physical_shards": count,
        "index_sha256": digest(raw),
        "sampled_shards": samples,
        "all_object_identities_checked": all_objects,
        "manifest_checked": not existing,
        "content_validation": "first/middle/last shards decoded and framing checked; no full-corpus hash",
        "validation_bounds": {
            key: settings[key]
            for key in (
                "minimum_records",
                "maximum_records",
                "minimum_shards",
                "maximum_shards",
            )
        },
    }
