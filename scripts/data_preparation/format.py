"""The existing LitData PyTree schema with raw RGB images and zstd chunks."""

from __future__ import annotations

import io
import struct

from prepare_data import LitWriter, encode_leaves

MAX_RECORD_BYTES = 128 * 1024**2
MAX_RAW_BYTES = 1024**3


def decoded_size(item, *, validate=False):
    """Exact output length from image headers; no RGB payload materialization."""
    from PIL import Image

    parts = leaves(item)
    blob = parts[0]
    count = struct.unpack_from(">I", blob)[0]
    offset, pixels = 4, 0
    for _ in range(count):
        size = struct.unpack_from(">I", blob, offset)[0]
        offset += 4
        with Image.open(io.BytesIO(blob[offset : offset + size])) as image:
            pixels += image.width * image.height * 3
            if validate:
                image.load()
                converted = image.convert("RGB")
                converted.close()
        offset += size
    if offset != len(blob) or pixels > MAX_RAW_BYTES:
        raise ValueError(
            "Selected image exceeds the RGB preparation bound; no replacement was selected"
        )
    return 20 + sum(map(len, parts[1:])) + 8 + 12 * count + pixels


def leaves(item: bytes) -> list[bytes]:
    if len(item) < 20:
        raise ValueError("Truncated VLM record")
    lengths = struct.unpack_from("<5I", item)
    if sum(lengths) + 20 != len(item):
        raise ValueError("Invalid VLM record lengths")
    result, start = [], 20
    for size in lengths:
        result.append(item[start : start + size])
        start += size
    return result


def decoded_record(item: bytes) -> bytes:
    """Preserve dimensions, text and metadata; convert each image to RGB."""
    from PIL import Image

    parts = leaves(item)
    blob = parts[0]
    count = struct.unpack_from(">I", blob)[0]
    header, pixels, offset, total = [count], [], 4, 0
    for _ in range(count):
        size = struct.unpack_from(">I", blob, offset)[0]
        offset += 4
        with Image.open(io.BytesIO(blob[offset : offset + size])) as image:
            total += image.width * image.height * 3
            if total > MAX_RAW_BYTES:
                raise ValueError(
                    "Decoded record exceeds the explicit 1 GiB pixel limit"
                )
            raw = image.convert("RGB").tobytes()
            header.extend((image.width, image.height, len(raw)))
            pixels.append(raw)
        offset += size
    if offset != len(blob) or struct.unpack("<q", parts[-1])[0] != count:
        raise ValueError("Image count or encoded blob length mismatch")
    parts[0] = struct.pack(f">{len(header) + 1}I", 4 * len(header), *header) + b"".join(
        pixels
    )
    return encode_leaves(parts)


def config(chunk_bytes, compression="zstd", mode="vlm"):
    # Same PyTree structure and leaf serializers as throughput/TTFB.
    result = LitWriter(None, mode, chunk_bytes).config
    result["compression"] = "zstd" if compression == "zstd" else None
    return result


def chunk(items, compression="zstd"):
    offsets = [4 * (len(items) + 2)]
    for item in items:
        offsets.append(offsets[-1] + len(item))
    if offsets[-1] >= 2**32:
        raise ValueError("LitData uint32 chunk offset limit exceeded")
    raw = struct.pack(f"<{len(offsets) + 1}I", len(items), *offsets) + b"".join(items)
    if compression == "none":
        return raw, len(raw)
    import zstd

    return zstd.compress(raw, 4, 1), len(raw)


def verify_chunk(
    data,
    expected_count,
    expected_bytes=None,
    *,
    compression="zstd",
    images="decoded-rgb",
    mode="vlm",
):
    """Check all record framing and RGB headers in a sampled compressed shard."""
    import zstd

    raw = zstd.decompress(data) if compression == "zstd" else data
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError("Uncompressed shard size differs from the LitData index")
    count = struct.unpack_from("<I", raw)[0]
    if count != expected_count:
        raise ValueError("Shard record count differs from index")
    offsets = struct.unpack_from(f"<{count + 1}I", raw, 4)
    if offsets[0] != 4 * (count + 2) or offsets[-1] != len(raw):
        raise ValueError("Invalid shard offsets")
    for a, b in zip(offsets, offsets[1:]):
        if a >= b:
            raise ValueError("Invalid record offsets")
        if mode == "text":
            item = raw[a:b]
            if (
                len(item) < 4
                or struct.unpack_from("<I", item)[0] != len(item) - 4
                or (len(item) - 4) % 4
            ):
                raise ValueError("Invalid uint32 token record")
            continue
        parts = leaves(raw[a:b])
        if images == "encoded":
            from PIL import Image

            blob = parts[0]
            n = struct.unpack_from(">I", blob)[0]
            offset = 4
            for _ in range(n):
                size = struct.unpack_from(">I", blob, offset)[0]
                offset += 4
                with Image.open(io.BytesIO(blob[offset : offset + size])) as image:
                    image.verify()
                offset += size
            if offset != len(blob) or struct.unpack("<q", parts[-1])[0] != n:
                raise ValueError("Invalid encoded image blob")
            continue
        blob = parts[0]
        header_bytes = struct.unpack_from(">I", blob)[0]
        if header_bytes < 4 or header_bytes % 4:
            raise ValueError("Expected decoded RGB images, not encoded JPEG/PNG")
        header = struct.unpack_from(f">{header_bytes // 4}I", blob, 4)
        n = header[0]
        if len(header) != 1 + 3 * n or struct.unpack("<q", parts[-1])[0] != n:
            raise ValueError("Invalid RGB image header")
        size = 4 + header_bytes
        for pos in range(1, len(header), 3):
            w, h, length = header[pos : pos + 3]
            if not w or not h or length != w * h * 3:
                raise ValueError("Image payload is not raw RGB")
            size += length
        if size != len(blob):
            raise ValueError("Truncated RGB pixels")
