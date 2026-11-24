"""Raw sample decoding and parsing for VLM data pipeline.

These functions handle the first stage of data processing:
- Deserializing raw PIL bytes from LitData format
- Parsing string text fields via ast.literal_eval
- Normalizing conversation formats

Used by the Grain and Zephon preprocessing pipelines.
"""

import ast
import logging
import struct
from collections.abc import Buffer
from typing import Any, Dict, List

import PIL.Image

from .conversation import normalize_conversation_format

logger = logging.getLogger(__name__)

UINT32_SIZE = 4


def deserialize_encoded_image_array(blob: Buffer) -> List[PIL.Image.Image]:
    """Deserialize an encoded-bytes blob (JPEG/PNG) back into PIL RGB images.

    - 4 bytes: n_images (uint32, big-endian)
    - For each image:
      - 4 bytes: content_length (uint32, big-endian)
      - content_length bytes: original encoded image bytes (JPEG/PNG)

    Raises:
        ValueError: If blob is malformed or cannot be deserialized.
    """
    from io import BytesIO

    blob_view = memoryview(blob).cast("B")
    blob_size = len(blob_view)

    if blob_size < UINT32_SIZE:
        raise ValueError(f"Encoded image blob too short: {blob_size} bytes")

    (n_images,) = struct.unpack_from(">I", blob_view, 0)
    offset = UINT32_SIZE
    images: List[PIL.Image.Image] = []

    for index in range(n_images):
        if offset + UINT32_SIZE > blob_size:
            raise ValueError(
                f"Encoded image blob truncated: no length header for image {index}"
            )
        (content_length,) = struct.unpack_from(">I", blob_view, offset)
        offset += UINT32_SIZE

        end = offset + content_length
        if end > blob_size:
            raise ValueError(
                f"Encoded image blob truncated for image {index}: "
                f"need {end}, have {blob_size}"
            )

        # Slice memoryview directly into BytesIO — avoids a full copy.
        # PIL reads from the buffer; .load() forces full decode so the
        # BytesIO can be GC'd immediately after.
        pil_image = PIL.Image.open(BytesIO(blob_view[offset:end]))
        pil_image.load()
        offset = end

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        images.append(pil_image)

    return images


def deserialize_pil_array(blob: Buffer) -> List[PIL.Image.Image]:
    """Deserialize a raw-bytes blob back into PIL RGB images using Image.frombytes().

    Format:
    - 4 bytes: header_size (uint32, big-endian)
    - header_size bytes: header containing [n_images, w1, h1, len1, w2, h2, len2, ...]
    - remaining bytes: concatenated raw RGB pixel data



    Raises:
        ValueError: If blob is malformed or cannot be deserialized.
    """
    blob_view = memoryview(blob).cast("B")
    blob_size = len(blob_view)

    if len(blob_view) < UINT32_SIZE:
        logger.warning(
            f"Malformed PIL blob: too short ({blob_size} bytes, need at least {UINT32_SIZE})"
        )
        raise ValueError(f"PIL blob too short: {blob_size} bytes")

    header_size = struct.unpack(">I", blob_view[:UINT32_SIZE])[0]
    if header_size % UINT32_SIZE != 0:
        logger.warning(
            f"Malformed PIL blob: header_size ({header_size}) not aligned to {UINT32_SIZE} bytes"
        )
        raise ValueError(f"PIL blob header misaligned: {header_size}")

    if blob_size < UINT32_SIZE + header_size:
        logger.warning(
            f"Malformed PIL blob: insufficient data for header (have {blob_size}, need {UINT32_SIZE + header_size})"
        )
        raise ValueError(
            f"PIL blob truncated: have {blob_size} bytes, need {UINT32_SIZE + header_size}"
        )

    header_int_count = header_size // UINT32_SIZE
    header = struct.unpack(
        f">{header_int_count}I",
        blob_view[UINT32_SIZE : UINT32_SIZE + header_size],
    )

    n_images = header[0]
    images: List[PIL.Image.Image] = []
    offset = UINT32_SIZE + header_size

    for index in range(n_images):
        start = 1 + index * 3
        if start + 3 > len(header):
            logger.warning(
                f"Malformed PIL blob header: not enough entries for image {index} (need {start + 3}, have {len(header)})"
            )
            raise ValueError(f"PIL blob header incomplete for image {index}")
        width, height, length = header[start : start + 3]
        if offset + length > blob_size:
            logger.warning(
                f"Malformed PIL blob: not enough pixel data for image {index} (need {offset + length}, have {blob_size})"
            )
            raise ValueError(f"PIL blob pixel data truncated for image {index}")
        raw = blob_view[offset : offset + length]
        images.append(PIL.Image.frombytes("RGB", (width, height), raw))
        offset += length

    logger.debug(f"Deserialized {len(images)} images from PIL blob ({blob_size} bytes)")
    return images


def safe_parse_sample(
    sample: Dict[str, Any],
    decoded_image_input: bool = True,
) -> Dict[str, Any]:
    """Normalize raw sample schema and parse fields.

    Supports two image serialization formats controlled by decoded_image_input:
    - True (default): Raw PIL pixel bytes (header + RGB data, ~10x faster load)
    - False: Encoded image bytes (JPEG/PNG, ~3x smaller storage, decoded on the fly)

    Also normalizes conversation format from source from/value to role/content.

    Args:
        sample: Raw sample dict from LitData with text and images fields
        decoded_image_input: If True, images are pre-decoded raw pixel bytes.
            If False, images are encoded JPEG/PNG bytes decoded on the fly.

    Returns:
        Modified sample dict with parsed text (list) and images (list of PIL.Image)

    Raises:
        ValueError: On unexpected formats to fail fast on data issues.
    """
    # Work on a shallow copy to avoid mutating the caller's dict
    sample = dict(sample)

    # Parse text field
    try:
        if "text" in sample and isinstance(sample["text"], str):
            sample["text"] = ast.literal_eval(sample["text"])
        else:
            sample.setdefault("text", [])
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Could not parse text field: {e}. Defaulting to empty list.")
        sample["text"] = []

    # Normalize conversation format (from/value -> role/content)
    if isinstance(sample["text"], list) and len(sample["text"]) > 0:
        normalized = normalize_conversation_format(sample["text"])
        if not normalized or any(
            "role" not in m or "content" not in m for m in normalized
        ):
            logger.error(
                "Sample has invalid conversation format (neither role/content nor from/value), skipping"
            )
            sample["text"] = []
        else:
            sample["text"] = normalized

    # Handle images
    _deserialize = (
        deserialize_pil_array
        if decoded_image_input
        else deserialize_encoded_image_array
    )
    image_list: List[PIL.Image.Image] = []
    images_val = sample["images"]

    if isinstance(images_val, Buffer):
        image_list = _deserialize(images_val)

    elif images_val is None:
        # Explicitly null - no images
        logger.debug("Sample has null images field")

    else:
        # Unexpected format - fail fast
        logger.error(
            f"Unexpected images field type: {type(images_val).__name__}. Expected bytes-like blob."
        )
        raise ValueError(
            f"Unexpected images format: {type(images_val).__name__}. Expected raw PIL bytes-like blob."
        )

    if not image_list:
        logger.debug("Sample has no images")

    sample["image"] = image_list
    sample["images"] = image_list
    sample.setdefault("metadata", {})

    return sample
