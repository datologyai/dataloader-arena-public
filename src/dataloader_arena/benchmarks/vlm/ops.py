"""Zephon operators for the shared VLM preprocessing and packing pipeline."""

import logging
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

import PIL.Image
import torch

from dataloader_arena.benchmarks.vlm.transforms import (
    align_image_placeholders,
    batch_transform_images,
    build_vision_token_sequence,
    classify_drop_reason,
    filter_placeholder_images,
    inject_vision_tokens,
    safe_parse_sample,
    tokenize_conversations,
    validate_batch,
)
from dataloader_arena.benchmarks.vlm.transforms.batch_construction import (
    construct_batch_tensors,
)
from dataloader_arena.benchmarks.vlm.transforms.diagnostics import (
    DIAG_SUMMARY_INTERVAL,
    IMAGE_DIAG_ENABLED,
    diag_log,
    make_diag_counters,
)
from dataloader_arena.benchmarks.vlm.transforms.models import Conversation, Message
from dataloader_arena.benchmarks.vlm.transforms.packing import Bin, shuffle_and_pack

logger = logging.getLogger(__name__)


def make_decode_fn(
    decoded_image_input: bool = True,
) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Create decode function for raw LitData payloads.

    Uses the shared safe_parse_sample decoder.

    Args:
        decoded_image_input: If True, images are pre-decoded raw pixel bytes.
            If False, images are encoded JPEG/PNG bytes decoded on the fly.

    Returns a function that extracts conversation and images from raw payload.
    """

    _diag: Dict[str, int] = make_diag_counters() if IMAGE_DIAG_ENABLED else {}

    def decode(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decode raw payload into conversation and images.

        Calls safe_parse_sample() to handle:
        - String text fields (ast.literal_eval to list)
        - Conversation format normalization (from/value -> role/content)
        - Raw PIL bytes deserialization
        """
        # Decode the raw sample.
        try:
            payload = safe_parse_sample(
                payload, decoded_image_input=decoded_image_input
            )
        except ValueError as e:
            logger.warning(f"Failed to parse sample: {e}, dropping")
            if _diag:
                _diag["dropped_parse"] += 1
            return None

        # Extract normalized fields
        text = payload.get("text")
        # safe_parse_sample puts images in "image" key (list of PIL images)
        images = payload.get("image", [])

        if not text:
            logger.warning(
                "Payload has empty 'text' field after parsing, dropping sample"
            )
            return None

        if not isinstance(text, list):
            logger.warning(
                f"Unexpected text format after parsing: {type(text)}, dropping sample"
            )
            return None

        if _diag:
            _diag["total"] += 1
            n_raw = len(images) if images else 0
            _diag["raw_images"] += n_raw
            if n_raw > 0:
                _diag["with_images"] += 1
            else:
                _diag["no_images"] += 1
            if _diag["total"] % DIAG_SUMMARY_INTERVAL == 0:
                diag_log("DECODE", "zephon", **_diag)

        return {
            "conversation": text,
            "images": images,
        }

    return decode


def make_preprocess_fn(
    tokens_per_image: int,
    wrap_user_message: Callable[[str], str],
    wrap_assistant_message: Callable[[str], str],
    start_sequence: str,
    enable_dynamic_tiling: bool = False,
    tiling_preprocessor: Optional[Any] = None,
    tokens_per_tile: int = 243,
    image_transform: Optional[Callable[[PIL.Image.Image], torch.Tensor]] = None,
    debug: bool = False,
) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Create preprocessing function for conversations.

    Uses shared transforms from dataloader_arena.benchmarks.vlm.transforms.

    Args:
        tokens_per_image: Number of image pad tokens per image (fixed tiling)
        wrap_user_message: Function to wrap user message content
        wrap_assistant_message: Function to wrap assistant message content
        start_sequence: Start sequence prompt to prepend to first message
        enable_dynamic_tiling: Whether to use dynamic tiling
        tiling_preprocessor: Tiling preprocessor for dynamic tiling
        tokens_per_tile: Tokens per tile for dynamic tiling
        debug: Enable debug logging

    Returns:
        Preprocessing function that takes a decoded payload and returns
        a preprocessed payload with Conversation object.
    """

    _diag: Dict[str, int] = make_diag_counters() if IMAGE_DIAG_ENABLED else {}

    def preprocess(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess conversation using shared transforms."""
        conversation_list = payload["conversation"]
        images = payload["images"]

        if _diag:
            _diag["total"] += 1
            n_raw = len(images) if images else 0
            _diag["raw_images"] += n_raw
            if n_raw > 0:
                _diag["with_images"] += 1
            else:
                _diag["no_images"] += 1

        # Convert to Conversation object
        try:
            messages = [
                Message(role=msg["role"], content=msg["content"])
                for msg in conversation_list
            ]
            conversation = Conversation(messages=messages, images=images)
        except Exception as e:
            logger.warning(f"Error converting conversation: {e}, dropping sample")
            return None

        # Filter placeholder 1x1 images (LitData workaround)
        conversation.images = filter_placeholder_images(conversation.images)

        if _diag:
            _diag["after_filter"] += (
                len(conversation.images) if conversation.images else 0
            )

        # Align image placeholders before injecting vision tokens.
        debug_context = str(conversation_list) if debug else None
        conversation = align_image_placeholders(
            conversation, debug=debug, debug_context=debug_context
        )
        if conversation is None:
            if _diag:
                _diag["dropped_align"] += 1
            return None

        # Build vision token sequences
        if enable_dynamic_tiling and tiling_preprocessor and conversation.images:
            # Dynamic tiling: split PIL images into tiles, variable tokens per image
            tiled_images = []
            tile_counts = []

            for img in conversation.images:
                tiles, _, _ = tiling_preprocessor.split_image(img)
                tiled_images.extend(tiles)
                tile_counts.append(len(tiles))

            conversation.images = tiled_images
            conversation.tile_counts = tile_counts

            vision_token_sequences = [
                build_vision_token_sequence(num_tiles * tokens_per_tile)
                for num_tiles in tile_counts
            ]
        else:
            # Fixed-length vision tokens
            num_images = len(conversation.images) if conversation.images else 0
            vision_token_sequences = [
                build_vision_token_sequence(tokens_per_image)
            ] * num_images

        # Inject vision tokens and wrap messages
        inject_vision_tokens(
            conversation,
            vision_token_sequences,
            wrap_user_message,
            wrap_assistant_message,
        )

        # Transform images AFTER tiling and vision token injection.
        # By this point images are validated — transform is strict (raises on failure).
        if (
            image_transform is not None
            and conversation.images is not None
            and len(conversation.images) > 0
        ):
            transformed = batch_transform_images([conversation.images], image_transform)
            conversation.images = transformed[0]

        # Add start sequence to first message
        if conversation.messages:
            conversation.messages[0].content = (
                start_sequence + conversation.messages[0].content
            )

        if _diag and _diag["total"] % DIAG_SUMMARY_INTERVAL == 0:
            diag_log("PREPROCESS", "zephon", **_diag)

        return {
            "conversation": conversation,
            "images": conversation.images if conversation.images is not None else [],
        }

    return preprocess


def make_tokenize_fn(
    tokenizer: Any,
    max_length: Optional[int] = None,
    flatten_payload: bool = False,
) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Create tokenization function.

    Uses shared tokenize_conversations from dataloader_arena.benchmarks.vlm.transforms.
    Drops samples exceeding max_length before packing.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: If set, drop samples with total tokens > max_length

    Returns:
        Tokenization function that adds token_ids to conversation messages.
    """

    def tokenize(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tokenize conversation messages."""
        conversation = payload["conversation"]

        # Use shared tokenization function (mutates in place)
        tokenize_conversations([conversation], tokenizer)

        # Calculate total length for packing
        total_length = sum(len(msg.token_ids or []) for msg in conversation.messages)
        conversation.length = total_length

        if max_length is not None and total_length > max_length:
            logger.debug(
                f"Dropping sample with {total_length} tokens (max_length={max_length})"
            )
            return None

        if flatten_payload:
            all_ids = []
            msg_lens = []
            roles = []
            for msg in conversation.messages:
                ids = msg.token_ids or []
                if ids:
                    all_ids.append(
                        torch.tensor(ids, dtype=torch.int64)
                        if not isinstance(ids, torch.Tensor)
                        else ids
                    )
                    msg_lens.append(len(ids))
                else:
                    msg_lens.append(0)
                roles.append(msg.role)

            return {
                "token_ids": torch.cat(all_ids)
                if all_ids
                else torch.empty(0, dtype=torch.int64),
                "msg_lengths": torch.tensor(msg_lens, dtype=torch.int32),
                "roles": roles,
                "images": payload["images"],
                "length": total_length,
            }

        return {
            "conversation": conversation,
            "images": payload["images"],
            "length": total_length,
        }

    return tokenize


def make_pack_merge_fn(
    fuse_extract: bool = False,
    flatten_payload: bool = False,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Create function to merge payloads from packed samples.

    This is called by the packing operator to combine multiple
    conversation payloads into a single packed payload.

    Args:
        fuse_extract: If True, also run extract_sample_data to produce
            the final batch format (saves one IPC boundary).
        flatten_payload: If True, payloads are flat dicts with tensor fields
            instead of Pydantic Conversation objects.

    Returns:
        Function that merges a list of payloads into one packed payload.
    """
    if fuse_extract and not flatten_payload:
        from dataloader_arena.benchmarks.vlm.transforms.batch_construction import (
            extract_sample_data,
        )
    if fuse_extract and flatten_payload:
        from dataloader_arena.benchmarks.vlm.transforms.batch_construction import (
            extract_sample_data_flat,
        )

    def merge_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple conversation payloads into a packed payload.

        Note: Images are stored in conversation.images (transformed earlier),
        and will be collected by extract_sample_data.
        """
        if fuse_extract and flatten_payload:
            return extract_sample_data_flat(payloads)

        if fuse_extract:
            conversations = [p["conversation"] for p in payloads]
            return extract_sample_data(conversations)

        # Non-fused modes
        total_length = sum(p.get("length", 0) for p in payloads)
        if flatten_payload:
            return {
                "flat_samples": payloads,
                "total_length": total_length,
                "num_sequences": len(payloads),
            }

        conversations = [p["conversation"] for p in payloads]
        return {
            "conversations": conversations,
            "total_length": total_length,
            "num_sequences": len(conversations),
        }

    return merge_payloads


def make_construct_and_validate_batch_fn(
    max_length: int,
    pad_token_id: int,
    allow_cross_document_attention: bool,
    attn_implementation: str,
    tokenizer: Any,
    require_images: bool = True,
    preserve_contributors: bool = False,
) -> Callable:
    """Create a post-batch operator that tensorizes and validates a SampleBatch.

    This runs inside the Zephon pipeline (after the Batch operator) so that
    tensor construction and validation happen off the training thread.

    Args:
        max_length: Maximum sequence length for batch construction
        pad_token_id: Token ID for padding
        allow_cross_document_attention: Whether to allow cross-document attention
        attn_implementation: Attention implementation type (e.g. "flash_attention_2")
        tokenizer: HuggingFace tokenizer (for vision token validation)
        require_images: Whether to drop batches without images
        preserve_contributors: Merge all input metadata for checkpoint progress.
            False keeps only the last input's metadata.

    Returns:
        Transform function that takes a SampleBatch and returns a SampleBatch
        with a single record whose payload is the tensorized batch dict,
        or None to drop invalid batches.
    """
    from zephon.types import SampleBatch, SampleRecord

    # Per-thread drop tracking. Each thread tracks its own drops independently —
    # correct for any parallelism value. Uses dicts (picklable) keyed by thread
    # ID instead of threading.local() which cannot be cloudpickled for MTP mode.
    _drops_by_tid: Dict[int, int] = {}
    _drop_reasons_by_tid: Dict[int, Dict[str, int]] = {}

    def _count_sample_images(samples: List[Dict[str, Any]]) -> List[int]:
        """Count images per sample for diagnostics."""
        counts = []
        for s in samples:
            imgs = s.get("images", [])
            if isinstance(imgs, torch.Tensor):
                counts.append(imgs.shape[0] if imgs.numel() > 0 else 0)
            elif isinstance(imgs, list):
                counts.append(len(imgs))
            else:
                counts.append(0)
        return counts

    def construct_and_validate(batch: SampleBatch) -> Optional[SampleBatch]:
        tid = threading.get_ident()

        samples = [r.payload for r in batch.records]

        if IMAGE_DIAG_ENABLED:
            per_sample_imgs = _count_sample_images(samples)
            pre_images = sum(per_sample_imgs)

        result = construct_batch_tensors(
            samples=samples,
            max_length=max_length,
            pad_token_id=pad_token_id,
            allow_cross_document_attention=allow_cross_document_attention,
            attn_implementation=attn_implementation,
        )

        val_result = validate_batch(
            batch=result,
            tokenizer=tokenizer,
            require_images=require_images,
            per_sample=True,
            collect_diagnostics_on_failure=False,
        )

        if not val_result.is_valid:
            logger.warning(f"Dropping batch: {val_result.reason}")
            _drops_by_tid[tid] = _drops_by_tid.get(tid, 0) + 1
            reason_key = classify_drop_reason(val_result.reason)
            tid_reasons = _drop_reasons_by_tid.setdefault(tid, {})
            tid_reasons[reason_key] = tid_reasons.get(reason_key, 0) + 1
            if IMAGE_DIAG_ENABLED:
                diag_log(
                    "VALIDATE",
                    "zephon",
                    action="DROP",
                    reason=reason_key,
                    batch_size=len(samples),
                    pre_images=pre_images,
                    per_sample=per_sample_imgs,
                )
            return None

        if IMAGE_DIAG_ENABLED:
            diag_log(
                "VALIDATE",
                "zephon",
                action="PASS",
                batch_size=len(samples),
                pre_images=pre_images,
            )

        dropped = _drops_by_tid.pop(tid, 0)
        attempted = dropped + 1
        reasons = _drop_reasons_by_tid.pop(tid, {})

        new_meta = replace(
            batch.records[-1].meta,
            tags={
                **batch.records[-1].meta.tags,
                "_val_batches_dropped": dropped,
                "_val_batches_attempted": attempted,
                "_val_drops_no_images": reasons.get("no_images", 0),
                "_val_drops_vision_mismatch": reasons.get("vision_mismatch", 0),
                "_val_drops_alignment": reasons.get("alignment", 0),
            },
        )
        if preserve_contributors:
            contributors = []
            sample_counts: Dict[int, int] = {}
            token_counts: Dict[int, int] = {}
            has_token_counts = False
            for record in batch.records:
                contributors.extend(record.meta.contribution_refs())
                for component, count in record.meta.component_sample_counts.items():
                    sample_counts[component] = sample_counts.get(component, 0) + count
                if record.meta.component_token_counts is not None:
                    has_token_counts = True
                    for component, count in record.meta.component_token_counts.items():
                        token_counts[component] = token_counts.get(component, 0) + count
            new_meta = replace(
                new_meta,
                component_sample_counts=sample_counts,
                component_token_counts=token_counts if has_token_counts else None,
            ).with_contributors(contributors)
        return SampleBatch(records=(SampleRecord(meta=new_meta, payload=result),))

    return construct_and_validate


def make_fused_transform_fn(
    tokenizer: Any,
    tokens_per_image: int,
    wrap_user_message: Callable[[str], str],
    wrap_assistant_message: Callable[[str], str],
    start_sequence: str,
    enable_dynamic_tiling: bool = False,
    tiling_preprocessor: Optional[Any] = None,
    tokens_per_tile: int = 243,
    image_transform: Optional[Callable[[PIL.Image.Image], torch.Tensor]] = None,
    max_length: Optional[int] = None,
    decoded_image_input: bool = True,
    debug: bool = False,
    flatten_payload: bool = False,
) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Create a FUSED function combining decode + preprocess + tokenize.

    This reduces IPC overhead by combining 3 stages into 1.
    Drops samples exceeding max_length before packing.

    Args:
        tokenizer: HuggingFace tokenizer
        tokens_per_image: Number of image pad tokens per image
        wrap_user_message: Function to wrap user message content
        wrap_assistant_message: Function to wrap assistant message content
        start_sequence: Start sequence prompt
        enable_dynamic_tiling: Whether to use dynamic tiling
        tiling_preprocessor: Tiling preprocessor for dynamic tiling
        tokens_per_tile: Tokens per tile for dynamic tiling
        image_transform: Optional image transformation function
        max_length: If set, drop samples with total tokens > max_length
        decoded_image_input: If True, images are pre-decoded raw pixel bytes.
            If False, images are encoded JPEG/PNG bytes decoded on the fly.
        debug: Enable debug logging

    Returns:
        Fused function that takes raw payload and returns tokenized conversation.
    """

    _diag: Dict[str, int] = make_diag_counters() if IMAGE_DIAG_ENABLED else {}

    def fused_transform(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fused decode + preprocess + tokenize in a single function."""

        # ═══════════════════════════════════════════════════════════════════
        # DECODE: Parse raw payload
        # ═══════════════════════════════════════════════════════════════════
        try:
            payload = safe_parse_sample(
                payload, decoded_image_input=decoded_image_input
            )
        except ValueError as e:
            logger.warning(f"Failed to parse sample: {e}, dropping")
            if _diag:
                _diag["dropped_parse"] += 1
            return None

        text = payload.get("text")
        images = payload.get("image", [])

        if not text:
            return None
        if not isinstance(text, list):
            return None

        # ═══════════════════════════════════════════════════════════════════
        # PREPROCESS: Convert to Conversation, transform images, align
        # ═══════════════════════════════════════════════════════════════════
        if _diag:
            _diag["total"] += 1
            n_raw = len(images) if images else 0
            _diag["raw_images"] += n_raw
            if n_raw > 0:
                _diag["with_images"] += 1
            else:
                _diag["no_images"] += 1

        try:
            messages = [
                Message(role=msg["role"], content=msg["content"]) for msg in text
            ]
            conversation = Conversation(messages=messages, images=images)
        except Exception as e:
            logger.warning(f"Error converting conversation: {e}, dropping")
            return None

        # Filter placeholder 1x1 images
        conversation.images = filter_placeholder_images(conversation.images)

        if _diag:
            _diag["after_filter"] += (
                len(conversation.images) if conversation.images else 0
            )

        # Align image placeholders before injecting vision tokens.
        debug_context = str(text) if debug else None
        conversation = align_image_placeholders(
            conversation, debug=debug, debug_context=debug_context
        )
        if conversation is None:
            if _diag:
                _diag["dropped_align"] += 1
            return None

        # Build vision token sequences
        if enable_dynamic_tiling and tiling_preprocessor and conversation.images:
            # Dynamic tiling: split PIL images into tiles
            tiled_images = []
            tile_counts = []
            for img in conversation.images:
                tiles, _, _ = tiling_preprocessor.split_image(img)
                tiled_images.extend(tiles)
                tile_counts.append(len(tiles))
            conversation.images = tiled_images
            conversation.tile_counts = tile_counts
            vision_token_sequences = [
                build_vision_token_sequence(num_tiles * tokens_per_tile)
                for num_tiles in tile_counts
            ]
        else:
            num_images = len(conversation.images) if conversation.images else 0
            vision_token_sequences = [
                build_vision_token_sequence(tokens_per_image)
            ] * num_images

        # Inject vision tokens and wrap messages
        inject_vision_tokens(
            conversation,
            vision_token_sequences,
            wrap_user_message,
            wrap_assistant_message,
        )

        # Transform images AFTER tiling and vision token injection.
        # By this point images are validated — transform is strict (raises on failure).
        if (
            image_transform is not None
            and conversation.images is not None
            and len(conversation.images) > 0
        ):
            transformed = batch_transform_images([conversation.images], image_transform)
            conversation.images = transformed[0]

        # Add start sequence
        if conversation.messages:
            conversation.messages[0].content = (
                start_sequence + conversation.messages[0].content
            )

        # ═══════════════════════════════════════════════════════════════════
        # TOKENIZE: Tokenize conversation
        # ═══════════════════════════════════════════════════════════════════
        tokenize_conversations([conversation], tokenizer)

        total_length = sum(len(msg.token_ids or []) for msg in conversation.messages)
        conversation.length = total_length

        if max_length is not None and total_length > max_length:
            logger.debug(
                f"Dropping sample with {total_length} tokens (max_length={max_length})"
            )
            if _diag:
                _diag["dropped_maxlen"] += 1
            return None

        if _diag and _diag["total"] % DIAG_SUMMARY_INTERVAL == 0:
            diag_log("FUSED_TRANSFORM", "zephon", **_diag)

        if flatten_payload:
            all_ids = []
            msg_lens = []
            roles = []
            for msg in conversation.messages:
                ids = msg.token_ids or []
                if ids:
                    all_ids.append(
                        torch.tensor(ids, dtype=torch.int64)
                        if not isinstance(ids, torch.Tensor)
                        else ids
                    )
                    msg_lens.append(len(ids))
                else:
                    msg_lens.append(0)
                roles.append(msg.role)

            return {
                "token_ids": torch.cat(all_ids)
                if all_ids
                else torch.empty(0, dtype=torch.int64),
                "msg_lengths": torch.tensor(msg_lens, dtype=torch.int32),
                "roles": roles,
                "images": conversation.images
                if conversation.images is not None
                else [],
                "length": total_length,
            }

        return {
            "conversation": conversation,
            "images": conversation.images if conversation.images is not None else [],
            "length": total_length,
        }

    return fused_transform


def add_shared_shuffle_pack(
    pipeline: Any,  # zephon.api.Pipeline
    *,
    window_size: int,
    max_length: int,
    length_fn: Callable[[Any], int],
    seed: int = 42,
    min_sequence_length: int = 50,
    pack_payloads_fn: Callable[[List[Any]], Any],
    parallelism: int = 1,
) -> Any:
    """Add shared shuffle+pack to pipeline using stateful_transform.

    The work is split between pump thread (lightweight) and workers (heavy):
    - push(): Buffers items, emits raw windows -- runs serial on pump thread
    - transform(): Filter + shuffle + pack + record creation -- runs in parallel

    Args:
        pipeline: Zephon pipeline to extend
        window_size: Samples to accumulate before shuffle+pack
        max_length: Maximum sequence length per packed bin
        length_fn: Function to get length from a SampleRecord
        seed: Base RNG seed (combined with content hash for per-window seed)
        min_sequence_length: "Full" threshold
        pack_payloads_fn: Function to merge payloads from packed samples
        parallelism: Number of parallel workers for shuffle+pack (default 1)

    Returns:
        Pipeline with shared shuffle+pack added
    """
    try:
        from zephon.types import SampleRecord
    except ImportError:
        raise ImportError(
            "Zephon is required for shared shuffle+pack: pip install zephon"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PUSH: Lightweight accumulator on pump thread
    # Only buffers items and emits raw windows. No heavy compute here.
    # ═══════════════════════════════════════════════════════════════════════════

    def init_state() -> Dict[str, Any]:
        return {
            "buffer": [],
            "batch_count": 0,
            "last_push_time": None,
            "init_time": time.monotonic(),
        }

    def push(
        state: Dict[str, Any], items: List[SampleRecord]
    ) -> tuple[Dict[str, Any], List[SampleRecord]]:
        """Buffer items and emit one window when ready.

        Emits at most one window per push() call to ensure each ReadyBatch
        dispatched to workers is exactly one window. Remaining items stay
        buffered for the next push() call.
        """
        now = time.monotonic()
        buffer: List[SampleRecord] = state["buffer"]
        batch_count: int = state["batch_count"]

        last_push = state.get("last_push_time")
        if last_push is None:
            init_time = state.get("init_time", now)
            logger.info(
                f"[shared_shuffle_pack] First push after {now - init_time:.1f}s "
                f"({len(items)} items)"
            )
        else:
            wait_sec = now - last_push
            if wait_sec > 5.0:
                logger.warning(
                    f"[shared_shuffle_pack] Waited {wait_sec:.1f}s for push "
                    f"(buffer={len(buffer)}/{window_size})"
                )

        # Tombstones (payload=None) pass through immediately — they must not
        # be buffered or counted toward window_size.
        tombstones = [r for r in items if r.meta.tombstone]
        real_items = [r for r in items if not r.meta.tombstone]

        buffer.extend(real_items)
        outputs: List[SampleRecord] = list(tombstones)

        # Emit at most one window per push to keep ReadyBatches separate
        if len(buffer) >= window_size:
            window = buffer[:window_size]
            buffer = buffer[window_size:]
            batch_count += 1
            outputs.extend(window)

        new_state = {
            "buffer": buffer,
            "batch_count": batch_count,
            "last_push_time": now,
        }
        return new_state, outputs

    def flush(state: Dict[str, Any]) -> List[SampleRecord]:
        """Emit remaining buffer at end of stream.

        Returns raw items -- transform() handles the shuffle+pack.
        """
        buffer: List[SampleRecord] = state["buffer"]
        if not buffer:
            return []
        logger.info(
            f"[shared_shuffle_pack] Flush: {len(buffer)} remaining samples -> transform"
        )
        return buffer

    # ═══════════════════════════════════════════════════════════════════════════
    # TRANSFORM: Heavy work runs in parallel worker processes
    # Receives a raw window of window_size items from push(), does
    # filter + shuffle + pack + record creation.
    # ═══════════════════════════════════════════════════════════════════════════

    def transform(batch: List[SampleRecord]) -> List[SampleRecord]:
        """Shuffle and pack a window of samples. Runs in parallel workers."""
        # Tombstones pass through unchanged — they carry no payload.
        tombstones = [r for r in batch if r.meta.tombstone]
        batch = [r for r in batch if not r.meta.tombstone]
        if not batch:
            return tombstones

        t0 = time.monotonic()

        # Seed from content, not from timing-dependent window numbering.
        content_hash = _batch_content_hash(batch, seed)
        window_generator = torch.Generator()
        window_generator.manual_seed(content_hash)

        # Earlier stages drop samples over max_length; this only guards it.
        oversized = [r for r in batch if length_fn(r) > max_length]
        if oversized:
            logger.warning(
                f"[shared_shuffle_pack] {len(oversized)} samples exceed "
                f"max_length={max_length} — expected upstream to filter these."
            )

        bins, efficiency = shuffle_and_pack(
            items=batch,
            max_length=max_length,
            length_fn=lambda r: length_fn(r),
            generator=window_generator,
            min_sequence_length=min_sequence_length,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            f"[shared_shuffle_pack] Window ({len(batch)} samples) -> "
            f"{len(bins)} bins, efficiency={efficiency:.2%}, transform={elapsed_ms:.1f}ms"
        )

        outputs = list(tombstones)
        for bin_data in bins:
            if bin_data.items:
                outputs.append(
                    _create_packed_record(
                        bin_data, max_length, pack_payloads_fn, length_fn
                    )
                )
        return outputs

    logger.info(
        f"[shared_shuffle_pack] parallelism={parallelism}, window_size={window_size} "
        f"(push=pump thread, transform=parallel workers)"
    )

    return pipeline.stateful_transform(
        "shared_shuffle_pack",
        init_state=init_state,
        push=push,
        flush=flush,
        transform=transform,
        parallelism=parallelism,
        preserves_cursor_order=False,
    )


def _batch_content_hash(batch: list, base_seed: int) -> int:
    """Seed from the batch's sample IDs and order, so it doesn't depend on timing."""
    h = base_seed
    for record in batch:
        sample_id = record.meta.sample_id
        # sample_id is typically (stream_idx, shard_idx, sample_idx)
        if isinstance(sample_id, tuple):
            for part in sample_id:
                h = h * 31 + hash(part)
        else:
            h = h * 31 + hash(sample_id)
    return h & 0x7FFFFFFF  # Keep positive for torch Generator


def collect_pack_contributions(
    samples: list[Any], length_fn: Callable[[Any], int]
) -> tuple[list[Any], dict[int, int], dict[int, int]]:
    """Collect component sample/token counts and contributor references."""
    from collections import defaultdict

    contributors = []
    sample_counts = defaultdict(int)
    token_counts = defaultdict(int)
    for sample in samples:
        contributors.extend(sample.meta.contribution_refs())
        for component, count in sample.meta.component_sample_counts.items():
            sample_counts[component] += count
        if sample.meta.component_token_counts is not None:
            for component, tokens in sample.meta.component_token_counts.items():
                token_counts[component] += tokens
        else:
            total = sum(sample.meta.component_sample_counts.values())
            for component, count in sample.meta.component_sample_counts.items():
                token_counts[component] += round(length_fn(sample) * count / total)
    return contributors, dict(sample_counts), dict(token_counts)


def _create_packed_record(
    bin_data: Bin,
    max_length: int,
    pack_payloads_fn: Callable[[List[Any]], Any],
    length_fn: Callable[[Any], int],
) -> Any:
    """Create a packed SampleRecord with component counts and contributor metadata.

    Collects contributor metadata locally and uses Zephon's ``pack_meta``
    helper to construct the packed record.
    """
    from zephon.ops.children import pack_meta
    from zephon.types import SampleRecord

    samples: List[SampleRecord] = bin_data.items
    payloads = [s.payload for s in samples]
    packed_payload = pack_payloads_fn(payloads)

    total_length = max_length - bin_data.remaining
    packing_efficiency = total_length / max_length if max_length > 0 else 0.0

    packed_payload["_packing_metadata"] = {
        "num_sequences": len(samples),
        "total_length": total_length,
        "packing_efficiency": packing_efficiency,
    }

    contributors, component_sample_counts, component_token_counts = (
        collect_pack_contributions(samples, length_fn)
    )

    base_meta = samples[0].meta
    primary_cursor = base_meta.cursor.child(0)

    packed_meta = pack_meta(
        primary_cursor=primary_cursor,
        contributors=contributors,
        lane_id=base_meta.lane_id,
        component_sample_counts=component_sample_counts,
        component_token_counts=component_token_counts,
    )
    return SampleRecord(meta=packed_meta, payload=packed_payload)
