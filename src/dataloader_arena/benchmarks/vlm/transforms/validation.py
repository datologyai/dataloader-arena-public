"""
Shared batch validation functions for VLM training.

Shared validation for the Grain and Zephon pipelines.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .tensors import check_for_valid_images, count_images

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of batch validation with optional diagnostics."""

    is_valid: bool
    reason: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = field(default_factory=dict)


def classify_drop_reason(reason: Optional[str]) -> str:
    """Classify a batch drop reason string into a category.

    Used for drop-rate metrics in both pipelines.
    """
    if reason is None:
        return "unknown"
    if "no valid images" in reason:
        return "no_images"
    if "Vision token mismatch" in reason or "vision token mismatch" in reason:
        return "vision_mismatch"
    if "mismatch" in reason.lower():
        return "alignment"
    return "unknown"


def get_vision_token_ids(
    tokenizer: Any,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Safely extract vision token IDs from tokenizer.

    Returns:
        Tuple of (vision_start_id, vision_end_id, image_pad_id)
    """
    try:
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        return vision_start_id, vision_end_id, image_pad_id
    except (KeyError, AttributeError, TypeError):
        return None, None, None


def validate_vision_token_pairing(
    input_ids: torch.Tensor,
    vision_start_id: int,
    vision_end_id: int,
    per_sample: bool = False,
) -> Tuple[bool, List[Tuple[int, int, int]]]:
    """Check that vision_start and vision_end token counts match.

    Args:
        input_ids: Batch input IDs (1D flat or 2D padded)
        vision_start_id: Token ID for vision_start
        vision_end_id: Token ID for vision_end
        per_sample: If True and input is 2D, validate each sample

    Returns:
        Tuple of (is_valid, list of (sample_idx, start_count, end_count) for mismatches)
    """
    mismatches = []

    if input_ids.dim() == 1:
        # Flat batch - single validation
        start_count = (input_ids == vision_start_id).sum().item()
        end_count = (input_ids == vision_end_id).sum().item()
        if start_count != end_count:
            mismatches.append((0, start_count, end_count))
        return len(mismatches) == 0, mismatches

    if not per_sample:
        # Batch-level validation
        start_count = (input_ids == vision_start_id).sum().item()
        end_count = (input_ids == vision_end_id).sum().item()
        if start_count != end_count:
            mismatches.append((-1, start_count, end_count))  # -1 indicates batch-level
        return len(mismatches) == 0, mismatches

    # Per-sample validation for padded batches
    batch_size = input_ids.shape[0]
    for idx in range(batch_size):
        sample_ids = input_ids[idx]
        start_count = (sample_ids == vision_start_id).sum().item()
        end_count = (sample_ids == vision_end_id).sum().item()
        if start_count != end_count:
            mismatches.append((idx, start_count, end_count))

    return len(mismatches) == 0, mismatches


def validate_image_alignment(
    input_ids: torch.Tensor,
    images: List[Any],
    vision_start_id: int,
    per_sample: bool = False,
) -> Tuple[bool, List[Tuple[int, int, int]]]:
    """Validate image count matches vision token count.

    Args:
        input_ids: Batch input IDs (1D flat or 2D padded)
        images: Batch images
        vision_start_id: Token ID for vision_start
        per_sample: If True and input is 2D, validate each sample

    Returns:
        Tuple of (is_valid, list of (sample_idx, vision_count, image_count) for mismatches)
    """
    mismatches = []

    if input_ids.dim() == 1 or not per_sample:
        # Flat batch or batch-level validation
        total_vision = (input_ids == vision_start_id).sum().item()
        total_images = sum(count_images(img) for img in images)
        if total_vision != total_images:
            mismatches.append((-1, total_vision, total_images))
        return len(mismatches) == 0, mismatches

    # Per-sample validation for padded batches
    batch_size = input_ids.shape[0]
    for idx in range(batch_size):
        sample_ids = input_ids[idx]
        sample_images = images[idx] if idx < len(images) else []

        vision_count = (sample_ids == vision_start_id).sum().item()
        image_count = count_images(sample_images)

        if vision_count > 0 and vision_count != image_count:
            mismatches.append((idx, vision_count, image_count))

    return len(mismatches) == 0, mismatches


def collect_diagnostics(
    input_ids: torch.Tensor,
    images: List[Any],
    vision_start_id: int,
    vision_end_id: int,
    image_pad_id: Optional[int],
    sample_idx: int,
) -> Dict[str, Any]:
    """Collect detailed diagnostics for a sample with validation issues.

    Args:
        input_ids: Sample input IDs (1D tensor)
        images: Sample images
        vision_start_id: Token ID for vision_start
        vision_end_id: Token ID for vision_end
        image_pad_id: Token ID for image_pad (optional)
        sample_idx: Index of the sample in batch

    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {"sample_idx": sample_idx}

    try:
        # Token positions
        start_positions = (
            (input_ids == vision_start_id).nonzero(as_tuple=True)[0].tolist()
        )
        end_positions = (input_ids == vision_end_id).nonzero(as_tuple=True)[0].tolist()
        diagnostics["start_positions"] = start_positions[:20]  # Limit to first 20
        diagnostics["end_positions"] = end_positions[:20]

        # IMAGE_PAD counts between vision token pairs
        if image_pad_id is not None:
            pad_counts = []
            for i in range(min(len(start_positions), len(end_positions))):
                s, e = start_positions[i], end_positions[i]
                if e > s + 1:
                    pad_count = (input_ids[s + 1 : e] == image_pad_id).sum().item()
                else:
                    pad_count = 0
                pad_counts.append(pad_count)
            diagnostics["pad_counts_between_pairs"] = pad_counts

        # Image shapes
        img_shapes = []
        if isinstance(images, list):
            for img in images:
                if hasattr(img, "shape"):
                    img_shapes.append(tuple(img.shape))
        elif hasattr(images, "shape"):
            img_shapes.append(tuple(images.shape))
        diagnostics["image_shapes"] = img_shapes

    except (KeyError, AttributeError, TypeError, IndexError) as e:
        diagnostics["error"] = str(e)

    return diagnostics


def validate_batch(
    batch: Dict[str, Any],
    tokenizer: Any,
    require_images: bool = True,
    per_sample: bool = True,
    collect_diagnostics_on_failure: bool = False,
) -> ValidationResult:
    """Validate a batch for VLM training.

    Checks:
    1. Has valid images (if require_images=True)
    2. Vision start/end token counts match
    3. Image count matches vision token count

    Args:
        batch: Batch dict with input_ids, images
        tokenizer: Tokenizer with vision tokens
        require_images: Drop text-only batches
        per_sample: Validate each sample (padded batches only)
        collect_diagnostics_on_failure: Include detailed diagnostics on validation failure

    Returns:
        ValidationResult with is_valid and optional diagnostics
    """
    input_ids = batch.get("input_ids")
    images = batch.get("images", [])

    if input_ids is None:
        return ValidationResult(False, "Missing input_ids")

    diagnostics = {} if collect_diagnostics_on_failure else None
    is_flat = input_ids.dim() == 1

    # Check 1: Valid images
    if require_images:
        has_images = check_for_valid_images(images)
        if not has_images:
            reason = "Text-only batch (no valid images)"
            if diagnostics is not None:
                # Collect image shape info
                try:
                    shapes = []
                    for sample_images in images:
                        if isinstance(sample_images, list) and sample_images:
                            shapes.append(
                                [
                                    tuple(getattr(t, "shape", ()))
                                    for t in sample_images
                                    if hasattr(t, "shape")
                                ]
                            )
                        elif hasattr(sample_images, "shape"):
                            shapes.append(tuple(sample_images.shape))
                        else:
                            shapes.append([])
                    diagnostics["image_shapes_per_sample"] = shapes
                except (AttributeError, TypeError, IndexError):
                    pass
            return ValidationResult(False, reason, diagnostics)

    # Get vision token IDs
    vision_start_id, vision_end_id, image_pad_id = get_vision_token_ids(tokenizer)
    if vision_start_id is None or vision_end_id is None:
        return ValidationResult(False, "Vision tokens not in tokenizer")

    # Check 2: Vision token pairing (start == end counts)
    pairing_valid, pairing_mismatches = validate_vision_token_pairing(
        input_ids, vision_start_id, vision_end_id, per_sample=per_sample and not is_flat
    )
    if not pairing_valid:
        # Log a warning for every failing sample
        for idx, starts, ends in pairing_mismatches:
            if idx == -1:
                logger.warning(f"Vision token mismatch: {starts} starts vs {ends} ends")
            else:
                logger.warning(
                    f"Sample {idx}: Vision token mismatch ({starts} starts vs {ends} ends)"
                )

        # Build reason from first mismatch (for ValidationResult)
        idx, starts, ends = pairing_mismatches[0]
        if idx == -1:
            reason = f"Vision token mismatch: {starts} starts vs {ends} ends"
        else:
            reason = (
                f"Sample {idx}: Vision token mismatch ({starts} starts vs {ends} ends)"
            )

        if diagnostics is not None:
            diagnostics["pairing_mismatches"] = pairing_mismatches
            # Collect per-sample diagnostics for ALL failing samples
            sample_diagnostics = []
            for m_idx, _, _ in pairing_mismatches:
                if m_idx >= 0 and input_ids.dim() > 1:
                    sample_diagnostics.append(
                        collect_diagnostics(
                            input_ids[m_idx],
                            images[m_idx] if m_idx < len(images) else [],
                            vision_start_id,
                            vision_end_id,
                            image_pad_id,
                            m_idx,
                        )
                    )
            if sample_diagnostics:
                diagnostics["sample_diagnostics"] = sample_diagnostics

        return ValidationResult(False, reason, diagnostics)

    # Check 3: Image alignment (images == vision tokens)
    # Skip for flat batches.
    # With dynamic tiling, images are flattened tiles (M) but vision_start
    # count reflects original images (N). Batch-level M != N is expected.
    if not is_flat:
        alignment_valid, alignment_mismatches = validate_image_alignment(
            input_ids, images, vision_start_id, per_sample=per_sample
        )
        if not alignment_valid:
            # Log a warning for every failing sample
            for idx, vision_count, image_count in alignment_mismatches:
                if idx == -1:
                    logger.warning(
                        f"Image count mismatch: {image_count} images vs {vision_count} vision tokens"
                    )
                else:
                    logger.warning(
                        f"Sample {idx}: {image_count} images vs {vision_count} vision tokens"
                    )

            # Build reason from first mismatch (for ValidationResult)
            idx, vision_count, image_count = alignment_mismatches[0]
            if idx == -1:
                reason = f"Image count mismatch: {image_count} images vs {vision_count} vision tokens"
            else:
                reason = f"Sample {idx}: {image_count} images vs {vision_count} vision tokens"

            if diagnostics is not None:
                diagnostics["alignment_mismatches"] = alignment_mismatches
                # Collect per-sample diagnostics for ALL failing samples
                sample_diagnostics = []
                for m_idx, _, _ in alignment_mismatches:
                    if m_idx >= 0 and input_ids.dim() > 1:
                        sample_diagnostics.append(
                            collect_diagnostics(
                                input_ids[m_idx],
                                images[m_idx] if m_idx < len(images) else [],
                                vision_start_id,
                                vision_end_id,
                                image_pad_id,
                                m_idx,
                            )
                        )
                if sample_diagnostics:
                    diagnostics["sample_diagnostics"] = sample_diagnostics

            return ValidationResult(False, reason, diagnostics)

    return ValidationResult(True, None, diagnostics)
