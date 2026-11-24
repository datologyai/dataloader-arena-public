"""
Shared conversation preprocessing for VLM training.

Shared preprocessing for the Grain and Zephon pipelines.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import PIL.Image

from .conversation import (
    align_image_placeholders,
    build_vision_token_sequence,
    filter_placeholder_images,
    inject_vision_tokens,
    normalize_conversation_format,
)
from .models import Conversation, Message

logger = logging.getLogger(__name__)

# Module-level counter for rate-limited tiling debug logging.
# Guarded by a lock since preprocessing may run in threads (Zephon)
# or forked workers (Grain).
_tiling_log_lock = threading.Lock()
_tiling_log_counter = 0
_TILING_LOG_LIMIT = 3


def preprocess_conversation(
    conversation_data: Tuple[List[Dict], Optional[List[PIL.Image.Image]]],
    tokens_per_image: int,
    wrap_user_message: Callable[[str], str],
    wrap_assistant_message: Callable[[str], str],
    start_sequence: str,
    enable_dynamic_tiling: bool = False,
    tiling_preprocessor: Optional[Any] = None,
    tokens_per_tile: int = 243,
    debug: bool = False,
    debug_context: Optional[str] = None,
) -> Optional[Conversation]:
    """Preprocess a raw conversation into a Conversation object.

    This preprocessing is shared by Grain and Zephon.

    Args:
        conversation_data: Tuple of (conversation_list, images) where conversation_list
                          is already parsed and normalized (role/content format)
        tokens_per_image: Number of image pad tokens per image (fixed tiling)
        wrap_user_message: Function to wrap user message content
        wrap_assistant_message: Function to wrap assistant message content
        start_sequence: Start sequence prompt to prepend to first message
        enable_dynamic_tiling: Whether to use dynamic tiling
        tiling_preprocessor: Tiling preprocessor for dynamic tiling
        tokens_per_tile: Tokens per tile for dynamic tiling
        debug: Enable debug logging
        debug_context: Optional context string for debug logging

    Returns:
        Processed Conversation object or None if processing fails
    """
    conversation_list, images = conversation_data
    images = images if images is not None else []

    # Normalize source from/value format to role/content if needed
    conversation_list = normalize_conversation_format(conversation_list)
    if not conversation_list or any(
        "role" not in m or "content" not in m for m in conversation_list
    ):
        logger.warning(
            "Invalid conversation format (neither role/content nor from/value), dropping sample"
        )
        return None

    # Convert to Conversation object
    try:
        messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in conversation_list
        ]
        conversation = Conversation(messages=messages, images=images)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Error converting conversation: {e}, dropping sample")
        return None

    # Filter placeholder 1x1 images (LitData workaround)
    conversation.images = filter_placeholder_images(conversation.images)

    # Validate and fix image token alignment
    conversation = align_image_placeholders(
        conversation, debug=debug, debug_context=debug_context
    )
    if conversation is None:
        return None

    # Build vision token sequences
    if enable_dynamic_tiling and tiling_preprocessor and conversation.images:
        global _tiling_log_counter
        # Dynamic tiling: variable tokens per image
        tiled_images = []
        tile_counts = []

        for idx, img in enumerate(conversation.images):
            tiles, num_base_tiles, grid = tiling_preprocessor.split_image(img)
            tiled_images.extend(tiles)
            tile_counts.append(len(tiles))

            # Log first few tiling samples for debugging (rate-limited)
            with _tiling_log_lock:
                if _tiling_log_counter < _TILING_LOG_LIMIT:
                    logger.info(
                        f"  [TILING DEBUG] Image {idx}: size={img.size}, "
                        f"grid={grid}, base_tiles={num_base_tiles}, "
                        f"total_tiles={len(tiles)} (incl. thumbnail)"
                    )
                    _tiling_log_counter += 1

        conversation.images = tiled_images
        conversation.tile_counts = tile_counts

        vision_token_sequences = [
            build_vision_token_sequence(num_tiles * tokens_per_tile)
            for num_tiles in tile_counts
        ]
    else:
        # Fixed-length vision tokens
        num_images = len(conversation.images or [])
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

    # Add start sequence to first message
    if conversation.messages:
        conversation.messages[0].content = (
            start_sequence + conversation.messages[0].content
        )

    return conversation
