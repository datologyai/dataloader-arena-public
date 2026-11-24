"""
Conversation transforms for VLM data processing.

Shared by the Zephon and Grain preprocessing pipelines.
"""

import logging
from typing import Callable, List, Optional

import PIL.Image

from .constants import (
    ASSISTANT_ROLE,
    GPT_ROLE,
    HUMAN_ROLE,
    IMAGE_PAD_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN,
    USER_ROLE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from .models import Conversation

logger = logging.getLogger(__name__)


def normalize_conversation_format(conversation: List[dict]) -> List[dict]:
    """
    Normalize conversation format from source from/value to canonical role/content.

    Returns a new list of normalized messages. Messages already in role/content
    format are passed through unchanged. Source from/value messages are converted.
    Messages matching neither format are passed through unchanged.
    """
    normalized = []
    for message in conversation:
        if "role" in message and "content" in message:
            normalized.append(message)
        elif "from" in message and "value" in message:
            from_value = message["from"]
            if from_value == HUMAN_ROLE:
                role = USER_ROLE
            elif from_value == GPT_ROLE:
                role = ASSISTANT_ROLE
            else:
                role = from_value
            normalized.append({"role": role, "content": message["value"]})
        else:
            normalized.append(message)
    return normalized


def filter_placeholder_images(
    images: Optional[List[PIL.Image.Image]],
) -> List[PIL.Image.Image]:
    """
    Filter out 1x1 placeholder images.

    Images with size (1, 1) are treated as placeholders. A missing or empty
    image list produces an empty list.

    Args:
        images: List of PIL images (may contain 1x1 placeholders)

    Returns:
        Filtered list with 1x1 images removed
    """
    if images is None or (isinstance(images, (list, tuple)) and len(images) == 0):
        return []

    filtered_images = []
    for img in images:
        if img is not None and hasattr(img, "size") and img.size != (1, 1):
            filtered_images.append(img)
    return filtered_images


def align_image_placeholders(
    conversation: Conversation,
    debug: bool = False,
    debug_context: Optional[str] = None,
) -> Optional[Conversation]:
    """
    Validate and fix image token alignment in conversation.


    If the number of <image> placeholders doesn't match the number of images,
    this function removes all placeholders and prepends the correct number to
    the first user message.

    Args:
        conversation: Conversation object with messages and images
        debug: If True, log diagnostic information
        debug_context: Optional string context for debugging (e.g., raw JSON or payload repr).
                       Logged when sample is dropped due to no user message.

    Returns:
        The conversation (mutated in place) or None if it should be dropped
    """
    # Count image tokens in user messages
    num_image_tokens = 0
    for message in conversation.messages:
        if message.role == "user":
            num_image_tokens += message.content.count(PLACEHOLDER_IMAGE_TOKEN)

    conversation_images = conversation.images if conversation.images is not None else []

    if num_image_tokens != len(conversation_images):
        if debug:
            # Log counts and a small snippet of the first user message
            first_user = None
            for _m in conversation.messages:
                if _m.role == USER_ROLE:
                    first_user = _m
                    break
            snippet = None
            if first_user and isinstance(first_user.content, str):
                snippet = first_user.content[:240]
            logger.warning(
                "[Diagnostics] Pre-normalization mismatch: placeholders=%d, images=%d",
                num_image_tokens,
                len(conversation_images),
            )
            if snippet is not None:
                logger.warning(
                    "[Diagnostics] First user message snippet (pre-normalization): %r",
                    snippet,
                )

        # Remove all image tokens from user messages
        for message in conversation.messages:
            if message.role == "user":
                message.content = message.content.replace(PLACEHOLDER_IMAGE_TOKEN, "")

        # Build correct placeholder prefix with one token per image to keep alignment strict
        placeholder_prefix = ""
        num_imgs = len(conversation_images) if conversation_images is not None else 0
        if num_imgs > 0:
            placeholder_tokens = [PLACEHOLDER_IMAGE_TOKEN] * num_imgs
            placeholder_prefix = " ".join(placeholder_tokens) + " "

        # Prepend correct number of image tokens to the first user message
        inserted = False
        for message in conversation.messages:
            if message.role == "user":
                message.content = placeholder_prefix + message.content
                inserted = True
                break

        if not inserted:
            # No user message found to host placeholders. Always warn at sample level,
            # even when debug mode is OFF, to surface data issues early.
            logger.warning(
                "[Data Warning] No user message found to insert <image> placeholders; "
                "images_present=%d, original_placeholder_count=%d. "
                "Dropping sample.",
                len(conversation_images),
                num_image_tokens,
            )
            if debug and debug_context:
                logger.warning(
                    "[Diagnostics] Full conversation context (no-user-message case): %s",
                    debug_context,
                )
            return None

    return conversation


def build_vision_token_sequence(num_tokens: int) -> str:
    """
    Build vision token sequence for a given number of image pad tokens.


    Args:
        num_tokens: Number of image pad tokens (e.g., 243 for fixed, or tiles * tokens_per_tile for dynamic)

    Returns:
        Vision token sequence: <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
    """
    return VISION_START_TOKEN + (IMAGE_PAD_TOKEN * num_tokens) + VISION_END_TOKEN


def inject_vision_tokens(
    conversation: Conversation,
    vision_token_sequences: List[str],
    wrap_user_message: Callable,
    wrap_assistant_message: Callable,
) -> None:
    """
    Replace <image> placeholders with vision token sequences and wrap messages.


    Mutates conversation.messages in place.

    Args:
        conversation: Conversation object with messages containing <image> placeholders
        vision_token_sequences: List of vision token sequences (one per image).
                                For fixed tiling: all sequences are identical.
                                For dynamic tiling: sequences vary by image tile count.
        wrap_user_message: Function to wrap user message content
        wrap_assistant_message: Function to wrap assistant message content
    """
    image_idx = 0

    for message in conversation.messages:
        # Wrap messages with chat template
        if message.role == USER_ROLE:
            message.content = wrap_user_message(message.content)
        elif message.role == ASSISTANT_ROLE:
            message.content = wrap_assistant_message(message.content)

        # Replace each <image> placeholder with its corresponding vision token sequence
        while PLACEHOLDER_IMAGE_TOKEN in message.content and image_idx < len(
            vision_token_sequences
        ):
            message.content = message.content.replace(
                PLACEHOLDER_IMAGE_TOKEN,
                vision_token_sequences[image_idx],
                1,  # Replace only first occurrence
            )
            image_idx += 1
