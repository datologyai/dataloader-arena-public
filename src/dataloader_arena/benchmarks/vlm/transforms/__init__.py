# Transforms module for VLM data processing
# Shared transforms for the Zephon and Grain preprocessing pipelines

from .batch_construction import (
    construct_batch_tensors,
    extract_sample_data,
)
from .constants import (
    ASSISTANT_ROLE,
    GPT_ROLE,
    HUMAN_ROLE,
    IGNORE_TOKEN_ID,
    IMAGE_PAD_TOKEN,
    MM_BOS_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN,
    USER_ROLE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from .conversation import (
    align_image_placeholders,
    build_vision_token_sequence,
    filter_placeholder_images,
    inject_vision_tokens,
    normalize_conversation_format,
)
from .decode import (
    deserialize_encoded_image_array,
    deserialize_pil_array,
    safe_parse_sample,
)
from .diagnostics import (
    DIAG_SUMMARY_INTERVAL,
    IMAGE_DIAG_ENABLED,
    diag_log,
    make_diag_counters,
)
from .dynamic_tiling import DynamicTilingPreprocessor
from .images import batch_transform_images
from .models import Conversation, Message
from .packing import (
    Bin,
    first_fit_pack,
    shuffle_and_pack,
    shuffle_items,
)
from .preprocessing import preprocess_conversation
from .preprocessing_config import PreprocessingConfig
from .tensors import (
    check_for_valid_images,
    count_images,
)
from .tokenization import tokenize_conversations
from .validation import (
    ValidationResult,
    classify_drop_reason,
    collect_diagnostics,
    get_vision_token_ids,
    validate_batch,
    validate_image_alignment,
    validate_vision_token_pairing,
)

__all__ = [
    # Packing
    "Bin",
    "shuffle_items",
    "first_fit_pack",
    "shuffle_and_pack",
    # Tokenization
    "tokenize_conversations",
    # Images
    "batch_transform_images",
    # Conversation
    "normalize_conversation_format",
    "filter_placeholder_images",
    "align_image_placeholders",
    "build_vision_token_sequence",
    "inject_vision_tokens",
    # Tensors
    "check_for_valid_images",
    "count_images",
    # Decode
    "deserialize_encoded_image_array",
    "deserialize_pil_array",
    "safe_parse_sample",
    # Batch construction
    "extract_sample_data",
    "construct_batch_tensors",
    # Validation
    "validate_batch",
    "validate_vision_token_pairing",
    "validate_image_alignment",
    "get_vision_token_ids",
    "collect_diagnostics",
    "classify_drop_reason",
    "ValidationResult",
    # Preprocessing
    "preprocess_conversation",
    # Constants
    "USER_ROLE",
    "HUMAN_ROLE",
    "ASSISTANT_ROLE",
    "GPT_ROLE",
    "MM_BOS_TOKEN",
    "PLACEHOLDER_IMAGE_TOKEN",
    "VISION_START_TOKEN",
    "VISION_END_TOKEN",
    "IMAGE_PAD_TOKEN",
    "IGNORE_TOKEN_ID",
    # Models
    "Message",
    "Conversation",
    # Preprocessing config
    "PreprocessingConfig",
    # Diagnostics
    "IMAGE_DIAG_ENABLED",
    "DIAG_SUMMARY_INTERVAL",
    "diag_log",
    "make_diag_counters",
    # Dynamic tiling
    "DynamicTilingPreprocessor",
]
