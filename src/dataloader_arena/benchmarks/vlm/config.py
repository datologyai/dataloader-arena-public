"""Shared configuration for the VLM preprocessing benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from dataloader_arena.config.models import ConfigModel

if TYPE_CHECKING:
    from .transforms.preprocessing_config import PreprocessingConfig


class VLMConfig(ConfigModel):
    """Shared configuration for VLM input pipeline benchmarks.

    Used by the Zephon and Grain benchmark factories.
    """

    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    tokenizer_revision: str | None = None
    additional_special_tokens: list[str] = Field(default_factory=list)
    tokens_per_image: int = Field(default=243, gt=0)
    tokens_per_tile: int = Field(default=243, gt=0)
    user_message_template: str = "<|im_start|>user\n{content}<|im_end|>\n"
    assistant_message_template: str = "<|im_start|>assistant\n{content}<|im_end|>\n"
    start_sequence: str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    image_processor_name: str | None = None
    image_processor_revision: str | None = None
    image_processor_use_fast: bool | None = None
    sequence_length: int = 2048
    enable_dynamic_tiling: bool = False
    attn_implementation: Literal["flash_attention_2", "sdpa"] = "flash_attention_2"
    allow_cross_document_attention: bool = False
    decoded_image_input: bool = False


class VLMPackingConfig(ConfigModel):
    """Packing settings shared across VLM backends (Grain, Zephon shared mode)."""

    window_size: int = 64
    seed: int = 42
    min_sequence_length: int = 50


class VLMZephonConfig(ConfigModel):
    """Zephon-specific pipeline settings for VLM benchmarks.

    Packing settings (window_size, seed, min_sequence_length) come from
    VLMPackingConfig which is shared with the Grain backend.
    """

    # shared preserves the throughput graph; per_lane isolates canonical lanes.
    shuffle_pack_mode: Literal["shared", "per_lane", "zephon"] = "shared"
    fused_transforms: bool = False
    process_parallelism: int = 24
    decode_parallelism: int | None = None
    preprocess_parallelism: int | None = None
    tokenize_parallelism: int | None = None
    extract_parallelism: int | None = None
    fused_parallelism: int | None = None
    batch_construct_parallelism: int = 1
    shuffle_pack_parallelism: int = 1
    # Zephon-native mode settings (shuffle_pack_mode="zephon")
    shuffle_buffer_size: int = 64
    shuffle_seed: int = 42
    num_bins: int = 8
    pack_algorithm: str = "first_fit"
    flush_strategy: str = "fullest"
    flatten_payload: bool = False
    # "historical" keeps only the last input's metadata, as the throughput runs do.
    # Checkpointing needs every contributor, or consumed inputs stay in flight.
    batch_metadata: Literal["historical", "preserve_contributors"] = "historical"


@dataclass(frozen=True)
class ImageProcessorTransform:
    """Pickle-safe Hugging Face image processing without loading model weights."""

    processor: Any

    def __call__(self, image: Any) -> Any:
        return self.processor(images=[image.convert("RGB")], return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)


def build_image_transform(config: VLMConfig) -> Any:
    """Use the default 224px transform unless a processor is selected."""
    if config.image_processor_name is None:
        return get_default_image_transform()
    from transformers import AutoImageProcessor

    options: dict[str, Any] = {"revision": config.image_processor_revision}
    if config.image_processor_use_fast is not None:
        options["use_fast"] = config.image_processor_use_fast
    return ImageProcessorTransform(
        AutoImageProcessor.from_pretrained(config.image_processor_name, **options)
    )


def _build_default_image_transform() -> Any:
    """Build the default image transform (lazy to avoid hard torchvision dep at import)."""
    from torchvision import transforms as tv_transforms

    return tv_transforms.Compose(
        [
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# Lazy singleton — built on first access.
_DEFAULT_IMAGE_TRANSFORM: Any = None


def get_default_image_transform() -> Any:
    """Return the default VLM image transform (creates it on first call)."""
    global _DEFAULT_IMAGE_TRANSFORM
    if _DEFAULT_IMAGE_TRANSFORM is None:
        _DEFAULT_IMAGE_TRANSFORM = _build_default_image_transform()
    return _DEFAULT_IMAGE_TRANSFORM


def build_preprocessing_config(config: VLMConfig) -> PreprocessingConfig:
    """Load a tokenizer and construct the shared worker-safe preprocessing settings."""
    from transformers import AutoTokenizer

    from .transforms.preprocessing_config import PreprocessingConfig

    tiling_preprocessor = None
    if config.enable_dynamic_tiling:
        from .transforms.dynamic_tiling import DynamicTilingPreprocessor

        tiling_preprocessor = DynamicTilingPreprocessor(
            tile_size=384, min_num_tiles=1, max_num_tiles=12, use_thumbnail=True
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, revision=config.tokenizer_revision
    )
    if config.additional_special_tokens:
        missing = [
            t
            for t in config.additional_special_tokens
            if t not in tokenizer.all_special_tokens
        ]
        if missing:
            # Transformers 5 renamed the extra-token attribute. The dictionary
            # key below remains supported by both 4.x and 5.x.
            existing = getattr(tokenizer, "extra_special_tokens", None)
            if existing is None:
                existing = tokenizer.additional_special_tokens
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [*existing, *missing]},
            )
    return PreprocessingConfig(
        tokenizer=tokenizer,
        tokens_per_image=config.tokens_per_image,
        user_message_template=config.user_message_template,
        assistant_message_template=config.assistant_message_template,
        start_sequence=config.start_sequence,
        enable_dynamic_tiling=config.enable_dynamic_tiling,
        tiling_preprocessor=tiling_preprocessor,
        tokens_per_tile=config.tokens_per_tile,
    )
