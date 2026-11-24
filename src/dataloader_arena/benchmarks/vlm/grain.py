"""Grain VLM benchmark factory.

Composes grain primitives to build the VLM pipeline:
  Map(decode+preprocess+tokenize) → Filter → batch(window) → FlatMap(pack+extract)
  → batch(batch_size, batch_fn=construct+validate) → mp_prefetch

With process prefetch enabled, workers run the chain and the main process
consumes completed batches.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from grain._src.core.transforms import FlatMap

from dataloader_arena.context import LoaderConfig
from dataloader_arena.loaders.grain import (
    GrainConfig,
    GrainPrefetchConfig,
    _apply_prefetch,
    _make_source_iter_dataset,
)

from .config import (
    VLMConfig,
    VLMPackingConfig,
    build_image_transform,
    build_preprocessing_config,
)
from .transforms import (
    batch_transform_images,
    construct_batch_tensors,
    extract_sample_data,
    safe_parse_sample,
    shuffle_and_pack,
    tokenize_conversations,
    validate_batch,
)
from .transforms.preprocessing import preprocess_conversation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Grain transforms
# ═══════════════════════════════════════════════════════════════════════════════


class VLMSampleTransform:
    """Per-sample transform: decode → preprocess → tokenize.

    Implements the grain MapTransform protocol (has a ``map`` method).
    Runs in Grain worker processes via mp_prefetch.
    """

    def __init__(
        self,
        preprocessing_config: Any,
        image_transform: Callable | None,
        max_length: int,
        decoded_image_input: bool = True,
    ) -> None:
        self._prep = preprocessing_config
        self._image_transform = image_transform
        self._max_length = max_length
        self._decoded = decoded_image_input

    def __call__(self, raw_sample: dict) -> Any:
        return self.map(raw_sample)

    def map(self, raw_sample: dict) -> Any:
        """Decode, preprocess, and tokenize a single raw sample.

        Returns a Conversation object with token_ids and length set,
        or None if the sample should be dropped.
        """
        # 1. Decode (safe_parse_sample returns a new dict, does not mutate input)
        try:
            parsed = safe_parse_sample(raw_sample, decoded_image_input=self._decoded)
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse sample: {e}")
            return None

        text = parsed.get("text")
        images = parsed.get("image", [])
        if not text or not isinstance(text, list):
            return None

        # 2. Preprocess
        conv = preprocess_conversation(
            (text, images),
            tokens_per_image=self._prep.tokens_per_image,
            wrap_user_message=self._prep.wrap_user_message,
            wrap_assistant_message=self._prep.wrap_assistant_message,
            start_sequence=self._prep.start_sequence,
            enable_dynamic_tiling=self._prep.enable_dynamic_tiling,
            tiling_preprocessor=self._prep.tiling_preprocessor,
            tokens_per_tile=self._prep.tokens_per_tile,
        )
        if conv is None:
            return None

        # 3. Image transform before packing.
        if self._image_transform is not None and conv.images and len(conv.images) > 0:
            conv.images = batch_transform_images([conv.images], self._image_transform)[
                0
            ]

        # 4. Tokenize (single conversation)
        tokenize_conversations([conv], self._prep.tokenizer)
        conv.length = sum(len(m.token_ids or []) for m in conv.messages)

        if conv.length > self._max_length:
            return None

        return conv


class PackAndExtractTransform(FlatMap):
    """FlatMap transform: takes a window of conversations, packs, extracts.

    Inherits from grain FlatMap ABC when available.
    Input: list of Conversation objects (from ds.batch(window_size))
    Output: list of extracted sample dicts (one per packed bin)
    """

    max_fan_out: int = 128  # safe upper bound

    def __init__(self, max_length: int, seed: int, min_sequence_length: int) -> None:
        self._max_length = max_length
        self._seed = seed
        self._min_seq = min_sequence_length
        self._window_count = 0

    def flat_map(self, window: Sequence[Any]) -> list[dict]:
        self._window_count += 1
        gen = torch.Generator().manual_seed(self._seed + self._window_count)
        bins, _ = shuffle_and_pack(
            list(window),
            self._max_length,
            length_fn=lambda c: c.length,
            generator=gen,
            min_sequence_length=self._min_seq,
        )
        return [extract_sample_data(b.items) for b in bins if b.items]


# ═══════════════════════════════════════════════════════════════════════════════
# Batch function for tensor construction + validation
# ═══════════════════════════════════════════════════════════════════════════════


def _make_construct_validate_fn(
    max_length: int,
    pad_token_id: int,
    allow_cross_doc: bool,
    attn_implementation: str,
    tokenizer: Any,
) -> Callable[[Sequence[dict]], dict | None]:
    """Return a batch_fn for ds.batch() that tensorizes + validates."""

    def batch_fn(samples: Sequence[dict]) -> dict | None:
        batch = construct_batch_tensors(
            list(samples),
            max_length,
            pad_token_id,
            allow_cross_doc,
            attn_implementation,
        )
        result = validate_batch(batch, tokenizer, require_images=True)
        if not result.is_valid:
            logger.debug(f"Dropping invalid batch: {result.reason}")
            return None
        return batch

    return batch_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Create Grain VLM loader with composable pipeline stages.

    Pipeline (all stages run in Grain worker processes via mp_prefetch):
      LitData source → Map(decode+preprocess+tokenize) → Filter(drop None)
      → batch(window_size) → FlatMap(pack+extract) → batch(batch_size, construct+validate)
    """
    from grain._src.python.dataset.transformations.filter import FilterIterDataset
    from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset

    vlm_cfg = VLMConfig.parse(config.config.get("vlm", {}))
    packing_cfg = VLMPackingConfig.parse(config.config.get("packing", {}))
    grain_cfg = GrainConfig.parse(config.config)

    preprocessing_config = build_preprocessing_config(vlm_cfg)
    tokenizer = preprocessing_config.tokenizer
    pad_token_id = tokenizer.pad_token_id or 0
    batch_size = config.batch_size or 1

    # 1. Build LitData source
    source_ds, source_resolved = _make_source_iter_dataset(grain_cfg.source, config)

    # 2. Per-sample transforms: decode → preprocess → tokenize
    sample_transform = VLMSampleTransform(
        preprocessing_config=preprocessing_config,
        image_transform=build_image_transform(vlm_cfg),
        max_length=vlm_cfg.sequence_length,
        decoded_image_input=vlm_cfg.decoded_image_input,
    )
    ds = source_ds.map(sample_transform)

    # 3. Filter dropped samples (None from failed decode/preprocess/tokenize)
    ds = FilterIterDataset(ds, lambda x: x is not None)

    # 4. Accumulate window → pack → extract (FlatMap: 1 window → N bins)
    #    batch_fn=list keeps items as a plain list (grain's default stacks into numpy)
    ds = ds.batch(packing_cfg.window_size, batch_fn=list)
    pack_transform = PackAndExtractTransform(
        max_length=vlm_cfg.sequence_length,
        seed=packing_cfg.seed,
        min_sequence_length=packing_cfg.min_sequence_length,
    )
    ds = FlatMapIterDataset(ds, pack_transform)

    # 5. Batch bins → construct tensor batch + validate
    construct_fn = _make_construct_validate_fn(
        vlm_cfg.sequence_length,
        pad_token_id,
        vlm_cfg.allow_cross_document_attention,
        vlm_cfg.attn_implementation,
        tokenizer,
    )
    ds = ds.batch(batch_size, batch_fn=construct_fn, drop_remainder=True)

    # 6. Filter invalid batches (construct_fn returns None for invalid)
    ds = FilterIterDataset(ds, lambda x: x is not None)

    # 7. Prefetch (runs entire pipeline in workers)
    prefetch_cfg = grain_cfg.pipeline.prefetch
    if prefetch_cfg is None and (config.num_workers or 0) > 0:
        prefetch_cfg = GrainPrefetchConfig(mode="process", buffer_size=1)
    ds, prefetch_info = _apply_prefetch(
        ds, prefetch_cfg, runtime_num_workers=config.num_workers
    )

    ds.__arena_resolved_config__ = {  # type: ignore[attr-defined]
        "parsed": grain_cfg.model_dump(),
        "source": source_resolved,
        "vlm": vlm_cfg.model_dump(),
        "packing": packing_cfg.model_dump(),
        "prefetch": prefetch_info,
        "run": {
            "batch_size": batch_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
        },
        "paths": {
            "local": str(config.local_path),
            "remote": config.remote_path,
        },
    }
    return ds
