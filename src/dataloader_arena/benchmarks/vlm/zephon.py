"""Zephon VLM benchmark factory.

Builds the VLM pipeline on top of the arena's shared Zephon infrastructure
(work source, fetch, observability, pipeline options) and adds VLM-specific
transform stages: decode → preprocess → tokenize → shuffle+pack → extract →
batch → construct+validate.

"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Iterable, Iterator

from dataloader_arena.context import LoaderConfig
from dataloader_arena.loaders.zephon import (
    build_zephon_pipeline,
    materialize_config,
)

from .config import (
    VLMConfig,
    VLMPackingConfig,
    VLMZephonConfig,
    build_image_transform,
    build_preprocessing_config,
)
from .lane_packing import add_lane_shuffle_pack
from .ops import (
    add_shared_shuffle_pack,
    make_construct_and_validate_batch_fn,
    make_decode_fn,
    make_fused_transform_fn,
    make_pack_merge_fn,
    make_preprocess_fn,
    make_tokenize_fn,
)
from .transforms.batch_construction import extract_sample_data, extract_sample_data_flat

logger = logging.getLogger(__name__)


class VLMPipelineWrapper:
    """Thin wrapper that unwraps Zephon SampleBatch records into batch dicts.

    Both Grain and Zephon yield batch dictionaries for the measurement harness.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._iterator: Any = None

    def checkpoint(self) -> dict[str, Any]:
        """Capture the same pipeline consumed by this loader."""
        return self._pipeline.checkpoint()

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore before starting a new iterator."""
        self._pipeline.restore(checkpoint)

    def checkpoint_metrics(self) -> dict[str, Any] | None:
        method = getattr(self._pipeline, "checkpoint_metrics", None)
        return method() if callable(method) else None

    def replay_metrics(self) -> dict[str, Any] | None:
        method = getattr(self._pipeline, "replay_metrics", None)
        return method() if callable(method) else None

    def close(self) -> None:
        """Close the live iterator before releasing pipeline workers."""
        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        close_pipeline = getattr(self._pipeline, "close", None)
        if callable(close_pipeline):
            close_pipeline()

    def __iter__(self) -> "VLMPipelineWrapper":
        self._iterator = iter(self._pipeline)
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._iterator is None:
            self._iterator = iter(self._pipeline)
        sample_batch = next(self._iterator)  # StopIteration propagates
        return sample_batch.records[0].payload


def _set_worker_thread_limits() -> None:
    """Limit library thread counts in pipeline worker subprocesses."""
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in env_vars.items():
        prev = os.environ.get(key)
        if prev is None:
            os.environ[key] = value


def build_vlm_pipeline(config: LoaderConfig) -> tuple[Any, dict[str, Any]]:
    """Build the shared VLM graph, retaining Zephon's checkpoint/restore interface."""
    _set_worker_thread_limits()

    # ── Parse configs ─────────────────────────────────────────────────────
    vlm_cfg = VLMConfig.parse(config.config.get("vlm", {}))
    packing_cfg = VLMPackingConfig.parse(config.config.get("packing", {}))
    vlm_zephon = VLMZephonConfig.parse(config.config.get("vlm_zephon", {}))

    micro_batch_size = config.batch_size or 1

    # ── Build base pipeline via arena infrastructure ──────────────────────
    zephon_cfg, resolved = materialize_config(config)
    pipeline, resolved = build_zephon_pipeline(config, zephon_cfg, resolved)

    # Build shared preprocessing settings.
    preprocessing_config = build_preprocessing_config(vlm_cfg)
    image_transform = build_image_transform(vlm_cfg)

    tokenizer = preprocessing_config.tokenizer
    pad_token_id = tokenizer.pad_token_id or 0
    max_length = vlm_cfg.sequence_length

    # ── Resolve per-operator parallelism ──────────────────────────────────
    decode_par = vlm_zephon.decode_parallelism or (zephon_cfg.fetch_parallelism or 4)
    preprocess_par = vlm_zephon.preprocess_parallelism or vlm_zephon.process_parallelism
    tokenize_par = vlm_zephon.tokenize_parallelism or vlm_zephon.process_parallelism
    extract_par = vlm_zephon.extract_parallelism or vlm_zephon.process_parallelism
    fused_par = vlm_zephon.fused_parallelism or vlm_zephon.process_parallelism

    # ── Add VLM transform stages ─────────────────────────────────────────
    pack_merge_fn = make_pack_merge_fn(
        fuse_extract=vlm_zephon.fused_transforms,
        flatten_payload=vlm_zephon.flatten_payload,
    )

    if vlm_zephon.fused_transforms:
        fused_fn = make_fused_transform_fn(
            tokenizer=tokenizer,
            tokens_per_image=preprocessing_config.tokens_per_image,
            wrap_user_message=preprocessing_config.wrap_user_message,
            wrap_assistant_message=preprocessing_config.wrap_assistant_message,
            start_sequence=preprocessing_config.start_sequence,
            enable_dynamic_tiling=preprocessing_config.enable_dynamic_tiling,
            tiling_preprocessor=preprocessing_config.tiling_preprocessor,
            tokens_per_tile=preprocessing_config.tokens_per_tile,
            image_transform=image_transform,
            max_length=max_length,
            decoded_image_input=vlm_cfg.decoded_image_input,
            flatten_payload=vlm_zephon.flatten_payload,
        )
        pipeline = pipeline.map_transform(
            fused_fn, drop_none=True, parallelism=fused_par
        )
        if config.verbose:
            print(f"[vlm-zephon] fused mode: parallelism={fused_par}", file=sys.stderr)
    else:
        decode_fn = make_decode_fn(decoded_image_input=vlm_cfg.decoded_image_input)
        preprocess_fn = make_preprocess_fn(
            tokens_per_image=preprocessing_config.tokens_per_image,
            wrap_user_message=preprocessing_config.wrap_user_message,
            wrap_assistant_message=preprocessing_config.wrap_assistant_message,
            start_sequence=preprocessing_config.start_sequence,
            enable_dynamic_tiling=preprocessing_config.enable_dynamic_tiling,
            tiling_preprocessor=preprocessing_config.tiling_preprocessor,
            tokens_per_tile=preprocessing_config.tokens_per_tile,
            image_transform=image_transform,
        )
        tokenize_fn = make_tokenize_fn(
            tokenizer, max_length=max_length, flatten_payload=vlm_zephon.flatten_payload
        )

        pipeline = pipeline.map_transform(
            decode_fn, drop_none=True, parallelism=decode_par
        )
        pipeline = pipeline.map_transform(
            preprocess_fn, drop_none=True, parallelism=preprocess_par
        )
        pipeline = pipeline.map_transform(tokenize_fn, parallelism=tokenize_par)
        if config.verbose:
            print(
                f"[vlm-zephon] split mode: decode={decode_par}, "
                f"preprocess={preprocess_par}, tokenize={tokenize_par}",
                file=sys.stderr,
            )

    # ── Shuffle + Pack ────────────────────────────────────────────────────
    if vlm_zephon.shuffle_pack_mode in ("shared", "per_lane"):
        logger.info("Using %s shuffle+pack mode", vlm_zephon.shuffle_pack_mode)
        pack = (
            add_lane_shuffle_pack
            if vlm_zephon.shuffle_pack_mode == "per_lane"
            else add_shared_shuffle_pack
        )
        pipeline = pack(
            pipeline,
            window_size=packing_cfg.window_size,
            max_length=max_length,
            length_fn=lambda r: r.payload.get("length", 0),
            seed=packing_cfg.seed,
            min_sequence_length=packing_cfg.min_sequence_length,
            pack_payloads_fn=pack_merge_fn,
            parallelism=vlm_zephon.shuffle_pack_parallelism,
        )
    else:
        logger.info("Using Zephon-native shuffle+pack mode (streaming)")
        pipeline = pipeline.shuffle(
            buffer_size=vlm_zephon.shuffle_buffer_size,
            seed=vlm_zephon.shuffle_seed,
        )
        pipeline = pipeline.pack_sequences(
            max_length=max_length,
            num_bins=vlm_zephon.num_bins,
            length_fn=lambda r: r.payload.get("length", 0),
            algorithm=vlm_zephon.pack_algorithm,
            min_sequence_length=packing_cfg.min_sequence_length,
            flush_strategy=vlm_zephon.flush_strategy,
            pack_payloads=pack_merge_fn,
        )

    # ── Extract (if not fused) ────────────────────────────────────────────
    if not vlm_zephon.fused_transforms:
        if vlm_zephon.flatten_payload:

            def _extract_sample(payload: Dict[str, Any]) -> Dict[str, Any]:
                return extract_sample_data_flat(payload["flat_samples"])
        else:

            def _extract_sample(payload: Dict[str, Any]) -> Dict[str, Any]:
                conversations = payload["conversations"]
                return extract_sample_data(conversations)

        pipeline = pipeline.map_transform(_extract_sample, parallelism=extract_par)

    # ── Batch ─────────────────────────────────────────────────────────────
    pipeline = pipeline.batch(micro_batch_size, drop_last=True)

    # ── Post-batch construct + validate ───────────────────────────────────
    construct_validate_fn = make_construct_and_validate_batch_fn(
        max_length=max_length,
        pad_token_id=pad_token_id,
        allow_cross_document_attention=vlm_cfg.allow_cross_document_attention,
        attn_implementation=vlm_cfg.attn_implementation,
        tokenizer=tokenizer,
        require_images=True,
        preserve_contributors=vlm_zephon.batch_metadata == "preserve_contributors",
    )
    pipeline = pipeline.map_batch(
        construct_validate_fn,
        drop_none=True,
        parallelism=vlm_zephon.batch_construct_parallelism,
    )

    if config.verbose:
        print(f"[vlm-zephon] Pipeline plan:\n{pipeline.explain()}", file=sys.stderr)

    resolved["pipeline_plan"] = pipeline.explain()
    resolved["vlm"] = vlm_cfg.model_dump()
    resolved["packing"] = packing_cfg.model_dump()
    resolved["vlm_zephon"] = vlm_zephon.model_dump()
    return pipeline, resolved


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Wrap the shared graph for throughput and TTFB measurements."""
    pipeline, resolved = build_vlm_pipeline(config)
    wrapper = VLMPipelineWrapper(pipeline)
    wrapper.__arena_resolved_config__ = resolved  # type: ignore[attr-defined]
    return wrapper
