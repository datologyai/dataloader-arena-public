"""
Packing transforms for VLM data processing.

Shared by the Zephon and Grain preprocessing pipelines.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Generic, List, Tuple, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Bin(Generic[T]):
    """A bin for packing sequences."""

    items: List[T]
    remaining: int


def shuffle_items(
    items: List[T],
    generator: torch.Generator,
) -> List[T]:
    """Shuffle items using provided RNG generator."""
    # Shuffle conversations using seeded generator for reproducibility
    shuffled = items.copy()
    indices = torch.randperm(len(shuffled), generator=generator)
    shuffled = [shuffled[i] for i in indices]
    return shuffled


def first_fit_pack(
    items: List[T],
    max_length: int,
    length_fn: Callable[[T], int],
    min_sequence_length: int = 50,
) -> Tuple[List[Bin[T]], float]:
    """First-fit bin packing with unlimited bins, emitting all at end.


    Args:
        items: Items to pack (already shuffled)
        max_length: Maximum tokens per bin
        length_fn: Function to get length of an item
        min_sequence_length: Bins with remaining < this are considered "full"

    Returns:
        Tuple of (all_bins, packing_efficiency)
    """
    bins = []  # Active bins that can still accept sequences
    completed_bins = []  # Bins that cannot fit more sequences

    for item in items:
        item_len = length_fn(item)
        placed = False

        # Try to place in any existing bin that fits
        for i, bin_data in enumerate(bins):
            if bin_data.remaining >= item_len:
                bin_data.items.append(item)
                bin_data.remaining -= item_len
                placed = True

                # Move to completed if no space for minimum sequence
                if bin_data.remaining < min_sequence_length:
                    completed_bins.append(bin_data)
                    bins.pop(i)
                break

        # If not placed, create new bin
        if not placed:
            new_bin = Bin(
                items=[item],
                remaining=max_length - item_len,
            )

            # Add to appropriate list based on remaining capacity
            if new_bin.remaining < min_sequence_length:
                completed_bins.append(new_bin)
            else:
                bins.append(new_bin)

    # Combine all bins
    all_bins = bins + completed_bins

    # Calculate packing efficiency
    if all_bins:
        unused_tokens = sum(bin_data.remaining for bin_data in all_bins)
        total_tokens = len(all_bins) * max_length
        efficiency = (total_tokens - unused_tokens) / total_tokens
    else:
        efficiency = 0.0

    return all_bins, efficiency


def shuffle_and_pack(
    items: List[T],
    max_length: int,
    length_fn: Callable[[T], int],
    generator: torch.Generator,
    min_sequence_length: int = 50,
) -> Tuple[List[Bin[T]], float]:
    """Shuffle then first-fit pack. Convenience wrapper.

    Entry point for shared packing in the Grain and Zephon pipelines.

    Args:
        items: Items to pack
        max_length: Maximum tokens per bin
        length_fn: Function to get length of an item
        generator: PyTorch random generator for reproducible shuffling
        min_sequence_length: Bins with remaining < this are considered "full"

    Returns:
        Tuple of (all_bins, packing_efficiency)
    """
    shuffled = shuffle_items(items, generator)
    return first_fit_pack(shuffled, max_length, length_fn, min_sequence_length)
