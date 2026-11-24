"""
Image transforms for VLM data processing.

Shared by the Zephon and Grain preprocessing pipelines.
"""

import logging
from typing import Callable, List, Optional, Union

import PIL.Image
import torch

logger = logging.getLogger(__name__)


def batch_transform_images(
    batch_images: List[List[PIL.Image.Image]],
    image_transform: Optional[Callable[[PIL.Image.Image], torch.Tensor]],
) -> List[Union[torch.Tensor, List]]:
    """
    Apply image transformations efficiently to all images in the batch.


    Args:
        batch_images: List of lists where batch_images[i] contains PIL images for sample i
        image_transform: Transform function to apply to each image (e.g., torchvision.transforms)

    Returns:
        List of tensors where each tensor contains all transformed images for that sample,
        or empty list if the sample has no images
    """
    if not batch_images or image_transform is None:
        return batch_images

    # Collect all images and track per-sample counts
    all_images = []
    sample_image_counts = []

    for sample_images in batch_images:
        sample_image_counts.append(len(sample_images))
        all_images.extend(sample_images)

    if not all_images:
        # No images to transform, return empty list for each sample
        return [[] for _ in batch_images]

    # Transform all images — strict mode: raise on failure.
    # By this point, all images should be valid PIL images (corrupt/invalid
    # images are filtered in decode/filter stages). A failure here is a bug.
    transformed_images = []
    for i, img in enumerate(all_images):
        if img is None:
            raise RuntimeError(
                f"Image {i} is None. Images should be validated before "
                f"reaching transform. Check decode/filter stages."
            )
        try:
            transformed_img = image_transform(img)
            transformed_images.append(transformed_img)
        except Exception as e:
            raise RuntimeError(
                f"Image transform failed on image {i} "
                f"(type={type(img).__name__}). Images should be validated "
                f"before reaching transform. Check decode/filter stages. "
                f"Original error: {e}"
            ) from e

    # Reorganize transformed images back into per-sample structure
    result = []
    img_idx = 0
    for sample_idx, count in enumerate(sample_image_counts):
        if count == 0:
            result.append([])
        else:
            sample_tensors = transformed_images[img_idx : img_idx + count]
            img_idx += count
            stacked = torch.stack(sample_tensors)
            result.append(stacked)

    return result
