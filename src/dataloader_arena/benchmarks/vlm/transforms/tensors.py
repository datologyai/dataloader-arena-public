"""Image tensor validation helpers shared by the VLM pipelines."""

import logging
from typing import List, Union

import torch

logger = logging.getLogger(__name__)


def check_for_valid_images(images: List) -> bool:
    """Check if batch contains any valid images.


    Args:
        images: List of image tensors or lists of tensors

    Returns:
        True if at least one valid image tensor exists
    """
    for img in images:
        if isinstance(img, torch.Tensor) and img.numel() > 0:
            return True
        elif isinstance(img, list) and len(img) > 0:
            # Check if list contains valid tensors
            for sub_img in img:
                if isinstance(sub_img, torch.Tensor) and sub_img.numel() > 0:
                    return True
    return False


def count_images(sample_images: Union[List, torch.Tensor]) -> int:
    """Count number of images in a sample.


    Args:
        sample_images: Either a list of images or a tensor with batch dimension

    Returns:
        Number of images
    """
    if isinstance(sample_images, list):
        return len(sample_images)
    elif hasattr(sample_images, "shape"):
        return sample_images.shape[0]
    return 0
