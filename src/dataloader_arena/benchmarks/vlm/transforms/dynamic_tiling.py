# Algorithm adapted from InternVL's dynamic_preprocess and
# find_closest_aspect_ratio functions:
# https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/dataset.py
#
# MIT License
#
# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2024 OpenGVLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Dynamic Tiling Preprocessor for Native Resolution Vision Encoding

Implements aspect-ratio-aware image tiling for variable-resolution VLM training.
Based on InternVL's dynamic_preprocess algorithm.
"""

from typing import List, Tuple

import PIL.Image


class DynamicTilingPreprocessor:
    """
    Splits images into aspect-ratio-aware tiles for native resolution processing.

    Algorithm:
    1. Generate all valid grid configurations (e.g., 1x1, 2x1, 2x2, 3x2, 4x3, etc.)
    2. Find grid that best matches image aspect ratio
    3. Resize image to exact multiples of tile_size
    4. Split into uniform tiles
    5. Optionally add thumbnail for global context

    Example:
        Input: 1920x1080 image (aspect ratio ~ 1.78)
        With max_num_tiles=12, selects 4x2 grid (aspect ratio = 2.0)
        Output: 8 tiles of 448x448 + 1 thumbnail = 9 images total
    """

    def __init__(
        self,
        tile_size: int = 448,
        min_num_tiles: int = 1,
        max_num_tiles: int = 12,
        use_thumbnail: bool = True,
    ):
        """
        Initialize dynamic tiling preprocessor.

        Args:
            tile_size: Size of each square tile in pixels (default: 448)
            min_num_tiles: Minimum number of tiles (default: 1)
            max_num_tiles: Maximum number of tiles (default: 12)
            use_thumbnail: Whether to add downsampled thumbnail for global context
        """
        self.tile_size = tile_size
        self.min_num_tiles = min_num_tiles
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail

        # Pre-compute all valid grid configurations for efficiency
        self.target_ratios = self._generate_target_ratios()

    def _generate_target_ratios(self) -> List[Tuple[int, int]]:
        """
        Generate all valid (cols, rows) grid configurations.

        Returns:
            Sorted list of (cols, rows) tuples representing valid grids

        Example with min=1, max=6:
            [(1,1), (1,2), (2,1), (1,3), (3,1), (2,2), (1,4), (4,1),
             (2,3), (3,2), (1,5), (5,1), (1,6), (6,1), (2,3), (3,2)]
        """
        ratios = set(
            (i, j)
            for n in range(self.min_num_tiles, self.max_num_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= self.max_num_tiles and i * j >= self.min_num_tiles
        )

        # Sort by total number of tiles (smallest to largest)
        return sorted(ratios, key=lambda x: x[0] * x[1])

    def find_closest_aspect_ratio(
        self,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int]:
        """
        Find optimal tile grid that best matches image aspect ratio.

        Algorithm:
        1. Calculate aspect ratio difference for each candidate grid
        2. Select grid with minimum difference
        3. If tied, prefer larger grid if image is large enough to benefit

        Args:
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            (cols, rows) tuple representing optimal grid size

        Example:
            Image: 1920x1080 (aspect ratio ~ 1.78)
            Best match: (4,2) grid with aspect ratio = 2.0
            Returns: (4, 2)
        """
        aspect_ratio = image_width / image_height
        area = image_width * image_height

        best_ratio_diff = float("inf")
        best_ratio = (1, 1)

        for cols, rows in self.target_ratios:
            # Calculate this grid's aspect ratio
            target_aspect = cols / rows

            # How different is it from our image's aspect ratio?
            ratio_diff = abs(aspect_ratio - target_aspect)

            # Primary criterion: closest aspect ratio match
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = (cols, rows)

            # Tiebreaker: prefer larger grid if image is large enough
            # This prevents over-tiling small images
            elif ratio_diff == best_ratio_diff:
                min_area = 0.5 * self.tile_size * self.tile_size * cols * rows
                if area > min_area:
                    best_ratio = (cols, rows)

        return best_ratio

    def split_image(
        self, image: PIL.Image.Image
    ) -> Tuple[List[PIL.Image.Image], int, Tuple[int, int]]:
        """
        Split image into tiles based on aspect ratio.

        Pipeline:
        1. Find optimal grid (e.g., 4x2 for 1920x1080)
        2. Resize to exact multiples of tile_size (e.g., 1792x896)
        3. Split into uniform tiles (e.g., 8 tiles of 448x448)
        4. Optionally add thumbnail (full image resized to 448x448)

        Args:
            image: PIL Image to split

        Returns:
            Tuple of:
            - List of 448x448 PIL Image tiles (+ optional thumbnail)
            - Number of tiles (excluding thumbnail)
            - Grid configuration (cols, rows)

        Example:
            Input: 1920x1080 image
            Output: ([tile0, ..., tile7, thumbnail], 8, (4, 2))
        """
        orig_width, orig_height = image.size

        # Find optimal grid configuration
        cols, rows = self.find_closest_aspect_ratio(orig_width, orig_height)

        # Calculate target dimensions (exact multiples of tile_size)
        target_width = self.tile_size * cols
        target_height = self.tile_size * rows
        num_tiles = cols * rows

        # Resize image to target dimensions
        # Use BICUBIC for high-quality resizing
        resized_image = image.resize((target_width, target_height), PIL.Image.BICUBIC)

        # Extract tiles in row-major order (left-to-right, top-to-bottom)
        tiles = []
        for idx in range(num_tiles):
            # Calculate tile position in grid
            col = idx % cols  # Column index (0 to cols-1)
            row = idx // cols  # Row index (0 to rows-1)

            # Define crop box (left, top, right, bottom)
            box = (
                col * self.tile_size,
                row * self.tile_size,
                (col + 1) * self.tile_size,
                (row + 1) * self.tile_size,
            )

            # Extract tile
            tile = resized_image.crop(box)
            tiles.append(tile)

        assert len(tiles) == num_tiles, f"Expected {num_tiles} tiles, got {len(tiles)}"

        # Add thumbnail for global context if multiple tiles
        if self.use_thumbnail and num_tiles > 1:
            thumbnail = image.resize(
                (self.tile_size, self.tile_size), PIL.Image.BICUBIC
            )
            tiles.append(thumbnail)

        return tiles, num_tiles, (cols, rows)

    def get_tile_count(
        self,
        image_width: int,
        image_height: int,
        include_thumbnail: bool = True,
    ) -> int:
        """
        Get number of tiles that will be produced for an image.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            include_thumbnail: Whether to count thumbnail in total

        Returns:
            Total number of tiles (including thumbnail if applicable)
        """
        cols, rows = self.find_closest_aspect_ratio(image_width, image_height)
        num_tiles = cols * rows

        # Add thumbnail if enabled and more than 1 tile
        if include_thumbnail and self.use_thumbnail and num_tiles > 1:
            num_tiles += 1

        return num_tiles

    def get_vision_token_count(
        self,
        image_width: int,
        image_height: int,
        tokens_per_tile: int = 256,
    ) -> int:
        """
        Calculate total vision tokens that will be generated for an image.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            tokens_per_tile: Number of vision tokens per tile (default: 256 for InternViT)

        Returns:
            Total number of vision tokens
        """
        num_tiles = self.get_tile_count(
            image_width, image_height, include_thumbnail=True
        )
        return num_tiles * tokens_per_tile
