"""
Data models for VLM training pipeline.

Contains pydantic models for conversations and messages used throughout the data pipeline.
"""

from typing import List, Optional

import PIL.Image
from pydantic import BaseModel


class Message(BaseModel):
    """Represents a single message in a conversation."""

    role: str
    content: str
    token_ids: Optional[List[int]] = None


class Conversation(BaseModel):
    """Represents a complete conversation with messages and optional images."""

    model_config = {"arbitrary_types_allowed": True}

    messages: List[Message]
    images: Optional[List[PIL.Image.Image]] = None
    length: Optional[int] = None
    tile_counts: Optional[List[int]] = None  # Track tiles per image for dynamic tiling
