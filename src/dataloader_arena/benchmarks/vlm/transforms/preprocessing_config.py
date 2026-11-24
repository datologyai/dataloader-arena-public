"""Serialization-safe preprocessing config for the shared VLM pipeline.

Shared by Grain and Zephon worker processes.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PreprocessingConfig:
    """Serialization-safe preprocessing config for the shared VLM pipeline.

    Contains only plain values and a tokenizer — no nn.Module references.
    Safe for cloudpickle serialization into Zephon worker processes.
    """

    tokenizer: Any
    tokens_per_image: int
    user_message_template: str
    assistant_message_template: str
    start_sequence: str
    enable_dynamic_tiling: bool = False
    tiling_preprocessor: Optional[Any] = None
    tokens_per_tile: int = 243

    def wrap_user_message(self, content: str) -> str:
        return self.user_message_template.format(content=content)

    def wrap_assistant_message(self, content: str) -> str:
        return self.assistant_message_template.format(content=content)
