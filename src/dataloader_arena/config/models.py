"""Shared Pydantic base model helpers for configuration parsing."""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="ConfigModel")


class ConfigModel(BaseModel):
    """Base config model with strict mapping validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        populate_by_name=True,
        frozen=True,
    )

    @classmethod
    def parse(cls: type[T], raw: Mapping[str, Any] | None) -> T:
        """Parse from a mapping-like object, defaulting to an empty mapping."""
        return cls.model_validate(raw or {})
