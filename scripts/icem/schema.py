"""Shared data objects for the I-CEM research implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def frozen_metadata(values: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class Row:
    row_id: str
    dataset: str
    split: str
    text: str
    label: str | int | None = None
    binary_label: int | None = None
    target_groups: tuple[str, ...] = ()
    rationale_token_mask: tuple[int, ...] | None = None
    functionality: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=frozen_metadata)


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    source: str
    score: float = 1.0
    replacement: str | None = None


@dataclass(frozen=True)
class ImportanceScore:
    token_index: int
    token_text: str
    delta: float
    baseline_score: float
    perturbed_score: float


@dataclass(frozen=True)
class ReleaseResult:
    row_id: str
    variant: str
    source_text: str
    released_text: str
    kept_spans: tuple[Span, ...] = ()
    masked_spans: tuple[Span, ...] = ()
    importance_scores: tuple[ImportanceScore, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=frozen_metadata)
