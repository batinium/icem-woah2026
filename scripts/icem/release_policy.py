"""Variant rendering policies for the I-CEM ablation study."""

from __future__ import annotations

from collections.abc import Sequence

from .context_rules import extend_anchor_indexes
from .render import render_masked_text, render_release_text
from .schema import ImportanceScore, ReleaseResult, Span, Token
from .spans import (
    direct_and_quasi_identifier_spans,
    direct_identifier_spans,
    merge_spans,
)


DEFAULT_VARIANTS = (
    "raw",
    "pii_mask",
    "pii_quasi_mask",
    "importance_only",
    "importance_window",
    "icem_context",
)


def full_text_span(text: str) -> tuple[Span, ...]:
    return (Span(0, len(text), "FULL_TEXT", "release_policy"),) if text else ()


def token_spans(tokens: Sequence[Token], indexes: Sequence[int], *, label: str = "EVIDENCE_TOKEN") -> tuple[Span, ...]:
    spans = [
        Span(tokens[index].start, tokens[index].end, label, "importance")
        for index in sorted(set(indexes))
        if 0 <= index < len(tokens)
    ]
    return tuple(spans)


def token_window_spans(
    tokens: Sequence[Token],
    indexes: Sequence[int],
    *,
    radius: int = 2,
    label: str = "EVIDENCE_WINDOW",
) -> tuple[Span, ...]:
    spans: list[Span] = []
    for index in sorted(set(indexes)):
        if not 0 <= index < len(tokens):
            continue
        left = max(0, index - radius)
        right = min(len(tokens) - 1, index + radius)
        spans.append(Span(tokens[left].start, tokens[right].end, label, "importance_window"))
    return merge_spans(spans, max_gap=1)


def release_raw(*, row_id: str, text: str) -> ReleaseResult:
    return ReleaseResult(
        row_id=row_id,
        variant="raw",
        source_text=text,
        released_text=text,
        kept_spans=full_text_span(text),
    )


def release_pii_mask(
    *,
    row_id: str,
    text: str,
    identifier_spans: Sequence[Span],
    include_quasi: bool = False,
) -> ReleaseResult:
    variant = "pii_quasi_mask" if include_quasi else "pii_mask"
    masked_spans = (
        direct_and_quasi_identifier_spans(identifier_spans)
        if include_quasi
        else direct_identifier_spans(identifier_spans)
    )
    return ReleaseResult(
        row_id=row_id,
        variant=variant,
        source_text=text,
        released_text=render_masked_text(text, masked_spans),
        kept_spans=full_text_span(text),
        masked_spans=tuple(masked_spans),
    )


def release_importance_only(
    *,
    row_id: str,
    text: str,
    tokens: Sequence[Token],
    anchor_indexes: Sequence[int],
    masked_spans: Sequence[Span] = (),
    importance_scores: Sequence[ImportanceScore] = (),
) -> ReleaseResult:
    kept_spans = token_spans(tokens, anchor_indexes)
    return ReleaseResult(
        row_id=row_id,
        variant="importance_only",
        source_text=text,
        released_text=render_release_text(text, kept_spans, masked_spans),
        kept_spans=kept_spans,
        masked_spans=tuple(masked_spans),
        importance_scores=tuple(importance_scores),
    )


def release_importance_window(
    *,
    row_id: str,
    text: str,
    tokens: Sequence[Token],
    anchor_indexes: Sequence[int],
    masked_spans: Sequence[Span] = (),
    window_radius: int = 2,
    importance_scores: Sequence[ImportanceScore] = (),
) -> ReleaseResult:
    kept_spans = token_window_spans(tokens, anchor_indexes, radius=window_radius)
    return ReleaseResult(
        row_id=row_id,
        variant="importance_window",
        source_text=text,
        released_text=render_release_text(text, kept_spans, masked_spans),
        kept_spans=kept_spans,
        masked_spans=tuple(masked_spans),
        importance_scores=tuple(importance_scores),
    )


def release_icem_context(
    *,
    row_id: str,
    text: str,
    tokens: Sequence[Token],
    anchor_indexes: Sequence[int],
    masked_spans: Sequence[Span] = (),
    window_radius: int = 2,
    target_groups: Sequence[str] = (),
    target_harm_max_gap: int = 8,
    stance_harm_max_gap: int = 8,
    importance_scores: Sequence[ImportanceScore] = (),
) -> ReleaseResult:
    kept_spans = extend_anchor_indexes(
        tokens,
        anchor_indexes,
        target_groups=target_groups,
        radius=window_radius,
        target_harm_gap=target_harm_max_gap,
        stance_gap=stance_harm_max_gap,
    )
    released = render_release_text(text, kept_spans, masked_spans)
    return ReleaseResult(
        row_id=row_id,
        variant="icem_context",
        source_text=text,
        released_text=released,
        kept_spans=kept_spans,
        masked_spans=tuple(masked_spans),
        importance_scores=tuple(importance_scores),
    )


def release_variant(
    *,
    variant: str,
    row_id: str,
    text: str,
    tokens: Sequence[Token],
    anchor_indexes: Sequence[int],
    identifier_spans: Sequence[Span],
    window_radius: int = 2,
    target_groups: Sequence[str] = (),
    target_harm_max_gap: int = 8,
    stance_harm_max_gap: int = 8,
    importance_scores: Sequence[ImportanceScore] = (),
) -> ReleaseResult:
    if variant == "raw":
        return release_raw(row_id=row_id, text=text)
    if variant == "pii_mask":
        return release_pii_mask(row_id=row_id, text=text, identifier_spans=identifier_spans)
    if variant == "pii_quasi_mask":
        return release_pii_mask(
            row_id=row_id,
            text=text,
            identifier_spans=identifier_spans,
            include_quasi=True,
        )

    masked_spans = direct_and_quasi_identifier_spans(identifier_spans)
    if variant == "importance_only":
        return release_importance_only(
            row_id=row_id,
            text=text,
            tokens=tokens,
            anchor_indexes=anchor_indexes,
            masked_spans=masked_spans,
            importance_scores=importance_scores,
        )
    if variant == "importance_window":
        return release_importance_window(
            row_id=row_id,
            text=text,
            tokens=tokens,
            anchor_indexes=anchor_indexes,
            masked_spans=masked_spans,
            window_radius=window_radius,
            importance_scores=importance_scores,
        )
    if variant == "icem_context":
        return release_icem_context(
            row_id=row_id,
            text=text,
            tokens=tokens,
            anchor_indexes=anchor_indexes,
            masked_spans=masked_spans,
            window_radius=window_radius,
            target_groups=target_groups,
            target_harm_max_gap=target_harm_max_gap,
            stance_harm_max_gap=stance_harm_max_gap,
            importance_scores=importance_scores,
        )
    raise ValueError(f"unknown release variant: {variant}")
