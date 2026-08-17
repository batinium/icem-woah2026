"""Render reduced release text from kept and masked spans."""

from __future__ import annotations

from collections.abc import Sequence
import re

from .schema import Span
from .spans import merge_spans


def render_release_text(
    text: str,
    kept_spans: Sequence[Span],
    masked_spans: Sequence[Span] = (),
    *,
    omission_marker: str = "...",
) -> str:
    if not kept_spans:
        return "[NO_TASK_EVIDENCE_RETAINED]"

    masked = sorted(masked_spans, key=lambda span: (span.start, span.end))
    pieces: list[str] = []
    cursor: int | None = None
    for span in merge_spans(kept_spans, max_gap=0):
        if cursor is not None and span.start > cursor:
            pieces.append(omission_marker)
        segment = text[span.start : span.end]
        offset = span.start
        overlapping_masks = [
            mask for mask in masked if not (mask.end <= span.start or mask.start >= span.end)
        ]
        for mask in sorted(overlapping_masks, key=lambda item: item.start, reverse=True):
            local_start = max(mask.start, span.start) - offset
            local_end = min(mask.end, span.end) - offset
            replacement = mask.replacement or f"[{mask.label}]"
            segment = f"{segment[:local_start]}{replacement}{segment[local_end:]}"
        pieces.append(segment.strip())
        cursor = span.end
    return cleanup_rendered_text(" ".join(piece for piece in pieces if piece).strip())


def render_masked_text(text: str, masked_spans: Sequence[Span]) -> str:
    """Render the full source text with selected spans replaced."""

    if not text:
        return ""
    return render_release_text(text, (Span(0, len(text), "FULL_TEXT", "render"),), masked_spans)


def cleanup_rendered_text(text: str) -> str:
    """Normalize whitespace introduced by span rendering without paraphrasing."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([,.;:!?%])", r"\1", cleaned)
    cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
    cleaned = cleaned.replace("`` ", '"').replace(" ''", '"')
    cleaned = cleaned.replace(" ' ", "'")
    cleaned = re.sub(r"\s+([\"'])", r" \1", cleaned)
    return cleaned.strip()
