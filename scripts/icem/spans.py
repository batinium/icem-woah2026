"""Span utilities for I-CEM release decisions."""

from __future__ import annotations

from collections.abc import Iterable
import re

from .schema import Span
from .synthetic_pii import (
    DATES,
    EMAILS,
    EVENTS,
    FAMILY_RELATIONS,
    HANDLES,
    LOCATIONS,
    PEOPLE,
    PHONES,
    SCHOOLS,
    WORKPLACES,
)


DIRECT_IDENTIFIER_LABELS = frozenset({"PERSON", "HANDLE", "EMAIL", "PHONE", "URL", "IP"})
QUASI_IDENTIFIER_LABELS = frozenset(
    {
        "DATE",
        "AGE",
        "LOCATION",
        "SCHOOL",
        "WORKPLACE",
        "ORG",
        "EVENT",
        "FAMILY_RELATION",
        "SOURCE_MARKER",
        "STYLE_MARKER",
    }
)

_REGEX_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>()]+")),
    ("HANDLE", re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,32}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d[\d().\-\s]{6,}\d|555-\d{4})\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (
        "DATE",
        re.compile(
            r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}|"
            r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
            re.IGNORECASE,
        ),
    ),
    ("AGE", re.compile(r"\b(?:age\s*)?\d{1,2}\s*(?:years old|year-old|yo|y/o)\b", re.IGNORECASE)),
    (
        "FAMILY_RELATION",
        re.compile(r"\b(?:cousin|older brother|aunt|uncle|mother|father|sister|brother)\b", re.IGNORECASE),
    ),
    ("SOURCE_MARKER", re.compile(r"\b(?:posted|wrote|shared by|contact)\b", re.IGNORECASE)),
)

_PHRASE_DETECTORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PERSON", PEOPLE),
    ("HANDLE", HANDLES),
    ("EMAIL", EMAILS),
    ("PHONE", PHONES),
    ("LOCATION", LOCATIONS),
    ("SCHOOL", SCHOOLS),
    ("WORKPLACE", WORKPLACES),
    ("DATE", DATES),
    ("FAMILY_RELATION", FAMILY_RELATIONS),
    ("EVENT", EVENTS),
)


def overlaps(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def contains(span: Span, offset: int) -> bool:
    return span.start <= offset < span.end


def merge_spans(spans: Iterable[Span], *, max_gap: int = 0) -> tuple[Span, ...]:
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    if not ordered:
        return ()
    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end + max_gap:
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, span.end),
                label=last.label if last.label == span.label else "MERGED",
                source=last.source if last.source == span.source else "merged",
                score=max(last.score, span.score),
                replacement=last.replacement if last.replacement == span.replacement else None,
            )
        else:
            merged.append(span)
    return tuple(merged)


def span_text(text: str, span: Span) -> str:
    return text[span.start : span.end]


def span_replacement(label: str) -> str:
    if label in DIRECT_IDENTIFIER_LABELS | QUASI_IDENTIFIER_LABELS:
        return f"[{label}]"
    return "[IDENTIFIER]"


def _phrase_spans(text: str, phrase: str, label: str) -> Iterable[Span]:
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
    for match in pattern.finditer(text):
        yield Span(match.start(), match.end(), label, "detector", replacement=span_replacement(label))


def resolve_overlapping_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Return non-overlapping spans, preferring gold/high-score/longer spans."""

    ranked = sorted(spans, key=lambda span: (span.start, -_span_rank(span)[0], -span.score, -(span.end - span.start)))
    kept: list[Span] = []
    for span in ranked:
        replacement = span.replacement or span_replacement(span.label)
        candidate = Span(span.start, span.end, span.label, span.source, span.score, replacement)
        conflicting = [index for index, existing in enumerate(kept) if overlaps(candidate, existing)]
        if not conflicting:
            kept.append(candidate)
            continue
        if all(_span_rank(candidate) > _span_rank(kept[index]) for index in conflicting):
            for index in reversed(conflicting):
                kept.pop(index)
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda span: (span.start, span.end)))


def _span_rank(span: Span) -> tuple[float, float, int]:
    return (1.0 if span.source == "synthetic" else 0.0, span.score, span.end - span.start)


def detect_identifier_spans(text: str, gold_spans: Iterable[Span] = ()) -> tuple[Span, ...]:
    """Detect direct and quasi identifiers with deterministic rules.

    This is intentionally small and tuned to the synthetic injection layer. It
    supports the release ablation and residual-risk proxy; it is not presented
    as a general-purpose PII recognizer.
    """

    spans: list[Span] = [
        Span(span.start, span.end, span.label, span.source, span.score, span.replacement or span_replacement(span.label))
        for span in gold_spans
    ]
    for label, pattern in _REGEX_DETECTORS:
        score = 1.2 if label == "DATE" else 1.0
        spans.extend(
            Span(match.start(), match.end(), label, "detector", score=score, replacement=span_replacement(label))
            for match in pattern.finditer(text)
        )
    for label, phrases in _PHRASE_DETECTORS:
        for phrase in phrases:
            spans.extend(_phrase_spans(text, phrase, label))
    return resolve_overlapping_spans(spans)


def filter_spans_by_labels(spans: Iterable[Span], labels: Iterable[str]) -> tuple[Span, ...]:
    wanted = set(labels)
    return tuple(span for span in spans if span.label in wanted)


def direct_identifier_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    return filter_spans_by_labels(spans, DIRECT_IDENTIFIER_LABELS)


def direct_and_quasi_identifier_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    return filter_spans_by_labels(spans, DIRECT_IDENTIFIER_LABELS | QUASI_IDENTIFIER_LABELS)


def token_overlaps_any(start: int, end: int, spans: Iterable[Span]) -> bool:
    token_span = Span(start, end, "TOKEN", "token")
    return any(overlaps(token_span, span) for span in spans)
