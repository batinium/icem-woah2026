"""Context-extension rules for target, harm, and stance cues."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

from .schema import Span, Token
from .spans import merge_spans


NEGATION_CUES = {"not", "never", "no", "cannot", "can't", "dont", "don't", "doesn't", "isn't", "do", "without"}
QUOTE_CUES = {"said", "says", "say", "saying", "quote", "quoted", "called", "claim", "claimed", "wrote", "posted"}
COUNTERSPEECH_CUES = {
    "condemn",
    "condemned",
    "disagree",
    "opposed",
    "oppose",
    "wrong",
    "reject",
    "rejected",
    "support",
    "agree",
}
TARGET_CUES = {
    "arab",
    "arabs",
    "muslim",
    "muslims",
    "moslem",
    "moslems",
    "jew",
    "jews",
    "jewish",
    "immigrant",
    "immigrants",
    "refugee",
    "refugees",
    "women",
    "woman",
    "black",
    "asian",
    "gay",
    "trans",
    "christian",
    "christians",
    "hindu",
    "hindus",
    "mexican",
    "mexicans",
    "people",
    "white",
    "whites",
}
HARM_CUES = {
    "ban",
    "banned",
    "cuck",
    "cucks",
    "deport",
    "deported",
    "dirty",
    "fag",
    "fags",
    "faggot",
    "faggots",
    "filthy",
    "leave",
    "goyim",
    "vermin",
    "hate",
    "hated",
    "inferior",
    "kike",
    "kikes",
    "kill",
    "killed",
    "kick",
    "kicked",
    "muzzie",
    "muzzies",
    "nigga",
    "niggar",
    "niggars",
    "niggas",
    "nigger",
    "niggers",
    "out",
    "queer",
    "queers",
    "retard",
    "retarded",
    "shithole",
    "shitholes",
    "terrorist",
    "terrorists",
    "threat",
    "threaten",
    "remove",
    "removed",
    "yid",
    "yids",
}


def token_window_span(tokens: Sequence[Token], index: int, *, radius: int) -> Span:
    left = max(0, index - radius)
    right = min(len(tokens) - 1, index + radius)
    return Span(tokens[left].start, tokens[right].end, "EVIDENCE_WINDOW", "context_rule")


def token_range_span(tokens: Sequence[Token], left: int, right: int, label: str) -> Span:
    start = max(0, min(left, right))
    end = min(len(tokens) - 1, max(left, right))
    return Span(tokens[start].start, tokens[end].end, label, "context_rule")


def _normalized_target_terms(target_groups: Sequence[str]) -> set[str]:
    terms = set(TARGET_CUES)
    for target in target_groups:
        normalized = re.sub(r"[_/-]+", " ", str(target).lower()).strip()
        if not normalized or normalized in {"none", "other", "nan"}:
            continue
        terms.add(normalized)
        terms.update(part for part in normalized.split() if len(part) > 2)
    return terms


def cue_label_for_token(token_text: str, target_groups: Sequence[str] = ()) -> str | None:
    lowered = token_text.lower()
    target_terms = _normalized_target_terms(target_groups)
    if lowered in target_terms:
        return "TARGET_CUE"
    if lowered in HARM_CUES:
        return "HARM_CUE"
    if lowered in NEGATION_CUES:
        return "NEGATION_CUE"
    if lowered in QUOTE_CUES or lowered in {'"', "'", "“", "”", "‘", "’"}:
        return "QUOTE_CUE"
    if lowered in COUNTERSPEECH_CUES:
        return "COUNTERSPEECH_CUE"
    return None


def detect_cue_spans(
    text: str,
    tokens: Sequence[Token],
    *,
    target_groups: Sequence[str] = (),
) -> tuple[Span, ...]:
    del text
    spans: list[Span] = []
    for token in tokens:
        label = cue_label_for_token(token.text, target_groups)
        if label:
            spans.append(Span(token.start, token.end, label, "cue_detector"))
    return tuple(spans)


def cue_token_indexes(tokens: Sequence[Token], labels: Iterable[str], target_groups: Sequence[str] = ()) -> tuple[int, ...]:
    wanted = set(labels)
    indexes: list[int] = []
    for token in tokens:
        label = cue_label_for_token(token.text, target_groups)
        if label in wanted:
            indexes.append(token.index)
    return tuple(indexes)


def extend_anchor_indexes(
    tokens: Sequence[Token],
    anchor_indexes: Iterable[int],
    *,
    target_groups: Sequence[str] = (),
    radius: int = 2,
    target_harm_gap: int = 8,
    stance_gap: int = 8,
    merge_gap: int = 12,
) -> tuple[Span, ...]:
    if not tokens:
        return ()

    anchors = tuple(sorted({index for index in anchor_indexes if 0 <= index < len(tokens)}))
    spans = [token_window_span(tokens, index, radius=radius) for index in anchors]
    anchor_set = set(anchors)
    target_indexes = cue_token_indexes(tokens, {"TARGET_CUE"}, target_groups)
    harm_indexes = cue_token_indexes(tokens, {"HARM_CUE"}, target_groups)
    stance_indexes = cue_token_indexes(
        tokens,
        {"NEGATION_CUE", "QUOTE_CUE", "COUNTERSPEECH_CUE"},
        target_groups,
    )

    for anchor in anchors:
        anchor_label = cue_label_for_token(tokens[anchor].text, target_groups)
        if anchor_label in {"TARGET_CUE", "HARM_CUE"}:
            spans.append(Span(tokens[anchor].start, tokens[anchor].end, anchor_label, "context_rule"))

    for target_index in target_indexes:
        nearby_harm = [
            index
            for index in set(harm_indexes) | anchor_set
            if abs(index - target_index) <= target_harm_gap
        ]
        for harm_index in nearby_harm:
            spans.append(token_range_span(tokens, target_index, harm_index, "TARGET_HARM_CONTEXT"))

    for token in tokens:
        label = cue_label_for_token(token.text, target_groups)
        if label not in {"NEGATION_CUE", "QUOTE_CUE", "COUNTERSPEECH_CUE"}:
            continue
        nearby_anchor = [
            index
            for index in set(anchors) | set(harm_indexes)
            if abs(token.index - index) <= stance_gap
        ]
        if nearby_anchor:
            closest = min(nearby_anchor, key=lambda index: abs(token.index - index))
            spans.append(token_range_span(tokens, token.index, closest, "STANCE_HARM_CONTEXT"))

    for stance_index in stance_indexes:
        for target_index in target_indexes:
            if abs(stance_index - target_index) <= stance_gap:
                spans.append(token_range_span(tokens, stance_index, target_index, "STANCE_TARGET_CONTEXT"))

    return merge_spans(spans, max_gap=merge_gap)
