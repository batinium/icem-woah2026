"""Occlusion-based token importance for I-CEM."""

from __future__ import annotations

from collections.abc import Sequence

from .classifier import HsdClassifier
from .schema import ImportanceScore, Span, Token
from .spans import token_overlaps_any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
CONTEXT_STOPWORD_EXCEPTIONS = {
    "not",
    "no",
    "never",
    "cannot",
    "can't",
    "do",
    "don't",
    "doesn't",
    "isn't",
    "said",
    "saying",
    "quoted",
    "condemned",
    "wrong",
}


def is_eligible_token(token: Token, excluded_spans: Sequence[Span] = ()) -> bool:
    lowered = token.text.lower()
    if token_overlaps_any(token.start, token.end, excluded_spans):
        return False
    if not token.text.strip() or not any(ch.isalnum() for ch in token.text):
        return False
    if token.text.startswith("[") and token.text.endswith("]"):
        return False
    if lowered in STOPWORDS and lowered not in CONTEXT_STOPWORD_EXCEPTIONS:
        return False
    return True


def occlusion_importance(
    text: str,
    tokens: Sequence[Token],
    classifier: HsdClassifier,
    *,
    replacement: str = "[MASKED]",
    excluded_spans: Sequence[Span] = (),
) -> tuple[ImportanceScore, ...]:
    baseline = classifier.predict_proba([text])[0]
    candidates: list[tuple[Token, str]] = []
    for token in tokens:
        if not is_eligible_token(token, excluded_spans):
            continue
        perturbed = f"{text[:token.start]}{replacement}{text[token.end:]}"
        candidates.append((token, perturbed))
    perturbed_scores = classifier.predict_proba([perturbed for _, perturbed in candidates]) if candidates else []
    scores: list[ImportanceScore] = []
    for (token, _), perturbed_score in zip(candidates, perturbed_scores, strict=True):
        scores.append(
            ImportanceScore(
                token_index=token.index,
                token_text=token.text,
                delta=baseline - perturbed_score,
                baseline_score=baseline,
                perturbed_score=perturbed_score,
            )
        )
    return tuple(scores)


def select_anchor_indexes(
    scores: Sequence[ImportanceScore],
    *,
    top_k: int = 5,
    min_delta: float = 0.02,
    max_anchor_fraction: float = 0.30,
    token_count: int | None = None,
    fallback_to_best: bool = True,
) -> tuple[int, ...]:
    if top_k <= 0:
        return ()
    if token_count:
        top_k = min(top_k, max(1, int(token_count * max_anchor_fraction)))
    all_ranked = sorted(scores, key=lambda score: (-score.delta, score.token_index))
    ranked = [score for score in all_ranked if score.delta >= min_delta]
    if not ranked and fallback_to_best:
        ranked = all_ranked
    return tuple(score.token_index for score in ranked[:top_k])
