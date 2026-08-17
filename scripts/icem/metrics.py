"""Aggregate metrics for I-CEM experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

from .context_rules import detect_cue_spans
from .schema import ReleaseResult, Row, Span
from .spans import (
    DIRECT_IDENTIFIER_LABELS,
    QUASI_IDENTIFIER_LABELS,
    detect_identifier_spans,
    merge_spans,
    overlaps,
    span_text,
)
from .tokenizer import tokenize_with_offsets


def retained_character_ratio(source: str, released: str) -> float:
    if not source:
        return 0.0
    return round(len(released) / len(source), 6)


def retained_token_ratio(source_tokens: int, released_tokens: int) -> float:
    if source_tokens <= 0:
        return 0.0
    return round(released_tokens / source_tokens, 6)


def aggregate_variant_metrics(
    rows: Sequence[Row],
    results: Sequence[ReleaseResult],
    *,
    released_scores: Sequence[float],
    raw_scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    if len(rows) != len(results):
        raise ValueError("rows and results must have the same length")
    if len(released_scores) != len(results) or len(raw_scores) != len(results):
        raise ValueError("score lengths must match results")

    gold = [row.binary_label for row in rows]
    raw_predictions = [1 if score >= threshold else 0 for score in raw_scores]
    predictions = [1 if score >= threshold else 0 for score in released_scores]
    utility = _classification_metrics(gold, predictions)

    token_ratios: list[float] = []
    char_ratios: list[float] = []
    output_token_counts: list[int] = []
    ngram3: list[float] = []
    ngram5: list[float] = []
    changed = 0
    placeholder_only = 0
    direct_gold_total = 0
    direct_gold_residual = 0
    quasi_gold_total = 0
    quasi_gold_residual = 0
    detector_direct_residual = 0
    detector_quasi_residual = 0
    detector_source_markers = 0
    target_scores: list[int] = []
    harm_scores: list[int] = []
    negation_scores: list[int] = []
    quote_scores: list[int] = []
    counter_scores: list[int] = []
    target_harm_scores: list[int] = []
    stance_harm_scores: list[int] = []
    rationale_overlaps: list[float] = []

    for row, result, raw_prediction, prediction in zip(rows, results, raw_predictions, predictions, strict=True):
        source_tokens = tokenize_with_offsets(result.source_text)
        released_tokens = tokenize_with_offsets(result.released_text)
        token_ratios.append(_retained_source_token_count(source_tokens, result.kept_spans) / max(len(source_tokens), 1))
        char_ratios.append(_retained_source_char_count(result.source_text, result.kept_spans) / max(len(result.source_text), 1))
        output_token_counts.append(len(released_tokens))
        ngram3.append(_ngram_retention(source_tokens, released_tokens, 3))
        ngram5.append(_ngram_retention(source_tokens, released_tokens, 5))
        changed += int(result.released_text != result.source_text)
        placeholder_only += int(result.released_text == "[NO_TASK_EVIDENCE_RETAINED]")
        detector_spans = detect_identifier_spans(result.released_text)
        detector_direct_residual += sum(1 for span in detector_spans if span.label in DIRECT_IDENTIFIER_LABELS)
        detector_quasi_residual += sum(1 for span in detector_spans if span.label in QUASI_IDENTIFIER_LABELS)
        detector_source_markers += sum(1 for span in detector_spans if span.label == "SOURCE_MARKER")

        for gold_span in _metadata_spans(row.metadata):
            original = row.text[gold_span.start : gold_span.end]
            residual = bool(original and original in result.released_text)
            if gold_span.label in DIRECT_IDENTIFIER_LABELS:
                direct_gold_total += 1
                direct_gold_residual += int(residual)
            if gold_span.label in QUASI_IDENTIFIER_LABELS:
                quasi_gold_total += 1
                quasi_gold_residual += int(residual)

        cue_scores = _context_scores(row, result)
        for key, value in cue_scores.items():
            if value is None:
                continue
            if key == "target":
                target_scores.append(value)
            elif key == "harm":
                harm_scores.append(value)
            elif key == "negation":
                negation_scores.append(value)
            elif key == "quote":
                quote_scores.append(value)
            elif key == "counter":
                counter_scores.append(value)
            elif key == "target_harm":
                target_harm_scores.append(value)
            elif key == "stance_harm":
                stance_harm_scores.append(value)
        rationale = _rationale_overlap(row, source_tokens, result.kept_spans)
        if rationale is not None:
            rationale_overlaps.append(rationale)

    flip_rate = _mean(int(raw != pred) for raw, pred in zip(raw_predictions, predictions, strict=True))
    metrics = {
        **utility,
        "prediction_flip_rate_raw": round(flip_rate, 6),
        "retained_token_pct": _pct(_mean(token_ratios)),
        "retained_char_pct": _pct(_mean(char_ratios)),
        "mean_output_tokens": round(_mean(output_token_counts), 3),
        "median_output_tokens": round(float(median(output_token_counts)) if output_token_counts else 0.0, 3),
        "changed_row_pct": _pct(changed / max(len(results), 1)),
        "placeholder_only_count": placeholder_only,
        "unique_3gram_retention_pct": _pct(_mean(ngram3)),
        "unique_5gram_retention_pct": _pct(_mean(ngram5)),
        "direct_pii_gold_total": direct_gold_total,
        "direct_pii_residual_count": direct_gold_residual,
        "direct_pii_residual_rate": round(direct_gold_residual / direct_gold_total, 6) if direct_gold_total else 0.0,
        "direct_pii_removed_recall": round(1.0 - direct_gold_residual / direct_gold_total, 6) if direct_gold_total else 0.0,
        "quasi_identifier_gold_total": quasi_gold_total,
        "quasi_identifier_residual_count": quasi_gold_residual,
        "quasi_identifier_residual_rate": round(quasi_gold_residual / quasi_gold_total, 6) if quasi_gold_total else 0.0,
        "detected_direct_identifier_residual_count": detector_direct_residual,
        "detected_quasi_identifier_residual_count": detector_quasi_residual,
        "detected_source_marker_residual_count": detector_source_markers,
        "target_cue_preservation": _mean_or_none(target_scores),
        "harm_cue_preservation": _mean_or_none(harm_scores),
        "negation_cue_preservation": _mean_or_none(negation_scores),
        "quotation_cue_preservation": _mean_or_none(quote_scores),
        "counterspeech_cue_preservation": _mean_or_none(counter_scores),
        "target_harm_pair_preservation": _mean_or_none(target_harm_scores),
        "stance_harm_relation_preservation": _mean_or_none(stance_harm_scores),
        "rationale_overlap": _mean_or_none(rationale_overlaps),
        "row_count": len(results),
    }
    return metrics


def _classification_metrics(gold: Sequence[int | None], predictions: Sequence[int]) -> dict[str, Any]:
    paired = [(int(label), pred) for label, pred in zip(gold, predictions, strict=True) if label in {0, 1}]
    if not paired:
        return {
            "accuracy": None,
            "macro_f1": None,
            "positive_precision": None,
            "positive_recall": None,
            "positive_f1": None,
        }
    labels = [label for label, _ in paired]
    preds = [pred for _, pred in paired]
    accuracy = sum(int(label == pred) for label, pred in paired) / len(paired)
    positive = _precision_recall_f1(labels, preds, positive=1)
    negative = _precision_recall_f1(labels, preds, positive=0)
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round((positive["f1"] + negative["f1"]) / 2, 6),
        "positive_precision": round(positive["precision"], 6),
        "positive_recall": round(positive["recall"], 6),
        "positive_f1": round(positive["f1"], 6),
    }


def _precision_recall_f1(labels: Sequence[int], preds: Sequence[int], *, positive: int) -> dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, preds, strict=True) if label == positive and pred == positive)
    fp = sum(1 for label, pred in zip(labels, preds, strict=True) if label != positive and pred == positive)
    fn = sum(1 for label, pred in zip(labels, preds, strict=True) if label == positive and pred != positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _retained_source_token_count(tokens: Sequence[Any], kept_spans: Sequence[Span]) -> int:
    return sum(1 for token in tokens if any(overlaps(Span(token.start, token.end, "TOKEN", "metrics"), span) for span in kept_spans))


def _retained_source_char_count(source: str, kept_spans: Sequence[Span]) -> int:
    del source
    return sum(max(0, span.end - span.start) for span in merge_spans(kept_spans, max_gap=0))


def _ngram_retention(source_tokens: Sequence[Any], released_tokens: Sequence[Any], n: int) -> float:
    source_ngrams = _ngrams([token.text.lower() for token in source_tokens if any(ch.isalnum() for ch in token.text)], n)
    if not source_ngrams:
        return 0.0
    released_ngrams = _ngrams([token.text.lower() for token in released_tokens if any(ch.isalnum() for ch in token.text)], n)
    return len(source_ngrams & released_ngrams) / len(source_ngrams)


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _metadata_spans(metadata: Mapping[str, Any]) -> tuple[Span, ...]:
    spans = []
    for raw in metadata.get("synthetic_pii_spans", []) if metadata else []:
        if isinstance(raw, Span):
            spans.append(raw)
        elif isinstance(raw, dict):
            spans.append(
                Span(
                    int(raw["start"]),
                    int(raw["end"]),
                    str(raw["label"]),
                    str(raw.get("source", "synthetic")),
                    float(raw.get("score", 1.0)),
                    raw.get("replacement"),
                )
            )
    return tuple(spans)


def _context_scores(row: Row, result: ReleaseResult) -> dict[str, int | None]:
    source_tokens = tokenize_with_offsets(result.source_text)
    source_cues = detect_cue_spans(result.source_text, source_tokens, target_groups=row.target_groups)
    release_lower = result.released_text.lower()

    def preserved(labels: set[str]) -> int | None:
        relevant = [span for span in source_cues if span.label in labels]
        if not relevant:
            return None
        return int(any(span_text(result.source_text, span).lower() in release_lower for span in relevant))

    target = preserved({"TARGET_CUE"})
    harm = preserved({"HARM_CUE"})
    negation = preserved({"NEGATION_CUE"})
    quote = preserved({"QUOTE_CUE"})
    counter = preserved({"COUNTERSPEECH_CUE"})
    target_harm = int(bool(target and harm)) if target is not None and harm is not None else None
    stance_any = max(value for value in (negation, quote, counter) if value is not None) if any(
        value is not None for value in (negation, quote, counter)
    ) else None
    stance_harm = int(bool(stance_any and harm)) if stance_any is not None and harm is not None else None
    return {
        "target": target,
        "harm": harm,
        "negation": negation,
        "quote": quote,
        "counter": counter,
        "target_harm": target_harm,
        "stance_harm": stance_harm,
    }


def _rationale_overlap(row: Row, tokens: Sequence[Any], kept_spans: Sequence[Span]) -> float | None:
    if not row.rationale_token_mask:
        return None
    rationale_indexes = [
        index for index, value in enumerate(row.rationale_token_mask) if value and index < len(tokens)
    ]
    if not rationale_indexes:
        return None
    retained = 0
    for index in rationale_indexes:
        token = tokens[index]
        token_span = Span(token.start, token.end, "TOKEN", "metrics")
        retained += int(any(overlaps(token_span, span) for span in kept_spans))
    return retained / len(rationale_indexes)


def _mean(values: Iterable[float | int]) -> float:
    items = list(values)
    return sum(float(value) for value in items) / len(items) if items else 0.0


def _mean_or_none(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return round(_mean(values), 6)


def _pct(value: float) -> float:
    return round(value * 100.0, 3)
