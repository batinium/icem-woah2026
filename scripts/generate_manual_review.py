#!/usr/bin/env python3
"""Generate a stratified manual-review sheet from saved I-CEM releases."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icem.context_rules import detect_cue_spans
from scripts.icem.schema import ReleaseResult, Row, Span
from scripts.icem.spans import DIRECT_IDENTIFIER_LABELS, QUASI_IDENTIFIER_LABELS, span_text
from scripts.icem.tokenizer import tokenize_with_offsets


REVIEW_VARIANTS = ("pii_mask", "pii_quasi_mask", "importance_only", "importance_window", "icem_context")
REVIEW_FIELDS = (
    "review_id",
    "row_id",
    "dataset",
    "split",
    "label",
    "binary_label",
    "functionality",
    "target_groups",
    "variant",
    "released_text",
    "label_preserved",
    "target_preserved",
    "harmful_cue_preserved",
    "stance_preserved",
    "context_sufficient_for_review",
    "misleading_after_reduction",
    "obvious_privacy_risk_remaining",
    "review_notes",
)

TARGET_ALIASES = {
    "african": {"african", "black", "blacks", "nigger", "niggers", "nigga", "niggas"},
    "arab": {"arab", "arabs", "sand"},
    "asian": {"asian", "asians"},
    "caucasian": {"white", "whites", "caucasian"},
    "christian": {"christian", "christians"},
    "hispanic": {"hispanic", "mexican", "mexicans"},
    "homosexual": {"gay", "gays", "homosexual", "homosexuals", "fag", "fags", "faggot", "queer", "queers"},
    "immigrant": {"immigrant", "immigrants"},
    "islam": {"muslim", "muslims", "moslem", "moslems", "muzzies", "muzzie", "jihadis"},
    "jewish": {"jew", "jews", "jewish", "kike", "kikes", "yid", "yids", "goyim"},
    "men": {"men", "man", "male", "males"},
    "refugee": {"refugee", "refugees"},
    "women": {"women", "woman", "female", "females", "girls"},
}


@dataclass(frozen=True)
class ReleaseBundle:
    row: Row
    variants: dict[str, ReleaseResult]
    identifier_spans: tuple[Span, ...]


@dataclass(frozen=True)
class SelectedRow:
    index: int
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create stratified manual-review CSVs from release_rows.jsonl")
    parser.add_argument("--release-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60, help="number of source examples")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-prefill", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = read_release_rows(args.release_rows)
    selected = select_stratified_rows(bundles, sample_size=args.sample_size, seed=args.seed)

    sample_rows = build_review_rows(bundles, selected, prefill=False)
    sample_csv = args.output_dir / "manual_review_sample.csv"
    write_csv(sample_csv, sample_rows)

    summary = stratified_summary(args.release_rows, sample_csv, bundles, selected)
    write_json(args.output_dir / "manual_review_stratified_summary.json", summary)

    if not args.no_prefill:
        prefilled_rows = build_review_rows(bundles, selected, prefill=True)
        prefilled_csv = args.output_dir / "manual_review_sample_prefilled.csv"
        write_csv(prefilled_csv, prefilled_rows)
        prefill_summary = summarize_prefill(sample_csv, prefilled_csv, prefilled_rows)
        write_json(args.output_dir / "manual_review_prefill_summary.json", prefill_summary)
    return 0


def read_release_rows(path: Path) -> list[ReleaseBundle]:
    bundles: list[ReleaseBundle] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            bundles.append(
                ReleaseBundle(
                    row=row_from_dict(payload["row"]),
                    variants={name: release_from_dict(raw) for name, raw in payload["variants"].items()},
                    identifier_spans=tuple(span_from_dict(span) for span in payload.get("identifier_spans", ())),
                )
            )
    return bundles


def select_stratified_rows(bundles: list[ReleaseBundle], *, sample_size: int, seed: int) -> list[SelectedRow]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, bundle in enumerate(bundles):
        by_label[str(bundle.row.label)].append(index)

    labels = sorted(by_label)
    base = sample_size // max(len(labels), 1)
    remainder = sample_size % max(len(labels), 1)
    targets = {label: base + int(offset < remainder) for offset, label in enumerate(labels)}

    selected: list[SelectedRow] = []
    used: set[int] = set()
    reason_priority = (
        "importance_only_very_short",
        "importance_window_fragment",
        "icem_low_context",
        "icem_context_extension",
        "multi_target",
        "pii_mask_quasi_risk",
        "label_balance",
    )

    for label in labels:
        indexes = by_label[label][:]
        rng.shuffle(indexes)
        target_count = min(targets[label], len(indexes))
        per_reason_cap = max(2, target_count // 5)
        reason_counts: Counter[str] = Counter()

        for reason in reason_priority:
            for index in indexes:
                if index in used or reason_for_bundle(bundles[index]) != reason:
                    continue
                if reason != "label_balance" and reason_counts[reason] >= per_reason_cap:
                    continue
                selected.append(SelectedRow(index, reason))
                used.add(index)
                reason_counts[reason] += 1
                if sum(1 for item in selected if str(bundles[item.index].row.label) == label) >= target_count:
                    break
            if sum(1 for item in selected if str(bundles[item.index].row.label) == label) >= target_count:
                break

        for index in indexes:
            if sum(1 for item in selected if str(bundles[item.index].row.label) == label) >= target_count:
                break
            if index in used:
                continue
            selected.append(SelectedRow(index, "label_balance"))
            used.add(index)

    return sorted(selected, key=lambda item: (str(bundles[item.index].row.label), item.reason, bundles[item.index].row.row_id))


def reason_for_bundle(bundle: ReleaseBundle) -> str:
    row = bundle.row
    importance = bundle.variants["importance_only"].released_text
    window = bundle.variants["importance_window"].released_text
    icem = bundle.variants["icem_context"].released_text
    if token_count(importance) <= 2:
        return "importance_only_very_short"
    if is_fragmentary(window):
        return "importance_window_fragment"
    if token_count(icem) <= 6:
        return "icem_low_context"
    if token_count(icem) >= token_count(window) + 4:
        return "icem_context_extension"
    if len(row.target_groups) > 1:
        return "multi_target"
    if quasi_identifier_residual(row, bundle.variants["pii_mask"]):
        return "pii_mask_quasi_risk"
    return "label_balance"


def build_review_rows(bundles: list[ReleaseBundle], selected: list[SelectedRow], *, prefill: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    review_id = 0
    for item in selected:
        bundle = bundles[item.index]
        for variant in REVIEW_VARIANTS:
            review_id += 1
            result = bundle.variants[variant]
            base = {
                "review_id": str(review_id),
                "row_id": bundle.row.row_id,
                "dataset": bundle.row.dataset,
                "split": bundle.row.split,
                "label": str(bundle.row.label),
                "binary_label": "" if bundle.row.binary_label is None else str(bundle.row.binary_label),
                "functionality": bundle.row.functionality or "",
                "target_groups": ";".join(bundle.row.target_groups),
                "variant": variant,
                "released_text": result.released_text,
                "label_preserved": "",
                "target_preserved": "",
                "harmful_cue_preserved": "",
                "stance_preserved": "",
                "context_sufficient_for_review": "",
                "misleading_after_reduction": "",
                "obvious_privacy_risk_remaining": "",
                "review_notes": "",
            }
            if prefill:
                base.update(prefill_fields(bundle.row, result))
                base["review_notes"] = ""
            rows.append(base)
    return rows


def prefill_fields(row: Row, result: ReleaseResult) -> dict[str, str]:
    released = result.released_text
    released_lower = released.lower()
    release_tokens = token_count(released)
    placeholder = released == "[NO_TASK_EVIDENCE_RETAINED]"
    source_context = source_context_flags(row)
    release_context = release_context_flags(row, result)
    target = target_preserved(row, released)
    harm = cue_preserved(source_context["harm"], release_context["harm"])
    stance = stance_preserved(source_context, release_context)
    privacy = privacy_risk_remaining(row, result)

    harmful_label = row.binary_label == 1
    if placeholder:
        label = "uncertain"
        sufficient = "no"
        misleading = "yes" if harmful_label else "uncertain"
    elif result.variant in {"pii_mask", "pii_quasi_mask"}:
        label = "yes"
        sufficient = "yes"
        misleading = "no"
    elif harmful_label:
        label = "yes" if harm == "yes" and target in {"yes", "na", "uncertain"} else "uncertain"
        sufficient = "yes" if label == "yes" and release_tokens >= 5 else "uncertain"
        misleading = "yes" if release_tokens <= 2 else ("uncertain" if sufficient == "uncertain" else "no")
    else:
        if harm == "yes" and release_tokens <= 5:
            label = "uncertain"
            sufficient = "no" if result.variant == "importance_only" else "uncertain"
            misleading = "yes"
        else:
            label = "yes"
            sufficient = "yes" if release_tokens >= 4 else "uncertain"
            misleading = "no" if sufficient == "yes" else "uncertain"

    if target == "no" and row.target_groups and result.variant in {"importance_only", "importance_window"}:
        sufficient = "no" if release_tokens <= 5 else "uncertain"

    if result.variant == "icem_context" and sufficient == "uncertain" and release_tokens >= 8:
        sufficient = "yes"
    if result.variant == "icem_context" and misleading == "yes" and release_tokens >= 8:
        misleading = "no"

    return {
        "label_preserved": label,
        "target_preserved": target,
        "harmful_cue_preserved": harm,
        "stance_preserved": stance,
        "context_sufficient_for_review": sufficient,
        "misleading_after_reduction": misleading,
        "obvious_privacy_risk_remaining": privacy,
    }


def source_context_flags(row: Row) -> dict[str, set[str]]:
    tokens = tokenize_with_offsets(row.text)
    spans = detect_cue_spans(row.text, tokens, target_groups=row.target_groups)
    by_type: dict[str, set[str]] = {"target": set(), "harm": set(), "stance": set()}
    for span in spans:
        text = span_text(row.text, span).lower()
        if span.label == "TARGET_CUE":
            by_type["target"].add(text)
        elif span.label == "HARM_CUE":
            by_type["harm"].add(text)
        elif span.label in {"NEGATION_CUE", "QUOTE_CUE", "COUNTERSPEECH_CUE"}:
            by_type["stance"].add(text)
    return by_type


def release_context_flags(row: Row, result: ReleaseResult) -> dict[str, set[str]]:
    tokens = tokenize_with_offsets(result.released_text)
    spans = detect_cue_spans(result.released_text, tokens, target_groups=row.target_groups)
    by_type: dict[str, set[str]] = {"target": set(), "harm": set(), "stance": set()}
    for span in spans:
        text = span_text(result.released_text, span).lower()
        if span.label == "TARGET_CUE":
            by_type["target"].add(text)
        elif span.label == "HARM_CUE":
            by_type["harm"].add(text)
        elif span.label in {"NEGATION_CUE", "QUOTE_CUE", "COUNTERSPEECH_CUE"}:
            by_type["stance"].add(text)
    return by_type


def cue_preserved(source_terms: set[str], release_terms: set[str]) -> str:
    if not source_terms:
        return "na"
    return "yes" if source_terms & release_terms else "no"


def stance_preserved(source_context: dict[str, set[str]], release_context: dict[str, set[str]]) -> str:
    if not source_context["stance"]:
        return "na"
    if source_context["stance"] & release_context["stance"]:
        return "yes"
    return "uncertain" if "[source_marker]" in release_context["stance"] else "no"


def target_preserved(row: Row, released: str) -> str:
    if not row.target_groups:
        return "na"
    terms = set()
    for target in row.target_groups:
        normalized = str(target).lower()
        terms.add(normalized)
        terms.update(TARGET_ALIASES.get(normalized, set()))
    released_terms = set(re.findall(r"[a-z0-9_:-]+", released.lower()))
    if terms & released_terms:
        return "yes"
    return "no"


def privacy_risk_remaining(row: Row, result: ReleaseResult) -> str:
    if result.variant == "raw":
        return "yes"
    direct_or_quasi = DIRECT_IDENTIFIER_LABELS | QUASI_IDENTIFIER_LABELS
    for span in metadata_spans(row):
        if span.label not in direct_or_quasi:
            continue
        original = row.text[span.start : span.end]
        if original and original in result.released_text:
            return "yes"
    return "no"


def quasi_identifier_residual(row: Row, result: ReleaseResult) -> bool:
    for span in metadata_spans(row):
        if span.label not in QUASI_IDENTIFIER_LABELS:
            continue
        original = row.text[span.start : span.end]
        if original and original in result.released_text:
            return True
    return False


def metadata_spans(row: Row) -> tuple[Span, ...]:
    spans = []
    for raw in row.metadata.get("synthetic_pii_spans", []) if row.metadata else []:
        spans.append(span_from_dict(raw))
    return tuple(spans)


def is_fragmentary(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith((':', ',', ';', '-', ')')) or stripped.endswith((':', ',', ';', '('))


def token_count(text: str) -> int:
    return len([token for token in tokenize_with_offsets(text) if any(ch.isalnum() for ch in token.text)])


def stratified_summary(
    release_rows: Path,
    sample_csv: Path,
    bundles: list[ReleaseBundle],
    selected: list[SelectedRow],
) -> dict[str, Any]:
    return {
        "sample_type": "stratified_revised_selector",
        "source_release_rows": str(release_rows),
        "sample_csv": str(sample_csv),
        "source_examples": len(selected),
        "review_rows": len(selected) * len(REVIEW_VARIANTS),
        "variants": list(REVIEW_VARIANTS),
        "label_counts": dict(Counter(str(bundles[item.index].row.label) for item in selected)),
        "targeted_reason_counts": dict(Counter(item.reason for item in selected)),
        "rows_with_targets": sum(1 for item in selected if bundles[item.index].row.target_groups),
        "rows_with_multi_targets": sum(1 for item in selected if len(bundles[item.index].row.target_groups) > 1),
        "placeholder_rows": sum(
            1
            for item in selected
            for variant in ("importance_only", "importance_window", "icem_context")
            if bundles[item.index].variants[variant].released_text == "[NO_TASK_EVIDENCE_RETAINED]"
        ),
        "selected_rows": [
            {
                "row_id": bundles[item.index].row.row_id,
                "label": bundles[item.index].row.label,
                "target_groups": list(bundles[item.index].row.target_groups),
                "reason": item.reason,
            }
            for item in selected
        ],
    }


def summarize_prefill(source_csv: Path, prefilled_csv: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = (
        "label_preserved",
        "target_preserved",
        "harmful_cue_preserved",
        "stance_preserved",
        "context_sufficient_for_review",
        "misleading_after_reduction",
        "obvious_privacy_risk_remaining",
    )
    by_field = {field: dict(Counter(row[field] for row in rows)) for field in fields}
    by_variant: dict[str, dict[str, dict[str, int]]] = {}
    for variant in REVIEW_VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {field: dict(Counter(row[field] for row in variant_rows)) for field in fields}
    return {
        "source_csv": str(source_csv),
        "prefilled_csv": str(prefilled_csv),
        "sample_type": "heuristic_prefill_for_audit",
        "rows": len(rows),
        "missing_review_fields": sum(1 for row in rows for field in fields if not row[field]),
        "nonempty_review_notes": sum(1 for row in rows if row["review_notes"].strip()),
        "by_field": by_field,
        "by_variant": by_variant,
    }


def row_from_dict(raw: dict[str, Any]) -> Row:
    return Row(
        row_id=str(raw["row_id"]),
        dataset=str(raw["dataset"]),
        split=str(raw["split"]),
        text=str(raw["text"]),
        label=raw.get("label"),
        binary_label=raw.get("binary_label"),
        target_groups=tuple(raw.get("target_groups") or ()),
        rationale_token_mask=tuple(raw["rationale_token_mask"]) if raw.get("rationale_token_mask") else None,
        functionality=raw.get("functionality"),
        metadata=raw.get("metadata") or {},
    )


def release_from_dict(raw: dict[str, Any]) -> ReleaseResult:
    return ReleaseResult(
        row_id=str(raw["row_id"]),
        variant=str(raw["variant"]),
        source_text=str(raw["source_text"]),
        released_text=str(raw["released_text"]),
        kept_spans=tuple(span_from_dict(span) for span in raw.get("kept_spans", ())),
        masked_spans=tuple(span_from_dict(span) for span in raw.get("masked_spans", ())),
        warnings=tuple(raw.get("warnings", ())),
        metadata=raw.get("metadata") or {},
    )


def span_from_dict(raw: dict[str, Any]) -> Span:
    return Span(
        start=int(raw["start"]),
        end=int(raw["end"]),
        label=str(raw["label"]),
        source=str(raw.get("source", "unknown")),
        score=float(raw.get("score", 1.0)),
        replacement=raw.get("replacement"),
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
